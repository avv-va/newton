# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase
from .kernels import (
    apply_particle_deltas,
    solve_particle_particle_contacts,
    solve_particle_shape_contacts,
    solve_shape_matching_batch_tiled,
)


@wp.kernel
def calculate_group_particle_mass(
    particle_mass: wp.array[float],
    group_particle_start: wp.array[wp.int32],
    group_particle_count: wp.array[wp.int32],
    group_particles_flat: wp.array[wp.int32],
    total_group_mass: wp.array[float],
):
    group_id = wp.tid()
    start_idx = group_particle_start[group_id]
    num_particles = group_particle_count[group_id]
    total = wp.float32(0.0)
    for p in range(num_particles):
        idx = group_particles_flat[start_idx + p]
        total += particle_mass[idx]
    total_group_mass[group_id] = total


class SolverBXPBD(SolverBase):
    """
    Similar to SolverXPBD. Only includes contact handling + shape matching constraints.
    This solver assumes complete rigid bodies. No soft bodies.
    Rigid bodies are modeled as a collection of particles.
    """

    def __init__(
        self,
        model: Model,
        iterations: int = 2,
        soft_body_relaxation: float = 0.9,
        soft_contact_relaxation: float = 0.9,
        rigid_contact_relaxation: float = 0.8,
        rigid_contact_con_weighting: bool = True,
        enable_restitution: bool = False,
    ):
        super().__init__(model=model)
        self.iterations = iterations

        self.soft_body_relaxation = soft_body_relaxation
        self.soft_contact_relaxation = soft_contact_relaxation
        self.rigid_contact_relaxation = rigid_contact_relaxation
        self.rigid_contact_con_weighting = rigid_contact_con_weighting
        self.enable_restitution = enable_restitution

        # helper variables to track constraint resolution vars
        self._particle_delta_counter = 0
        self.particle_q_rest = wp.clone(model.particle_q)

        self._particle_particle_enabled = bool(
            model.particle_count
            and model.particle_grid is not None
            and model.particle_max_radius > 0.0
            and model.particle_group_count > 1
        )
        if self._particle_particle_enabled:
            with wp.ScopedDevice(model.device):
                model.particle_grid.reserve(model.particle_count)

        # Precompute shape matching data for all dynamic groups
        self._dynamic_group_ids = []
        self._group_particle_start = []
        self._group_particle_count = []
        self._group_particles_flat = []
        self._num_dynamic_groups = 0

        if model.particle_count > 0 and model.particle_group_count > 0:
            group = model.particle_group.numpy()
            mass = model.particle_mass.numpy()

            grouped = np.flatnonzero(group >= 0).astype(np.int32)
            if grouped.size > 0:
                gid = group[grouped]

                # A group is dynamic if any of its particles has mass; static groups are
                # excluded from shape matching entirely.
                num_groups = model.particle_group_count
                has_mass = np.bincount(gid, weights=(mass[grouped] > 0.0), minlength=num_groups) > 0.0
                dynamic_ids = np.flatnonzero(has_mass).astype(np.int32)

                if dynamic_ids.size > 0:
                    # Stable sort keeps groups in ascending id and particles in ascending
                    # index within a group, matching the order add_particle_volume assigns.
                    order = np.argsort(gid, kind="stable")
                    flat = grouped[order][np.isin(gid[order], dynamic_ids)]
                    counts = np.bincount(gid, minlength=num_groups)[dynamic_ids].astype(np.int32)
                    starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int32)

                    self._dynamic_group_ids = wp.array(dynamic_ids, dtype=wp.int32, device=model.device)
                    self._group_particle_start = wp.array(starts, dtype=wp.int32, device=model.device)
                    self._group_particle_count = wp.array(counts, dtype=wp.int32, device=model.device)
                    self._group_particles_flat = wp.array(flat, dtype=wp.int32, device=model.device)
                    self._num_dynamic_groups = int(dynamic_ids.size)

        if self._num_dynamic_groups:
            # Compute block_dim for tiled shape matching: cap at 256, round to warp size (32)
            max_particles = max(self._group_particle_count.numpy())
            self._shape_match_block_dim = min(256, int(max_particles))
            # Round up to nearest multiple of 32 (warp size)
            self._shape_match_block_dim = max(32, ((self._shape_match_block_dim + 31) // 32) * 32)

            self.total_group_mass = wp.zeros(self._num_dynamic_groups, dtype=wp.float32, device=model.device)
            wp.launch(
                kernel=calculate_group_particle_mass,
                dim=self._num_dynamic_groups,
                inputs=[
                    model.particle_mass,
                    self._group_particle_start,
                    self._group_particle_count,
                    self._group_particles_flat,
                ],
                outputs=[self.total_group_mass],
                device=model.device,
            )

    def apply_particle_deltas(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        particle_deltas: wp.array,
        dt: float,
    ):
        if state_in.requires_grad:
            particle_q = state_out.particle_q
            # allocate new particle arrays so gradients can be tracked correctly without overwriting
            new_particle_q = wp.empty_like(state_out.particle_q)
            new_particle_qd = wp.empty_like(state_out.particle_qd)
            self._particle_delta_counter += 1
        else:
            if self._particle_delta_counter == 0:
                particle_q = state_out.particle_q
                new_particle_q = state_in.particle_q
                new_particle_qd = state_in.particle_qd
            else:
                particle_q = state_in.particle_q
                new_particle_q = state_out.particle_q
                new_particle_qd = state_out.particle_qd
            self._particle_delta_counter = 1 - self._particle_delta_counter

        wp.launch(
            kernel=apply_particle_deltas,
            dim=model.particle_count,
            inputs=[
                self.particle_q_init,
                particle_q,
                model.particle_flags,
                model.particle_mass,
                particle_deltas,
                dt,
                model.particle_max_velocity,
            ],
            outputs=[new_particle_q, new_particle_qd],
            device=model.device,
        )

        if state_in.requires_grad:
            state_out.particle_q = new_particle_q
            state_out.particle_qd = new_particle_qd

        return new_particle_q, new_particle_qd

    @override
    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float):
        requires_grad = state_in.requires_grad
        self._particle_delta_counter = 0
        model = self.model
        particle_q = None
        particle_qd = None
        particle_deltas = None
        body_deltas = None

        if control is None:
            control = model.control(clone_variables=False)

        with wp.ScopedTimer("simulate", False):
            if model.particle_count:
                particle_q = state_out.particle_q
                particle_qd = state_out.particle_qd
                self.particle_q_init = wp.clone(state_in.particle_q)
                self.particle_qd_init = wp.clone(state_in.particle_qd)
                particle_deltas = wp.empty_like(state_out.particle_qd)
                self.integrate_particles(model, state_in, state_out, dt)

                if self._particle_particle_enabled:
                    search_radius = model.particle_max_radius * 2.0 + model.particle_cohesion
                    with wp.ScopedDevice(model.device):
                        model.particle_grid.build(state_out.particle_q, radius=search_radius)

            if model.body_count:
                body_deltas = wp.zeros_like(state_out.body_qd)

            for i in range(self.iterations):
                with wp.ScopedTimer(f"iteration_{i}", False):
                    if model.particle_count:
                        # Clear deltas at start of iteration
                        if requires_grad and i > 0:
                            particle_deltas = wp.zeros_like(particle_deltas)
                        else:
                            particle_deltas.zero_()

                        # 1. Particle-shape contacts prevents particles from penetrating static/dynamic shapes in the scene
                        # 2. Particle-particle contacts handles collisions between particles in different rigid bodies (i.e. groups)
                        # 3. Shape matching constraints ensures rigid body behavior

                        # Solve contact constraints
                        if model.shape_count and contacts is not None:
                            wp.launch(
                                kernel=solve_particle_shape_contacts,
                                dim=contacts.soft_contact_max,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.particle_radius,
                                    model.particle_flags,
                                    state_out.body_q,
                                    state_out.body_qd,
                                    model.body_com,
                                    model.body_inv_mass,
                                    model.body_inv_inertia,
                                    model.body_flags,
                                    model.shape_body,
                                    model.shape_material_mu,
                                    model.soft_contact_mu,
                                    model.particle_adhesion,
                                    contacts.soft_contact_count,
                                    contacts.soft_contact_particle,
                                    contacts.soft_contact_shape,
                                    contacts.soft_contact_body_pos,
                                    contacts.soft_contact_body_vel,
                                    contacts.soft_contact_normal,
                                    contacts.soft_contact_max,
                                    dt,
                                    self.soft_contact_relaxation,
                                ],
                                outputs=[particle_deltas, body_deltas],
                                device=model.device,
                            )

                        # Solve particle-particle contacts (inter-group collisions)
                        if self._particle_particle_enabled:
                            wp.launch(
                                kernel=solve_particle_particle_contacts,
                                dim=model.particle_count,
                                inputs=[
                                    model.particle_grid.id,
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.particle_radius,
                                    model.particle_flags,
                                    model.particle_group,
                                    model.particle_mu,
                                    model.particle_cohesion,
                                    model.particle_max_radius,
                                    dt,
                                    self.soft_contact_relaxation,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )
                        # Apply all accumulated deltas at once
                        particle_q, particle_qd = self.apply_particle_deltas(
                            model, state_in, state_out, particle_deltas, dt
                        )

                        if self._num_dynamic_groups > 0:
                            particle_deltas.zero_()
                            bd = self._shape_match_block_dim
                            wp.launch(
                                kernel=solve_shape_matching_batch_tiled,
                                dim=(self._num_dynamic_groups, bd),
                                inputs=[
                                    particle_q,
                                    self.particle_q_rest,
                                    self.total_group_mass,
                                    model.particle_mass,
                                    self._group_particle_start,
                                    self._group_particle_count,
                                    self._group_particles_flat,
                                    particle_deltas,
                                ],
                                block_dim=bd,
                                device=model.device,
                            )
                            particle_q, particle_qd = self.apply_particle_deltas(
                                model, state_in, state_out, particle_deltas, dt
                            )

            if model.particle_count:
                if particle_q.ptr != state_out.particle_q.ptr:
                    state_out.particle_q.assign(particle_q)
                    state_out.particle_qd.assign(particle_qd)

            return state_out
