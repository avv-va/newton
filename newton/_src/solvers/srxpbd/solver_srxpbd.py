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
from ...sim import Contacts, Control, Model, ModelFlags, State
from ...utils.deprecation import deprecate_nonkeyword_arguments
from ..solver import SolverBase
from ..xpbd import kernels as xpbd_kernels
from . import kernels
from .kernels import (
    apply_particle_deltas,
    apply_particle_shape_restitution,
    calculate_group_particle_mass,
    enforce_momentum_conservation_tiled,
    solve_particle_particle_contacts,
    solve_particle_shape_contacts,
    solve_shape_matching_batch_tiled,
)


class SolverSRXPBD(SolverBase):
    """Shape-matching rigid solver for particle-represented rigid bodies.

    Each rigid body is a *particle group* -- a set of particles sharing an entry
    in :attr:`newton.Model.particle_group`, as produced by
    :meth:`~newton.ModelBuilder.add_particle_volume`.  Every solver iteration
    resolves particle-shape and inter-group particle-particle contacts, restores
    the body's rigidity with a shape-matching projection, and then removes the
    momentum that projection introduced.

    Rigidity is enforced against a *rest shape* captured from
    :attr:`newton.Model.particle_q` when the solver is constructed, so the model
    must already hold the desired world-space packing at that point.  Use
    :meth:`set_rest_shape` to re-capture it later.

    References:
        - Matthias Mueller, Bruno Heidelberger, Matthias Teschner, and Markus Gross. 2005.
          Meshless deformations based on shape matching. ACM Trans. Graph. 24, 3, 471-478.
          https://doi.org/10.1145/1073204.1073216

    Limitations:
        - Particles only.  Springs, cloth bending, FEM tetrahedra, joints, rigid-body
          integration and rigid-rigid contacts are not simulated; rigid bodies act purely
          as colliders.  Pair this solver with :class:`~newton.solvers.SolverMuJoCo` or
          :class:`~newton.solvers.SolverFeatherstone` for scenes that also contain
          articulated bodies.
        - Shape matching and momentum conservation ignore
          :class:`~newton.ParticleFlags`, so every particle of a group participates
          regardless of whether it is active.  Contact solving does honour the flags.
        - Groups whose particles are collinear (including single-particle groups) have a
          singular inertia tensor, and their angular momentum correction is dropped unless
          ``inertia_regularization`` is set.
        - Not differentiable: the momentum-conservation pass updates positions and
          velocities in place.
        - :attr:`~newton.ParticleFlags.PROXY` solver coupling is not supported.

    Example
    -------

    .. code-block:: python

        solver = newton.solvers.SolverSRXPBD(model, iterations=10)

        for _ in range(100):
            state_in.clear_forces()
            contacts = model.collide(state_in)
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in
    """

    @deprecate_nonkeyword_arguments
    def __init__(
        self,
        model: Model,
        *,
        iterations: int = 2,
        soft_contact_relaxation: float = 0.9,
        shape_matching_relaxation: float = 1.0,
        enforce_momentum_conservation: bool = True,
        inertia_regularization: float = 0.0,
        enable_restitution: bool = False,
        deterministic: wp.DeterministicMode | None = None,
    ):
        """Initialize the shape-matching rigid XPBD solver.

        Args:
            model: Simulation model to integrate.
            iterations: Number of constraint-solver iterations per time step. Defaults to 2.
            soft_contact_relaxation: Relaxation factor applied to particle-particle and
                particle-shape contact corrections [dimensionless]. Defaults to 0.9.
            shape_matching_relaxation: Fraction of the shape-matching goal correction applied
                per iteration [dimensionless]. ``1.0`` enforces rigidity as a hard constraint.
                Defaults to 1.0.
            enforce_momentum_conservation: Whether to restore the linear and angular momentum
                that the shape-matching projection introduces. Defaults to ``True``.
            inertia_regularization: Tikhonov term added to each group's inertia tensor before
                inverting it, as a fraction of ``trace(I) / 3`` [dimensionless]. ``0.0``
                (default) inverts the tensor unmodified. Raise it for groups whose particles
                are collinear.
            enable_restitution: Whether to apply restitution to particle-shape contacts after
                the positional solve. Defaults to ``False``.
            deterministic: Opt-in determinism for this solver's atomic-emitting kernel
                modules. Pass a :class:`warp.DeterministicMode`, or ``None`` (default) to
                inherit the current ``wp.config.deterministic`` mode.
        """
        super().__init__(model=model)

        effective_deterministic = deterministic if deterministic is not None else wp.config.deterministic
        module_options = {
            "deterministic": effective_deterministic,
            "deterministic_max_records": 0,
        }
        self._set_module_options(module_options, module=kernels)
        # A kernel belongs to the module that defined it, so the restitution
        # kernel reused from XPBD needs its own registration.
        self._set_module_options(module_options, module=xpbd_kernels)

        self.iterations = iterations
        self.soft_contact_relaxation = soft_contact_relaxation
        self.shape_matching_relaxation = shape_matching_relaxation
        self.enforce_momentum_conservation = enforce_momentum_conservation
        self.inertia_regularization = inertia_regularization
        self.enable_restitution = enable_restitution

        self._init_kinematic_state()

        # helper variable to track constraint resolution vars
        self._particle_delta_counter = 0

        self.particle_q_init = None
        self.particle_qd_init = None

        self.set_rest_shape()
        self.rebuild_group_layout()

        # Inter-group contacts are the only particle-particle contacts this
        # solver resolves; a scene with a single group never needs the grid.
        self._particle_particle_enabled = (
            model.particle_count > 1
            and model.particle_grid is not None
            and model.particle_max_radius > 0.0
            and model.particle_group_count > 1
        )
        if self._particle_particle_enabled:
            with wp.ScopedDevice(model.device):
                model.particle_grid.reserve(model.particle_count)

    def set_rest_shape(self, particle_q: wp.array[wp.vec3] | None = None) -> None:
        """Capture the configuration that shape matching restores.

        Args:
            particle_q: Rest positions [m], shape ``[particle_count]``. If ``None``,
                :attr:`newton.Model.particle_q` is used.
        """
        source = self.model.particle_q if particle_q is None else particle_q
        self.particle_q_rest = wp.clone(source)

    def rebuild_group_layout(self) -> None:
        """Rebuild the cached particle-group layout from the model.

        Call this after changing :attr:`newton.Model.particle_mass` or
        :attr:`newton.Model.particle_group`, neither of which is covered by a
        :class:`~newton.ModelFlags` bit.
        """
        model = self.model

        self._dynamic_group_ids = None
        self._group_particle_start = None
        self._group_particle_count = None
        self._group_particles_flat = None
        self._num_dynamic_groups = 0
        self._shape_match_block_dim = 32
        self.total_group_mass = None
        self._linear_momentum_pre_sm = None
        self._angular_momentum_pre_sm = None

        if model.particle_count == 0 or model.particle_group_count == 0:
            return

        group = model.particle_group.numpy()
        mass = model.particle_mass.numpy()

        grouped = np.flatnonzero(group >= 0).astype(np.int32)
        if grouped.size == 0:
            return
        gid = group[grouped]

        # A group is dynamic if any of its particles has mass; static groups are
        # excluded from shape matching and momentum conservation entirely.
        num_groups = model.particle_group_count
        has_mass = np.bincount(gid, weights=(mass[grouped] > 0.0), minlength=num_groups) > 0.0
        dynamic_ids = np.flatnonzero(has_mass).astype(np.int32)
        if dynamic_ids.size == 0:
            return

        # Stable sort keeps groups in ascending id and particles in ascending
        # index within a group, matching the order add_particle_volume assigns.
        order = np.argsort(gid, kind="stable")
        flat = grouped[order][np.isin(gid[order], dynamic_ids)]
        counts = np.bincount(gid, minlength=num_groups)[dynamic_ids].astype(np.int32)
        starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int32)

        device = model.device
        self._dynamic_group_ids = wp.array(dynamic_ids, dtype=wp.int32, device=device)
        self._group_particle_start = wp.array(starts, dtype=wp.int32, device=device)
        self._group_particle_count = wp.array(counts, dtype=wp.int32, device=device)
        self._group_particles_flat = wp.array(flat, dtype=wp.int32, device=device)
        self._num_dynamic_groups = int(dynamic_ids.size)

        # One block per group, rounded to a whole number of warps; larger groups
        # are covered by the strided loops inside the tiled kernels.
        block_dim = min(256, int(counts.max()))
        self._shape_match_block_dim = max(32, ((block_dim + 31) // 32) * 32)

        self.total_group_mass = wp.zeros(self._num_dynamic_groups, dtype=float, device=device)
        self._linear_momentum_pre_sm = wp.zeros(self._num_dynamic_groups, dtype=wp.vec3, device=device)
        self._angular_momentum_pre_sm = wp.zeros(self._num_dynamic_groups, dtype=wp.vec3, device=device)

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
            device=device,
        )

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Refresh cached body data after model properties change.

        Effective inverse masses and inertia tensors are refreshed when
        :attr:`~newton.ModelFlags.BODY_PROPERTIES` or
        :attr:`~newton.ModelFlags.BODY_INERTIAL_PROPERTIES` is set. Other flags are
        ignored; particle group changes are not covered by any flag, see
        :meth:`rebuild_group_layout`.

        Args:
            flags: Bitmask of :class:`~newton.ModelFlags` or custom ``int`` bits indicating
                which model properties changed.
        """
        self._apply_module_options()
        if flags & (ModelFlags.BODY_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES):
            self._refresh_kinematic_state()

    def _apply_particle_deltas(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        particle_deltas: wp.array[wp.vec3],
        dt: float,
    ) -> tuple[wp.array[wp.vec3], wp.array[wp.vec3]]:
        # Unlike SolverXPBD, the velocity buffer has to ping-pong alongside the
        # position buffer, because the kernel derives the new velocity from the
        # predicted one rather than from the start-of-step position.
        if state_in.requires_grad:
            particle_q = state_out.particle_q
            particle_qd = state_out.particle_qd
            # allocate new particle arrays so gradients can be tracked correctly without overwriting
            new_particle_q = wp.empty_like(state_out.particle_q)
            new_particle_qd = wp.empty_like(state_out.particle_qd)
            self._particle_delta_counter += 1
        else:
            if self._particle_delta_counter == 0:
                particle_q = state_out.particle_q
                particle_qd = state_out.particle_qd
                new_particle_q = state_in.particle_q
                new_particle_qd = state_in.particle_qd
            else:
                particle_q = state_in.particle_q
                particle_qd = state_in.particle_qd
                new_particle_q = state_out.particle_q
                new_particle_qd = state_out.particle_qd
            self._particle_delta_counter = 1 - self._particle_delta_counter

        wp.launch(
            kernel=apply_particle_deltas,
            dim=model.particle_count,
            inputs=[
                particle_q,
                particle_qd,
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
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance the simulation state by one time step.

        Args:
            state_in: State at the beginning of the time step.
            state_out: State that receives the simulation result.
            control: Unused; accepted for compatibility with :meth:`~newton.solvers.SolverBase.step`.
            contacts: Contact data produced by :meth:`~newton.Model.collide`. If ``None``,
                particle-shape contact handling is skipped.
            dt: Time step size [s].
        """
        self._apply_module_options()

        model = self.model
        if model.particle_count == 0:
            return

        requires_grad = state_in.requires_grad
        self._particle_delta_counter = 0

        with wp.ScopedTimer("simulate", False):
            particle_q = state_out.particle_q
            particle_qd = state_out.particle_qd
            particle_deltas = wp.empty_like(state_out.particle_qd)

            body_q = None
            body_qd = None
            body_deltas = None
            if model.body_count:
                body_q = state_out.body_q
                body_qd = state_out.body_qd
                # Written by the contact kernel and then discarded: this solver
                # treats rigid bodies as colliders and never integrates them.
                body_deltas = wp.zeros_like(state_out.body_qd)

            if self.enable_restitution:
                self.particle_q_init = wp.clone(state_in.particle_q)
                self.particle_qd_init = wp.clone(state_in.particle_qd)

            self.integrate_particles(model, state_in, state_out, dt)

            if self._particle_particle_enabled:
                # Must cover the largest interaction distance the query uses.
                search_radius = model.particle_max_radius * 2.0 + model.particle_cohesion
                with wp.ScopedDevice(model.device):
                    model.particle_grid.build(state_out.particle_q, radius=search_radius)

            for i in range(self.iterations):
                with wp.ScopedTimer(f"iteration_{i}", False):
                    if requires_grad and i > 0:
                        particle_deltas = wp.zeros_like(particle_deltas)
                    else:
                        particle_deltas.zero_()

                    if model.shape_count and contacts is not None:
                        contacts._assert_particle_only_soft_contacts("SolverSRXPBD")
                        wp.launch(
                            kernel=solve_particle_shape_contacts,
                            dim=contacts.soft_contact_max,
                            inputs=[
                                particle_q,
                                particle_qd,
                                model.particle_inv_mass,
                                model.particle_radius,
                                model.particle_flags,
                                body_q,
                                body_qd,
                                model.body_com,
                                self.body_inv_mass_effective,
                                self.body_inv_inertia_effective,
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

                    particle_q, particle_qd = self._apply_particle_deltas(
                        model, state_in, state_out, particle_deltas, dt
                    )

                    if self._num_dynamic_groups:
                        block_dim = self._shape_match_block_dim
                        particle_deltas.zero_()
                        wp.launch(
                            kernel=solve_shape_matching_batch_tiled,
                            dim=(self._num_dynamic_groups, block_dim),
                            block_dim=block_dim,
                            inputs=[
                                particle_q,
                                self.particle_q_rest,
                                particle_qd,
                                self.total_group_mass,
                                model.particle_mass,
                                self._group_particle_start,
                                self._group_particle_count,
                                self._group_particles_flat,
                                self.shape_matching_relaxation,
                            ],
                            outputs=[
                                particle_deltas,
                                self._linear_momentum_pre_sm,
                                self._angular_momentum_pre_sm,
                            ],
                            device=model.device,
                        )

                        particle_q, particle_qd = self._apply_particle_deltas(
                            model, state_in, state_out, particle_deltas, dt
                        )

                        if self.enforce_momentum_conservation:
                            wp.launch(
                                kernel=enforce_momentum_conservation_tiled,
                                dim=(self._num_dynamic_groups, block_dim),
                                block_dim=block_dim,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    self.total_group_mass,
                                    model.particle_mass,
                                    self._linear_momentum_pre_sm,
                                    self._angular_momentum_pre_sm,
                                    dt,
                                    self._group_particle_start,
                                    self._group_particle_count,
                                    self._group_particles_flat,
                                    self.inertia_regularization,
                                ],
                                outputs=[particle_q, particle_qd],
                                device=model.device,
                            )

            if particle_q.ptr != state_out.particle_q.ptr:
                state_out.particle_q.assign(particle_q)
                state_out.particle_qd.assign(particle_qd)

            if self.enable_restitution and contacts is not None:
                wp.launch(
                    kernel=apply_particle_shape_restitution,
                    dim=contacts.soft_contact_max,
                    inputs=[
                        state_out.particle_qd,
                        self.particle_q_init,
                        self.particle_qd_init,
                        model.particle_radius,
                        model.particle_flags,
                        model.particle_world,
                        body_q,
                        # Rigid bodies are not integrated here, so the incoming
                        # state is also their pre-step state.
                        state_in.body_q,
                        body_qd,
                        state_in.body_qd,
                        model.body_com,
                        model.shape_body,
                        model.particle_adhesion,
                        model.soft_contact_restitution,
                        model.gravity,
                        dt,
                        contacts.soft_contact_count,
                        contacts.soft_contact_particle,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_body_vel,
                        contacts.soft_contact_normal,
                        contacts.soft_contact_max,
                    ],
                    outputs=[state_out.particle_qd],
                    device=model.device,
                )
