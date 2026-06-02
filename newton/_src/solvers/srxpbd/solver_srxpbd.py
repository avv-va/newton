from newton._src.sim import model
import warp as wp
import numpy as np

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase
from .kernels import (
    apply_particle_deltas,
    solve_particle_particle_contacts,
    solve_particle_shape_contacts,
    enforce_momemntum_conservation_tiled,
    solve_shape_matching_batch_tiled,
)

@wp.kernel
def calculate_group_particle_mass(
    particle_mass: wp.array(dtype=float),
    group_particle_start: wp.array(dtype=wp.int32),
    group_particle_count: wp.array(dtype=wp.int32),
    group_particles_flat: wp.array(dtype=wp.int32),
    total_group_mass: wp.array(dtype=float),
):
    group_id = wp.tid()
    start_idx = group_particle_start[group_id]
    num_particles = group_particle_count[group_id]
    total = wp.float32(0.0)
    for p in range(num_particles):
        idx = group_particles_flat[start_idx + p]
        total += particle_mass[idx]
    total_group_mass[group_id] = total


class SolverSRXPBD(SolverBase):
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

        # Precompute which particle groups have mass
        if model.particle_count and model.particle_group_count > 0:
            if not hasattr(self, '_group_has_mass_cache'):
                self._group_has_mass_cache = []
                for group_id in range(model.particle_group_count):
                    group_particle_indices = model.particle_groups[group_id]
                    group_masses = model.particle_mass.numpy(
                    )[group_particle_indices.numpy()]
                    has_mass = bool(np.any(group_masses > 0.0))
                    self._group_has_mass_cache.append(has_mass)

        # Precompute shape matching data for all dynamic groups
        self._dynamic_group_ids = []
        self._group_particle_start = []
        self._group_particle_count = []
        self._group_particles_flat = []
        self._num_dynamic_groups = 0
        
        if model.particle_count > 0 and model.particle_group_count > 0:
            particle_offset = 0
            for group_id in range(model.particle_group_count):
                group_particle_indices = model.particle_groups[group_id]
                group_masses = model.particle_mass.numpy(
                )[group_particle_indices.numpy()]
                has_mass = bool(np.any(group_masses > 0.0))

                if has_mass:  # Only store dynamic groups
                    num_particles = len(group_particle_indices.numpy())

                    self._dynamic_group_ids.append(group_id)
                    self._group_particle_start.append(particle_offset)
                    self._group_particle_count.append(num_particles)
                    self._group_particles_flat.extend(
                        group_particle_indices.numpy().tolist())

                    particle_offset += num_particles

            # Convert to Warp arrays
            self._dynamic_group_ids = wp.array(
                self._dynamic_group_ids, dtype=wp.int32, device=model.device)
            self._group_particle_start = wp.array(
                self._group_particle_start, dtype=wp.int32, device=model.device)
            self._group_particle_count = wp.array(
                self._group_particle_count, dtype=wp.int32, device=model.device)
            self._group_particles_flat = wp.array(
                self._group_particles_flat, dtype=wp.int32, device=model.device)

            self._num_dynamic_groups = len(self._dynamic_group_ids.numpy())

            max_particles = max(self._group_particle_count.numpy())
            self._shape_match_block_dim = min(256, int(max_particles))
            self._shape_match_block_dim = max(32, ((self._shape_match_block_dim + 31) // 32) * 32)

            self.total_group_mass = wp.zeros(self._num_dynamic_groups, dtype=wp.float32, device=model.device)
            # Pre-allocate momentum buffers to avoid per-iteration GPU allocations
            self._linear_momentum_b4_SM = wp.zeros(self._num_dynamic_groups, dtype=wp.vec3, device=model.device)
            self._angular_momentum_b4_SM = wp.zeros(self._num_dynamic_groups, dtype=wp.vec3, device=model.device)
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
                self.particle_q_init,
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

            if model.body_count:
                body_q = state_out.body_q
                body_qd = state_out.body_qd
                body_deltas = wp.empty_like(state_out.body_qd)

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
                        if model.shape_count:
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
                        # Need at least 2 groups to collide
                        if model.particle_group_count > 1:
                            # Build hash grid for spatial queries
                            model.particle_grid.build(particle_q, model.particle_max_radius)

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
                            self._linear_momentum_b4_SM.zero_()
                            self._angular_momentum_b4_SM.zero_()
                            particle_deltas.zero_()
                            bd = self._shape_match_block_dim
                            wp.launch(
                                kernel=solve_shape_matching_batch_tiled,
                                dim=(self._num_dynamic_groups, bd),
                                inputs=[
                                    particle_q,
                                    self.particle_q_rest,
                                    particle_qd,
                                    self.total_group_mass,
                                    model.particle_mass,
                                    self._group_particle_start,
                                    self._group_particle_count,
                                    self._group_particles_flat,
                                    particle_deltas,
                                    self._linear_momentum_b4_SM,
                                    self._angular_momentum_b4_SM,
                                ],
                                block_dim=bd,
                                device=model.device
                            )
                            particle_q, particle_qd = self.apply_particle_deltas(
                                model, state_in, state_out, particle_deltas, dt
                            )
                            wp.launch(
                                kernel=enforce_momemntum_conservation_tiled,
                                dim=(self._num_dynamic_groups, bd),
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    self.total_group_mass,
                                    self.model.particle_mass,
                                    self._linear_momentum_b4_SM,
                                    self._angular_momentum_b4_SM,
                                    dt,
                                    self._group_particle_start,
                                    self._group_particle_count,
                                    self._group_particles_flat,
                                ],
                                outputs=[particle_q, particle_qd],
                                block_dim=bd,
                                device=self.model.device
                            )

            if model.particle_count:
                if particle_q.ptr != state_out.particle_q.ptr:
                    state_out.particle_q.assign(particle_q)
                    state_out.particle_qd.assign(particle_qd)

            return state_out
