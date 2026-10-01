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

"""Kernels for :class:`~newton.solvers.SolverSRXPBD`.

Only the particle constraint path is implemented here; springs, cloth bending,
FEM tetrahedra, joints and rigid-body contacts are out of scope for this solver.

The two contact kernels are near-verbatim copies of their counterparts in
``newton/_src/solvers/xpbd/kernels.py`` (``solve_particle_shape_contacts`` and
``solve_particle_particle_contacts``).  They are forked rather than reused
because SRXPBD needs two behaviors the shared versions do not offer: an
:attr:`~newton.ParticleFlags.INTEGRATE_ONLY` early-out, and skipping contacts
between particles of the same group.  Re-diff against the XPBD originals when
rebasing onto a newer Newton.
"""

import warp as wp

from ...geometry import ParticleFlags
from ...sim import BodyFlags

# Restitution is applied after the positional solve and needs no SRXPBD-specific
# behavior, so the XPBD kernel is reused as-is.  Note that a kernel keeps the
# ``wp.Module`` of the file that defined it, so the solver must register the XPBD
# kernel module with ``_set_module_options`` as well for module options to apply.
from ..xpbd.kernels import apply_particle_shape_restitution

__all__ = [
    "apply_particle_deltas",
    "apply_particle_shape_restitution",
    "calculate_group_particle_mass",
    "enforce_momentum_conservation_tiled",
    "solve_particle_particle_contacts",
    "solve_particle_shape_contacts",
    "solve_shape_matching_batch_tiled",
]


@wp.kernel
def calculate_group_particle_mass(
    particle_mass: wp.array[float],
    group_particle_start: wp.array[wp.int32],
    group_particle_count: wp.array[wp.int32],
    group_particles_flat: wp.array[wp.int32],
    # outputs
    total_group_mass: wp.array[float],
):
    """Sum the particle masses of each dynamic group [kg].

    Summed on device in float32 rather than on the host in numpy: the total
    divides every centroid in :func:`solve_shape_matching_batch_tiled`, so the
    accumulation order has to match the one the solve itself would produce.

    The sum covers *every* particle of the group, including inactive ones,
    because the tiled kernels accumulate over the same unfiltered set.  Any
    future flag filtering has to be applied here by the same predicate.
    """
    group_id = wp.tid()
    start_idx = group_particle_start[group_id]
    num_particles = group_particle_count[group_id]
    total = wp.float32(0.0)
    for p in range(num_particles):
        idx = group_particles_flat[start_idx + p]
        total += particle_mass[idx]
    total_group_mass[group_id] = total


@wp.kernel
def solve_particle_shape_contacts(
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_invmass: wp.array[float],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_m_inv: wp.array[float],
    body_I_inv: wp.array[wp.mat33],
    body_flags: wp.array[wp.int32],
    shape_body: wp.array[int],
    shape_material_mu: wp.array[float],
    particle_mu: float,
    particle_ka: float,
    contact_count: wp.array[int],
    contact_particle: wp.array[int],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_max: int,
    dt: float,
    relaxation: float,
    # outputs
    delta: wp.array[wp.vec3],
    body_delta: wp.array[wp.spatial_vector],
):
    tid = wp.tid()

    count = min(contact_max, contact_count[0])
    if tid >= count:
        return

    shape_index = contact_shape[tid]
    body_index = shape_body[shape_index]
    particle_index = contact_particle[tid]

    particle_flag = particle_flags[particle_index]
    if (particle_flag & ParticleFlags.ACTIVE) == 0:
        return
    if (particle_flag & ParticleFlags.PROXY) != 0:
        if body_index < 0:
            return
        if (body_flags[body_index] & int(BodyFlags.PROXY)) != 0:
            return
        if body_m_inv[body_index] == 0.0:
            return
    # Particles driven kinematically by the scene are integrated but never
    # receive contact corrections (see examples/push/rod_pushed_box.py).
    if (particle_flag & ParticleFlags.INTEGRATE_ONLY) != 0:
        return

    px = particle_x[particle_index]
    pv = particle_v[particle_index]

    X_wb = wp.transform_identity()
    X_com = wp.vec3()

    if body_index >= 0:
        X_wb = body_q[body_index]
        X_com = body_com[body_index]

    # body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    r = bx - wp.transform_point(X_wb, X_com)

    n = contact_normal[tid]
    c = wp.dot(n, px - bx) - particle_radius[particle_index]

    if c > particle_ka:
        return

    # take average material properties of shape and particle parameters
    mu = 0.5 * (particle_mu + shape_material_mu[shape_index])

    # body velocity
    body_v_s = wp.spatial_vector()
    if body_index >= 0:
        body_v_s = body_qd[body_index]

    body_w = wp.spatial_bottom(body_v_s)
    body_v = wp.spatial_top(body_v_s)

    # compute the body velocity at the particle position
    bv = body_v + wp.cross(body_w, r) + wp.transform_vector(X_wb, contact_body_vel[tid])

    # relative velocity
    v = pv - bv

    # normal
    lambda_n = c
    delta_n = n * lambda_n

    # friction
    vn = wp.dot(n, v)
    vt = v - n * vn

    # compute inverse masses
    w1 = particle_invmass[particle_index]
    w2 = 0.0
    if body_index >= 0:
        angular = wp.cross(r, n)
        q = wp.transform_get_rotation(X_wb)
        rot_angular = wp.quat_rotate_inv(q, angular)
        I_inv = body_I_inv[body_index]
        w2 = body_m_inv[body_index] + wp.dot(rot_angular, I_inv * rot_angular)
    denom = w1 + w2
    if denom == 0.0:
        return

    lambda_f = wp.max(mu * lambda_n, -wp.length(vt) * dt)
    delta_f = wp.normalize(vt) * lambda_f
    delta_total = (delta_f - delta_n) / denom * relaxation

    wp.atomic_add(delta, particle_index, w1 * delta_total)

    if body_index >= 0:
        # SolverSRXPBD does not integrate bodies, so this accumulation is
        # discarded; the convention is kept aligned with SolverXPBD anyway.
        delta_v = delta_total / dt
        delta_w = wp.cross(r, delta_v)
        wp.atomic_sub(body_delta, body_index, wp.spatial_vector(delta_v, delta_w))


@wp.kernel
def solve_particle_particle_contacts(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_invmass: wp.array[float],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_group: wp.array[wp.int32],
    k_mu: float,
    k_cohesion: float,
    max_radius: float,
    dt: float,
    relaxation: float,
    # outputs
    deltas: wp.array[wp.vec3],
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        # hash grid has not been built yet
        return
    particle_flag = particle_flags[i]
    if (particle_flag & ParticleFlags.ACTIVE) == 0:
        return
    if (particle_flag & ParticleFlags.INTEGRATE_ONLY) != 0:
        return
    is_proxy = particle_flag & ParticleFlags.PROXY

    x = particle_x[i]
    v = particle_v[i]
    radius = particle_radius[i]
    w1 = particle_invmass[i]
    my_group = particle_group[i]

    # particle contact
    query = wp.hash_grid_query(grid, x, radius + max_radius + k_cohesion)
    index = int(0)

    delta = wp.vec3(0.0)

    while wp.hash_grid_query_next(query, index):
        neighbor_flag = particle_flags[index]
        if (
            (neighbor_flag & ParticleFlags.ACTIVE) != 0
            and (is_proxy == 0 or ((neighbor_flag & ParticleFlags.PROXY) == 0 and particle_invmass[index] > 0.0))
            and index != i
            # Particles of the same body are held together by the shape-matching
            # constraint; resolving them as contacts too would fight it.
            and (my_group < 0 or my_group != particle_group[index])
        ):
            # compute distance to point
            n = x - particle_x[index]
            d = wp.length(n)
            err = d - radius - particle_radius[index]

            # compute inverse masses
            w2 = particle_invmass[index]
            denom = w1 + w2

            if err <= k_cohesion and denom > 0.0 and d > 0.0:
                n = n / d
                vrel = v - particle_v[index]

                # normal
                lambda_n = err
                delta_n = n * lambda_n

                # friction
                vn = wp.dot(n, vrel)
                vt = vrel - n * vn

                lambda_f = wp.max(k_mu * lambda_n, -wp.length(vt) * dt)
                delta_f = wp.normalize(vt) * lambda_f
                delta += (delta_f - delta_n) / denom

    wp.atomic_add(deltas, i, delta * w1 * relaxation)


@wp.kernel
def apply_particle_deltas(
    x_pred: wp.array[wp.vec3],
    v_pred: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_mass: wp.array[float],
    delta: wp.array[wp.vec3],
    dt: float,
    v_max: float,
    # outputs
    x_out: wp.array[wp.vec3],
    v_out: wp.array[wp.vec3],
):
    """Add the accumulated constraint corrections to the predicted particle state.

    Differs from the XPBD version in two ways.  Particles of zero mass are
    pinned outright, because the shape-matching constraint emits a correction
    for every particle of a group regardless of its mass.  And the velocity is
    updated *incrementally* from the correction rather than re-derived from the
    total step displacement: the XPBD form ``(x_new - x0) / dt`` fails to
    reproduce the predicted velocity exactly when the correction is zero, and
    that residual accumulates over long horizons.
    """
    tid = wp.tid()

    # Pinned particles hold position and stay at rest.
    if particle_mass[tid] == 0.0:
        x_out[tid] = x_pred[tid]
        v_out[tid] = wp.vec3(0.0)
        return

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        # Pass the predicted state through rather than returning: the caller
        # ping-pongs between two buffers, so skipping the write would resurrect
        # a stale value from an earlier apply.
        x_out[tid] = x_pred[tid]
        v_out[tid] = v_pred[tid]
        return

    xp = x_pred[tid]
    vp = v_pred[tid]

    # constraint deltas
    d = delta[tid]

    v_new = vp + d / dt
    x_new = xp + d

    # enforce velocity limit to prevent instability; position is deliberately
    # left un-rescaled, as ``x_pred`` rather than the step start is its anchor
    v_new_mag = wp.length(v_new)
    if v_new_mag > v_max:
        v_new *= v_max / v_new_mag

    x_out[tid] = x_new
    v_out[tid] = v_new


@wp.kernel
def solve_shape_matching_batch_tiled(
    particle_q: wp.array[wp.vec3],
    particle_q_rest: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    group_mass: wp.array[float],
    particle_mass: wp.array[float],
    group_particle_start: wp.array[wp.int32],
    group_particle_count: wp.array[wp.int32],
    group_particles_flat: wp.array[wp.int32],
    relaxation: float,
    # outputs
    delta: wp.array[wp.vec3],
    linear_momentum: wp.array[wp.vec3],
    angular_momentum: wp.array[wp.vec3],
):
    """Restore rigidity of each particle group by shape matching.

    Fits the rest shape to the current configuration in the least-squares sense
    (Mueller et al. 2005): the optimal rotation is the polar factor of the
    mass-weighted covariance ``A = sum m (x - t) (x0 - t0)^T``, obtained here
    from an SVD.  Each particle is then pulled toward its goal position.

    The group's linear and angular momentum are measured *before* the
    correction is applied and reported through ``linear_momentum`` /
    ``angular_momentum``, so that
    :func:`enforce_momentum_conservation_tiled` can undo the momentum the
    projection injects.

    One block handles one group and each lane strides over the group's
    particles, so groups larger than the block size are supported.  Launch with
    ``dim=(num_groups, block_dim)`` and ``block_dim=block_dim``.
    """
    group_id, lane = wp.tid()

    start_idx = group_particle_start[group_id]
    num_particles = group_particle_count[group_id]
    M = group_mass[group_id]
    bd = wp.block_dim()

    # --- Phase 1: strided accumulation of centroid, rest centroid, momentum ---
    acc_mx = wp.vec3(0.0)
    acc_mx0 = wp.vec3(0.0)
    acc_p = wp.vec3(0.0)

    p = lane
    while p < num_particles:
        idx = group_particles_flat[start_idx + p]
        m = particle_mass[idx]
        x = particle_q[idx]
        x0 = particle_q_rest[idx]
        v = particle_qd[idx]
        acc_mx += m * x
        acc_mx0 += m * x0
        acc_p += m * v
        p += bd

    # Each tile_reduce is also the block-wide barrier separating the phases.
    t = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(acc_mx, preserve_type=True)), 0) / M
    t0 = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(acc_mx0, preserve_type=True)), 0) / M
    P = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(acc_p, preserve_type=True)), 0)
    vcom = P / M

    # --- Phase 1b: angular momentum about the centroid ---
    acc_L = wp.vec3(0.0)

    p = lane
    while p < num_particles:
        idx = group_particles_flat[start_idx + p]
        m = particle_mass[idx]
        r = particle_q[idx] - t
        vrel = particle_qd[idx] - vcom
        acc_L += wp.cross(r, m * vrel)
        p += bd

    L = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(acc_L, preserve_type=True)), 0)

    # tile_extract broadcasts the same value to every lane; one store suffices.
    if lane == 0:
        linear_momentum[group_id] = P
        angular_momentum[group_id] = L

    # --- Phase 2: covariance matrix A ---
    # Accumulated as three column vectors because wp.tile_reduce has no mat33
    # overload.
    acc_col0 = wp.vec3(0.0)
    acc_col1 = wp.vec3(0.0)
    acc_col2 = wp.vec3(0.0)

    p = lane
    while p < num_particles:
        idx = group_particles_flat[start_idx + p]
        m = particle_mass[idx]
        x = particle_q[idx]
        x0 = particle_q_rest[idx]
        pi = x - t
        qi = x0 - t0
        acc_col0 += pi * (qi[0] * m)
        acc_col1 += pi * (qi[1] * m)
        acc_col2 += pi * (qi[2] * m)
        p += bd

    sum_col0 = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(acc_col0, preserve_type=True)), 0)
    sum_col1 = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(acc_col1, preserve_type=True)), 0)
    sum_col2 = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(acc_col2, preserve_type=True)), 0)

    # fmt: off
    A = wp.mat33(
        sum_col0[0], sum_col1[0], sum_col2[0],
        sum_col0[1], sum_col1[1], sum_col2[1],
        sum_col0[2], sum_col1[2], sum_col2[2],
    )
    # fmt: on

    # --- Polar decomposition via SVD ---
    U = wp.mat33()
    S = wp.vec3()
    V = wp.mat33()
    wp.svd3(A, U, S, V)
    R = U @ wp.transpose(V)

    # Guard against a reflection when the packing is degenerate or inverted.
    if wp.determinant(R) < 0.0:
        U[:, 2] = -U[:, 2]
        R = U @ wp.transpose(V)

    # --- Phase 3: write the goal corrections ---
    p = lane
    while p < num_particles:
        idx = group_particles_flat[start_idx + p]
        x0 = particle_q_rest[idx]
        x = particle_q[idx]
        goal = R @ (x0 - t0) + t
        # A plain store, not an atomic: the caller zeroes ``delta`` immediately
        # before this launch and each particle belongs to exactly one group.
        delta[idx] = (goal - x) * relaxation
        p += bd


@wp.kernel
def enforce_momentum_conservation_tiled(
    x_pred: wp.array[wp.vec3],
    v_pred: wp.array[wp.vec3],
    group_mass: wp.array[float],
    particle_mass: wp.array[float],
    target_p: wp.array[wp.vec3],
    target_l: wp.array[wp.vec3],
    dt: float,
    group_particle_start: wp.array[wp.int32],
    group_particle_count: wp.array[wp.int32],
    group_particles_flat: wp.array[wp.int32],
    inertia_regularization: float,
    # outputs
    x_out: wp.array[wp.vec3],
    v_out: wp.array[wp.vec3],
):
    """Remove the momentum that shape matching injected into each group.

    Shape matching is a positional projection, so it does not preserve
    momentum.  This restores the pre-projection linear momentum with a uniform
    velocity shift, then the angular momentum with a rigid-body field
    ``omega_err x r``, where ``omega_err`` solves ``I omega_err = L' - L``.
    The angular part is a first-order correction, exact only for small errors.

    ``x_out`` / ``v_out`` are intended to alias ``x_pred`` / ``v_pred``: phase 2
    writes them and phase 3 reads them back.  This is safe because each lane
    only reads back the indices it wrote itself, but it makes the kernel
    unusable under ``requires_grad``.

    As in :func:`solve_shape_matching_batch_tiled`, each ``wp.tile_reduce`` is
    also the block-wide barrier separating the phases, which is why they cannot
    be merged.  Launch with ``dim=(num_groups, block_dim)`` and
    ``block_dim=block_dim``.
    """
    group_id, lane = wp.tid()
    start_idx = group_particle_start[group_id]
    num_particles = group_particle_count[group_id]
    M = group_mass[group_id]
    bd = wp.block_dim()

    # --- Phase 1: current linear momentum ---
    acc_p_now = wp.vec3(0.0)
    p = lane
    while p < num_particles:
        idx = group_particles_flat[start_idx + p]
        acc_p_now += particle_mass[idx] * v_pred[idx]
        p += bd

    p_now = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(acc_p_now, preserve_type=True)), 0)

    # Linear momentum correction, uniform across the group.
    dv = (target_p[group_id] - p_now) / M

    # --- Phase 2: apply the linear correction, accumulate com and vcom ---
    acc_com = wp.vec3(0.0)
    acc_vcom = wp.vec3(0.0)
    p = lane
    while p < num_particles:
        idx = group_particles_flat[start_idx + p]
        m = particle_mass[idx]
        v_corr = v_pred[idx] + dv
        x_corr = x_pred[idx] + dv * dt
        v_out[idx] = v_corr
        x_out[idx] = x_corr
        acc_com += m * x_corr
        acc_vcom += m * v_corr
        p += bd

    com = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(acc_com, preserve_type=True)), 0) / M
    vcom = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(acc_vcom, preserve_type=True)), 0) / M

    # --- Phase 3: inertia tensor and current angular momentum ---
    # I is symmetric, but accumulated as three columns for the same reason as
    # the covariance matrix above.
    acc_i_col0 = wp.vec3(0.0)
    acc_i_col1 = wp.vec3(0.0)
    acc_i_col2 = wp.vec3(0.0)
    acc_l_now = wp.vec3(0.0)

    identity = wp.identity(n=3, dtype=float)

    p = lane
    while p < num_particles:
        idx = group_particles_flat[start_idx + p]
        m = particle_mass[idx]
        r = x_out[idx] - com
        r2 = wp.dot(r, r)
        i_contrib = m * (r2 * identity - wp.outer(r, r))
        acc_i_col0 += wp.vec3(i_contrib[0, 0], i_contrib[1, 0], i_contrib[2, 0])
        acc_i_col1 += wp.vec3(i_contrib[0, 1], i_contrib[1, 1], i_contrib[2, 1])
        acc_i_col2 += wp.vec3(i_contrib[0, 2], i_contrib[1, 2], i_contrib[2, 2])
        vrel = v_out[idx] - vcom
        acc_l_now += wp.cross(r, m * vrel)
        p += bd

    s_i0 = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(acc_i_col0, preserve_type=True)), 0)
    s_i1 = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(acc_i_col1, preserve_type=True)), 0)
    s_i2 = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(acc_i_col2, preserve_type=True)), 0)
    l_now = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(acc_l_now, preserve_type=True)), 0)

    # fmt: off
    inertia = wp.mat33(
        s_i0[0], s_i1[0], s_i2[0],
        s_i0[1], s_i1[1], s_i2[1],
        s_i0[2], s_i1[2], s_i2[2],
    )
    # fmt: on

    if inertia_regularization > 0.0:
        # Collinear or single-particle groups have a rank-deficient inertia
        # tensor, for which wp.inverse() returns zero and the angular
        # correction is silently dropped.
        inertia += (wp.trace(inertia) / 3.0) * inertia_regularization * identity

    dl = l_now - target_l[group_id]
    omega_err = wp.inverse(inertia) @ dl

    # --- Phase 4: apply the angular correction ---
    p = lane
    while p < num_particles:
        idx = group_particles_flat[start_idx + p]
        r = x_out[idx] - com
        v_out[idx] = v_out[idx] - wp.cross(omega_err, r)
        x_out[idx] = x_out[idx] - wp.cross(omega_err, r) * dt
        p += bd
