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

import warp as wp

from ...geometry import ParticleFlags
from ...sim import BodyFlags


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
def solve_shape_matching_batch_tiled(
    particle_q: wp.array[wp.vec3],
    particle_q_rest: wp.array[wp.vec3],
    group_mass: wp.array[float],
    particle_mass: wp.array[float],
    group_particle_start: wp.array[wp.int32],
    group_particle_count: wp.array[wp.int32],
    group_particles_flat: wp.array[wp.int32],
    delta: wp.array[wp.vec3],
):
    """
    Tile-based shape matching: one block per group.
    Each thread strides over particles to accumulate local sums,
    then tile_reduce cooperatively reduces across the block.
    Supports any number of particles per group.
    Launch with dim=(num_groups, block_dim), block_dim=block_dim.
    """
    group_id, lane = wp.tid()

    start_idx = group_particle_start[group_id]
    num_particles = group_particle_count[group_id]
    M = group_mass[group_id]
    bd = wp.block_dim()

    # --- Phase 1: Each thread accumulates its strided share (com, rest com, linear momentum) ---
    acc_mx = wp.vec3(0.0)
    acc_mx0 = wp.vec3(0.0)

    p = lane
    while p < num_particles:
        idx = group_particles_flat[start_idx + p]
        m = particle_mass[idx]
        x = particle_q[idx]
        x0 = particle_q_rest[idx]
        acc_mx += m * x
        acc_mx0 += m * x0
        p += bd

    # --- Cooperative reduction across the block ---
    t = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(acc_mx, preserve_type=True)), 0) / M
    t0 = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(acc_mx0, preserve_type=True)), 0) / M

    # --- Phase 2: Covariance matrix A (strided accumulation + tile reduce) ---
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

    A = wp.mat33(
        sum_col0[0],
        sum_col1[0],
        sum_col2[0],
        sum_col0[1],
        sum_col1[1],
        sum_col2[1],
        sum_col0[2],
        sum_col1[2],
        sum_col2[2],
    )

    # --- SVD and rotation ---
    U = wp.mat33()
    S = wp.vec3()
    V = wp.mat33()
    wp.svd3(A, U, S, V)
    R = U @ wp.transpose(V)

    if wp.determinant(R) < 0.0:
        U[:, 2] = -U[:, 2]
        R = U @ wp.transpose(V)

    # --- Phase 3: Apply deltas (each thread strides over its particles) ---
    p = lane
    while p < num_particles:
        idx = group_particles_flat[start_idx + p]
        x0 = particle_q_rest[idx]
        x = particle_q[idx]
        goal = R @ (x0 - t0) + t
        dx = goal - x
        wp.atomic_add(delta, idx, dx)
        p += bd


@wp.kernel
def apply_particle_deltas(
    x_orig: wp.array[wp.vec3],
    x_pred: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_mass: wp.array[float],
    delta: wp.array[wp.vec3],
    dt: float,
    v_max: float,
    # outputs
    x_out: wp.array[wp.vec3],
    v_out: wp.array[wp.vec3],
):
    tid = wp.tid()

    if particle_mass[tid] == 0.0:
        x_out[tid] = x_pred[tid]
        v_out[tid] = wp.vec3(0.0)
        return

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return

    x0 = x_orig[tid]
    xp = x_pred[tid]

    # constraint deltas
    d = delta[tid]

    x_new = xp + d
    v_new = (x_new - x0) / dt

    # enforce velocity limit to prevent instability
    v_new_mag = wp.length(v_new)
    if v_new_mag > v_max:
        v_new *= v_max / v_new_mag
        x_new = x0 + v_new * dt

    x_out[tid] = x_new
    v_out[tid] = v_new
