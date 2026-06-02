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

from ...core import velocity_at_point
from ...geometry import ParticleFlags
from ...sim import JointType
from ...utils import (
    vec_abs,
    vec_leaky_max,
    vec_leaky_min,
    vec_max,
    vec_min,
)


@wp.kernel
def apply_particle_shape_restitution(
    particle_x_new: wp.array(dtype=wp.vec3),
    particle_v_new: wp.array(dtype=wp.vec3),
    particle_x_old: wp.array(dtype=wp.vec3),
    particle_v_old: wp.array(dtype=wp.vec3),
    particle_invmass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_m_inv: wp.array(dtype=float),
    body_I_inv: wp.array(dtype=wp.mat33),
    shape_body: wp.array(dtype=int),
    particle_ka: float,
    restitution: float,
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_max: int,
    dt: float,
    relaxation: float,
    particle_v_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    count = min(contact_max, contact_count[0])
    if tid >= count:
        return

    shape_index = contact_shape[tid]
    body_index = shape_body[shape_index]
    particle_index = contact_particle[tid]

    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return

    # x_new = particle_x_new[particle_index]
    v_new = particle_v_new[particle_index]
    px = particle_x_old[particle_index]
    v_old = particle_v_old[particle_index]

    X_wb = wp.transform_identity()
    # X_com = wp.vec3()

    if body_index >= 0:
        X_wb = body_q[body_index]
        # X_com = body_com[body_index]

    # body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    # r = bx - wp.transform_point(X_wb, X_com)

    n = contact_normal[tid]
    c = wp.dot(n, px - bx) - particle_radius[particle_index]

    if c > particle_ka:
        return

    rel_vel_old = wp.dot(n, v_old)
    rel_vel_new = wp.dot(n, v_new)

    if rel_vel_old < 0.0:
        # dv = -n * wp.max(-rel_vel_new + wp.max(-restitution * rel_vel_old, 0.0), 0.0)
        dv = n * (-rel_vel_new + wp.max(-restitution * rel_vel_old, 0.0))

        # compute inverse masses
        # w1 = particle_invmass[particle_index]
        # w2 = 0.0
        # if body_index >= 0:
        #     angular = wp.cross(r, n)
        #     q = wp.transform_get_rotation(X_wb)
        #     rot_angular = wp.quat_rotate_inv(q, angular)
        #     I_inv = body_I_inv[body_index]
        #     w2 = body_m_inv[body_index] + wp.dot(rot_angular, I_inv * rot_angular)
        # denom = w1 + w2
        # if denom == 0.0:
        #     return

        wp.atomic_add(particle_v_out, tid, dv)


@wp.kernel
def solve_particle_shape_contacts(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_invmass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_m_inv: wp.array(dtype=float),
    body_I_inv: wp.array(dtype=wp.mat33),
    shape_body: wp.array(dtype=int),
    shape_material_mu: wp.array(dtype=float),
    particle_mu: float,
    particle_ka: float,
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_max: int,
    dt: float,
    relaxation: float,
    # outputs
    delta: wp.array(dtype=wp.vec3),
    body_delta: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    count = min(contact_max, contact_count[0])
    if tid >= count:
        return

    shape_index = contact_shape[tid]
    body_index = shape_body[shape_index]
    particle_index = contact_particle[tid]

    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return
    if (particle_flags[particle_index] & ParticleFlags.INTEGRATE_ONLY) == 2:
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
    delta_total = ((delta_f - delta_n) / denom) * relaxation
    wp.atomic_add(delta, particle_index, w1 * delta_total)

    if body_index >= 0:
        delta_t = wp.cross(r, delta_total)
        wp.atomic_sub(body_delta, body_index, wp.spatial_vector(delta_total, delta_t))


@wp.kernel
def solve_particle_particle_contacts(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_invmass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_group: wp.array(dtype=wp.int32),
    k_mu: float,
    k_cohesion: float,
    max_radius: float,
    dt: float,
    relaxation: float,
    # outputs
    deltas: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        # hash grid has not been built yet
        return
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return
    
    if (particle_flags[i] & ParticleFlags.INTEGRATE_ONLY) == 2:
        return

    x = particle_x[i]
    v = particle_v[i]
    radius = particle_radius[i]
    w1 = particle_invmass[i]

    # particle contact
    query = wp.hash_grid_query(grid, x, radius + max_radius + k_cohesion)
    index = int(0)

    delta = wp.vec3(0.0)

    while wp.hash_grid_query_next(query, index):
        # Only collide with particles from different groups (inter-group collisions are 
        # handled by shape matching)
        my_group = particle_group[i]
        other_group = particle_group[index]

        # Skip if same group
        if my_group >= 0 and my_group == other_group:
            continue

        if (particle_flags[index] & ParticleFlags.ACTIVE) != 0 and index != i:
            # compute distance to point
            n = x - particle_x[index]
            d = wp.length(n)
            err = d - radius - particle_radius[index]

            # compute inverse masses
            w2 = particle_invmass[index]
            denom = w1 + w2

            if err <= k_cohesion and denom > 0.0:
                n = n / d
                vrel = v - particle_v[index]

                # normal
                lambda_n = err
                delta_n = n * lambda_n

                # friction
                vn = wp.dot(n, vrel)
                vt = v - n * vn

                lambda_f = wp.max(k_mu * lambda_n, -wp.length(vt) * dt)
                delta_f = wp.normalize(vt) * lambda_f
                delta += (delta_f - delta_n) / denom

    wp.atomic_add(deltas, i, delta * w1 * relaxation)


@wp.kernel
def solve_springs(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    invmass: wp.array(dtype=float),
    spring_indices: wp.array(dtype=int),
    spring_rest_lengths: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
    dt: float,
    lambdas: wp.array(dtype=float),
    delta: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]

    ke = spring_stiffness[tid]
    kd = spring_damping[tid]
    rest = spring_rest_lengths[tid]

    xi = x[i]
    xj = x[j]

    vi = v[i]
    vj = v[j]

    xij = xi - xj
    vij = vi - vj

    l = wp.length(xij)

    if l == 0.0:
        return

    n = xij / l

    c = l - rest
    grad_c_xi = n
    grad_c_xj = -1.0 * n

    wi = invmass[i]
    wj = invmass[j]

    denom = wi + wj

    # Note strict inequality for damping -- 0 damping is ok
    if denom <= 0.0 or ke <= 0.0 or kd < 0.0:
        return

    alpha = 1.0 / (ke * dt * dt)
    gamma = kd / (ke * dt)

    grad_c_dot_v = dt * wp.dot(grad_c_xi, vij)  # Note: dt because from the paper we want x_i - x^n, not v...
    dlambda = -1.0 * (c + alpha * lambdas[tid] + gamma * grad_c_dot_v) / ((1.0 + gamma) * denom + alpha)

    dxi = wi * dlambda * grad_c_xi
    dxj = wj * dlambda * grad_c_xj

    lambdas[tid] = lambdas[tid] + dlambda

    wp.atomic_add(delta, i, dxi)
    wp.atomic_add(delta, j, dxj)


@wp.kernel
def bending_constraint(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    invmass: wp.array(dtype=float),
    indices: wp.array2d(dtype=int),
    rest: wp.array(dtype=float),
    bending_properties: wp.array2d(dtype=float),
    dt: float,
    lambdas: wp.array(dtype=float),
    delta: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    eps = 1.0e-6

    ke = bending_properties[tid, 0]
    kd = bending_properties[tid, 1]

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    if i == -1 or j == -1 or k == -1 or l == -1:
        return

    rest_angle = rest[tid]

    x1 = x[i]
    x2 = x[j]
    x3 = x[k]
    x4 = x[l]

    v1 = v[i]
    v2 = v[j]
    v3 = v[k]
    v4 = v[l]

    w1 = invmass[i]
    w2 = invmass[j]
    w3 = invmass[k]
    w4 = invmass[l]

    n1 = wp.cross(x3 - x1, x4 - x1)  # normal to face 1
    n2 = wp.cross(x4 - x2, x3 - x2)  # normal to face 2
    e = x4 - x3

    n1_length = wp.length(n1)
    n2_length = wp.length(n2)
    e_length = wp.length(e)

    # Check for degenerate cases
    if n1_length < eps or n2_length < eps or e_length < eps:
        return

    n1_hat = n1 / n1_length
    n2_hat = n2 / n2_length
    e_hat = e / e_length

    cos_theta = wp.dot(n1_hat, n2_hat)
    sin_theta = wp.dot(wp.cross(n1_hat, n2_hat), e_hat)
    theta = wp.atan2(sin_theta, cos_theta)

    c = theta - rest_angle

    grad_x1 = -n1_hat * e_length
    grad_x2 = -n2_hat * e_length
    grad_x3 = -n1_hat * wp.dot(x1 - x4, e_hat) - n2_hat * wp.dot(x2 - x4, e_hat)
    grad_x4 = -n1_hat * wp.dot(x3 - x1, e_hat) - n2_hat * wp.dot(x3 - x2, e_hat)

    denominator = (
        w1 * wp.length_sq(grad_x1)
        + w2 * wp.length_sq(grad_x2)
        + w3 * wp.length_sq(grad_x3)
        + w4 * wp.length_sq(grad_x4)
    )

    # Note strict inequality for damping -- 0 damping is ok
    if denominator <= 0.0 or ke <= 0.0 or kd < 0.0:
        return

    alpha = 1.0 / (ke * dt * dt)
    gamma = kd / (ke * dt)

    grad_dot_v = dt * (wp.dot(grad_x1, v1) + wp.dot(grad_x2, v2) + wp.dot(grad_x3, v3) + wp.dot(grad_x4, v4))

    dlambda = -1.0 * (c + alpha * lambdas[tid] + gamma * grad_dot_v) / ((1.0 + gamma) * denominator + alpha)

    delta0 = w1 * dlambda * grad_x1
    delta1 = w2 * dlambda * grad_x2
    delta2 = w3 * dlambda * grad_x3
    delta3 = w4 * dlambda * grad_x4

    lambdas[tid] = lambdas[tid] + dlambda

    wp.atomic_add(delta, i, delta0)
    wp.atomic_add(delta, j, delta1)
    wp.atomic_add(delta, k, delta2)
    wp.atomic_add(delta, l, delta3)


@wp.kernel
def solve_tetrahedra(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    indices: wp.array(dtype=int, ndim=2),
    rest_matrix: wp.array(dtype=wp.mat33),
    activation: wp.array(dtype=float),
    materials: wp.array(dtype=float, ndim=2),
    dt: float,
    relaxation: float,
    delta: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    # act = activation[tid]

    # k_mu = materials[tid, 0]
    # k_lambda = materials[tid, 1]
    # k_damp = materials[tid, 2]

    x0 = x[i]
    x1 = x[j]
    x2 = x[k]
    x3 = x[l]

    # v0 = v[i]
    # v1 = v[j]
    # v2 = v[k]
    # v3 = v[l]

    w0 = inv_mass[i]
    w1 = inv_mass[j]
    w2 = inv_mass[k]
    w3 = inv_mass[l]

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    Ds = wp.matrix_from_cols(x10, x20, x30)
    Dm = rest_matrix[tid]
    inv_QT = wp.transpose(Dm)

    inv_rest_volume = wp.determinant(Dm) * 6.0

    # F = Xs*Xm^-1
    F = Ds * Dm

    f1 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    f2 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    f3 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])

    tr = wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3)

    C = float(0.0)
    dC = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    compliance = float(0.0)

    stretching_compliance = relaxation
    volume_compliance = relaxation

    num_terms = 2
    for term in range(0, num_terms):
        if term == 0:
            # deviatoric, stable
            C = tr - 3.0
            dC = F * 2.0
            compliance = stretching_compliance
        elif term == 1:
            # volume conservation
            C = wp.determinant(F) - 1.0
            dC = wp.matrix_from_cols(wp.cross(f2, f3), wp.cross(f3, f1), wp.cross(f1, f2))
            compliance = volume_compliance

        if C != 0.0:
            dP = dC * inv_QT
            grad1 = wp.vec3(dP[0][0], dP[1][0], dP[2][0])
            grad2 = wp.vec3(dP[0][1], dP[1][1], dP[2][1])
            grad3 = wp.vec3(dP[0][2], dP[1][2], dP[2][2])
            grad0 = -grad1 - grad2 - grad3

            w = (
                wp.dot(grad0, grad0) * w0
                + wp.dot(grad1, grad1) * w1
                + wp.dot(grad2, grad2) * w2
                + wp.dot(grad3, grad3) * w3
            )

            if w > 0.0:
                alpha = compliance / dt / dt
                if inv_rest_volume > 0.0:
                    alpha *= inv_rest_volume
                dlambda = -C / (w + alpha)

                wp.atomic_add(delta, i, w0 * dlambda * grad0)
                wp.atomic_add(delta, j, w1 * dlambda * grad1)
                wp.atomic_add(delta, k, w2 * dlambda * grad2)
                wp.atomic_add(delta, l, w3 * dlambda * grad3)
                # wp.atomic_add(particle.num_corr, id0, 1)
                # wp.atomic_add(particle.num_corr, id1, 1)
                # wp.atomic_add(particle.num_corr, id2, 1)
                # wp.atomic_add(particle.num_corr, id3, 1)

    # C_Spherical
    # r_s = wp.sqrt(wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3))
    # r_s_inv = 1.0/r_s
    # C = r_s - wp.sqrt(3.0)
    # dCdx = F*wp.transpose(Dm)*r_s_inv
    # alpha = 1.0

    # C_D
    # r_s = wp.sqrt(wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3))
    # C = r_s*r_s - 3.0
    # dCdx = F*wp.transpose(Dm)*2.0
    # alpha = 1.0

    # grad1 = wp.vec3(dCdx[0, 0], dCdx[1, 0], dCdx[2, 0])
    # grad2 = wp.vec3(dCdx[0, 1], dCdx[1, 1], dCdx[2, 1])
    # grad3 = wp.vec3(dCdx[0, 2], dCdx[1, 2], dCdx[2, 2])
    # grad0 = (grad1 + grad2 + grad3) * (0.0 - 1.0)

    # denom = (
    #     wp.dot(grad0, grad0) * w0 + wp.dot(grad1, grad1) * w1 + wp.dot(grad2, grad2) * w2 + wp.dot(grad3, grad3) * w3
    # )
    # multiplier = C / (denom + 1.0 / (k_mu * dt * dt * rest_volume))

    # delta0 = grad0 * multiplier
    # delta1 = grad1 * multiplier
    # delta2 = grad2 * multiplier
    # delta3 = grad3 * multiplier

    # # hydrostatic part
    # J = wp.determinant(F)

    # C_vol = J - alpha
    # # dCdx = wp.matrix_from_cols(wp.cross(f2, f3), wp.cross(f3, f1), wp.cross(f1, f2))*wp.transpose(Dm)

    # # grad1 = wp.vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0])
    # # grad2 = wp.vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1])
    # # grad3 = wp.vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2])
    # # grad0 = (grad1 + grad2 + grad3)*(0.0 - 1.0)

    # s = inv_rest_volume / 6.0
    # grad1 = wp.cross(x20, x30) * s
    # grad2 = wp.cross(x30, x10) * s
    # grad3 = wp.cross(x10, x20) * s
    # grad0 = -(grad1 + grad2 + grad3)

    # denom = (
    #     wp.dot(grad0, grad0) * w0 + wp.dot(grad1, grad1) * w1 + wp.dot(grad2, grad2) * w2 + wp.dot(grad3, grad3) * w3
    # )
    # multiplier = C_vol / (denom + 1.0 / (k_lambda * dt * dt * rest_volume))

    # delta0 += grad0 * multiplier
    # delta1 += grad1 * multiplier
    # delta2 += grad2 * multiplier
    # delta3 += grad3 * multiplier

    # # # apply forces
    # # wp.atomic_sub(delta, i, delta0 * w0 * relaxation)
    # # wp.atomic_sub(delta, j, delta1 * w1 * relaxation)
    # # wp.atomic_sub(delta, k, delta2 * w2 * relaxation)
    # # wp.atomic_sub(delta, l, delta3 * w3 * relaxation)


@wp.kernel
def solve_tetrahedra2(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    indices: wp.array(dtype=int, ndim=2),
    pose: wp.array(dtype=wp.mat33),
    activation: wp.array(dtype=float),
    materials: wp.array(dtype=float, ndim=2),
    dt: float,
    relaxation: float,
    delta: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    # act = activation[tid]

    k_mu = materials[tid, 0]
    k_lambda = materials[tid, 1]
    # k_damp = materials[tid, 2]

    x0 = x[i]
    x1 = x[j]
    x2 = x[k]
    x3 = x[l]

    # v0 = v[i]
    # v1 = v[j]
    # v2 = v[k]
    # v3 = v[l]

    w0 = inv_mass[i]
    w1 = inv_mass[j]
    w2 = inv_mass[k]
    w3 = inv_mass[l]

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    Ds = wp.matrix_from_cols(x10, x20, x30)
    Dm = pose[tid]

    inv_rest_volume = wp.determinant(Dm) * 6.0
    rest_volume = 1.0 / inv_rest_volume

    # F = Xs*Xm^-1
    F = Ds * Dm

    f1 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    f2 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    f3 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])

    # C_sqrt
    # tr = wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3)
    # r_s = wp.sqrt(abs(tr - 3.0))
    # C = r_s

    # if (r_s == 0.0):
    #     return

    # if (tr < 3.0):
    #     r_s = 0.0 - r_s

    # dCdx = F*wp.transpose(Dm)*(1.0/r_s)
    # alpha = 1.0 + k_mu / k_lambda

    # C_Neo
    r_s = wp.sqrt(wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3))
    if r_s == 0.0:
        return
    # tr = wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3)
    # if (tr < 3.0):
    #     r_s = -r_s
    r_s_inv = 1.0 / r_s
    C = r_s
    dCdx = F * wp.transpose(Dm) * r_s_inv
    alpha = 1.0 + k_mu / k_lambda

    # C_Spherical
    # r_s = wp.sqrt(wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3))
    # r_s_inv = 1.0/r_s
    # C = r_s - wp.sqrt(3.0)
    # dCdx = F*wp.transpose(Dm)*r_s_inv
    # alpha = 1.0

    # C_D
    # r_s = wp.sqrt(wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3))
    # C = r_s*r_s - 3.0
    # dCdx = F*wp.transpose(Dm)*2.0
    # alpha = 1.0

    grad1 = wp.vec3(dCdx[0, 0], dCdx[1, 0], dCdx[2, 0])
    grad2 = wp.vec3(dCdx[0, 1], dCdx[1, 1], dCdx[2, 1])
    grad3 = wp.vec3(dCdx[0, 2], dCdx[1, 2], dCdx[2, 2])
    grad0 = (grad1 + grad2 + grad3) * (0.0 - 1.0)

    denom = (
        wp.dot(grad0, grad0) * w0 + wp.dot(grad1, grad1) * w1 + wp.dot(grad2, grad2) * w2 + wp.dot(grad3, grad3) * w3
    )
    multiplier = C / (denom + 1.0 / (k_mu * dt * dt * rest_volume))

    delta0 = grad0 * multiplier
    delta1 = grad1 * multiplier
    delta2 = grad2 * multiplier
    delta3 = grad3 * multiplier

    # hydrostatic part
    J = wp.determinant(F)

    C_vol = J - alpha
    # dCdx = wp.matrix_from_cols(wp.cross(f2, f3), wp.cross(f3, f1), wp.cross(f1, f2))*wp.transpose(Dm)

    # grad1 = wp.vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0])
    # grad2 = wp.vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1])
    # grad3 = wp.vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2])
    # grad0 = (grad1 + grad2 + grad3)*(0.0 - 1.0)

    s = inv_rest_volume / 6.0
    grad1 = wp.cross(x20, x30) * s
    grad2 = wp.cross(x30, x10) * s
    grad3 = wp.cross(x10, x20) * s
    grad0 = -(grad1 + grad2 + grad3)

    denom = (
        wp.dot(grad0, grad0) * w0 + wp.dot(grad1, grad1) * w1 + wp.dot(grad2, grad2) * w2 + wp.dot(grad3, grad3) * w3
    )
    multiplier = C_vol / (denom + 1.0 / (k_lambda * dt * dt * rest_volume))

    delta0 += grad0 * multiplier
    delta1 += grad1 * multiplier
    delta2 += grad2 * multiplier
    delta3 += grad3 * multiplier

    # apply forces
    wp.atomic_sub(delta, i, delta0 * w0 * relaxation)
    wp.atomic_sub(delta, j, delta1 * w1 * relaxation)
    wp.atomic_sub(delta, k, delta2 * w2 * relaxation)
    wp.atomic_sub(delta, l, delta3 * w3 * relaxation)


@wp.kernel
def apply_particle_deltas(
    x_orig: wp.array(dtype=wp.vec3),
    x_pred: wp.array(dtype=wp.vec3),
    v_pred: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    particle_mass: wp.array(dtype=float),
    delta: wp.array(dtype=wp.vec3),
    dt: float,
    v_max: float,
    x_out: wp.array(dtype=wp.vec3),
    v_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    # Static particles (mass=0) don't move - just preserve current state
    if particle_mass[tid] == 0.0:
        x_out[tid] = x_pred[tid]
        v_out[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    # Inactive particles do not move
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return

    x0 = x_orig[tid]
    xp = x_pred[tid]
    vp = v_pred[tid]

    # constraint deltas
    d = delta[tid]

    '''
    Previously: 
        x_new =  xp + d
        v_new = (x_new - x0)/dt
    But this leads to inaccurate results for long horizon simulation (such as t=10s). 
    Assume d = 0, then x_new = xp, and v_new = (xp_x0)/dt which must be = vp 
    But due to numerical errors it is not exactly equal to vp, leading to cumulative errors over time.
    For example, if d = 0, then v_new = vp exactly. and x_new = xp exactly.
    '''
    v_new = vp + d/dt
    x_new = xp + d

    # enforce velocity limit to prevent instability
    v_new_mag = wp.length(v_new)
    if v_new_mag > v_max:
        v_new *= v_max / v_new_mag

    x_out[tid] = x_new
    v_out[tid] = v_new


@wp.kernel
def enforce_momemntum_conservation_tiled(
    x_pred: wp.array(dtype=wp.vec3),
    v_pred: wp.array(dtype=wp.vec3),
    group_mass: wp.array(dtype=float),
    particle_mass: wp.array(dtype=float),
    target_P: wp.array(dtype=wp.vec3),
    target_L: wp.array(dtype=wp.vec3),
    dt: float,
    group_particle_start: wp.array(dtype=wp.int32),
    group_particle_count: wp.array(dtype=wp.int32),
    group_particles_flat: wp.array(dtype=wp.int32),
    x_out: wp.array(dtype=wp.vec3),
    v_out: wp.array(dtype=wp.vec3),
):
    """
    Tile-based momentum conservation: one block per group.
    Each thread strides over particles, then tile_reduce cooperatively.
    Supports any number of particles per group.
    Launch with dim=(num_groups, block_dim), block_dim=block_dim.
    """
    group_id, lane = wp.tid()
    start_idx = group_particle_start[group_id]
    num_particles = group_particle_count[group_id]
    M = group_mass[group_id]
    bd = wp.block_dim()

    # --- Phase 1: Compute current linear momentum Pprime ---
    acc_Pprime = wp.vec3(0.0)
    p = lane
    while p < num_particles:
        idx = group_particles_flat[start_idx + p]
        acc_Pprime += particle_mass[idx] * v_pred[idx]
        p += bd

    Pprime = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(acc_Pprime, preserve_type=True)), 0)

    # Linear momentum correction (uniform across all particles)
    dv = (target_P[group_id] - Pprime) / M

    # --- Phase 2: Apply linear correction, accumulate com and vcom ---
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

    # --- Phase 3: Compute inertia tensor I and angular momentum Lprime ---
    # Accumulate I as 3 column vectors (I is symmetric but we store full 3x3)
    acc_I_col0 = wp.vec3(0.0)
    acc_I_col1 = wp.vec3(0.0)
    acc_I_col2 = wp.vec3(0.0)
    acc_Lprime = wp.vec3(0.0)

    identity = wp.mat33(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    )

    p = lane
    while p < num_particles:
        idx = group_particles_flat[start_idx + p]
        m = particle_mass[idx]
        r = x_out[idx] - com
        r2 = wp.dot(r, r)
        # I += m * (r2 * identity - outer(r, r))
        I_contrib = m * (r2 * identity - wp.outer(r, r))
        acc_I_col0 += wp.vec3(I_contrib[0, 0], I_contrib[1, 0], I_contrib[2, 0])
        acc_I_col1 += wp.vec3(I_contrib[0, 1], I_contrib[1, 1], I_contrib[2, 1])
        acc_I_col2 += wp.vec3(I_contrib[0, 2], I_contrib[1, 2], I_contrib[2, 2])
        vrel = v_out[idx] - vcom
        acc_Lprime += wp.cross(r, m * vrel)
        p += bd

    s_I0 = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(acc_I_col0, preserve_type=True)), 0)
    s_I1 = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(acc_I_col1, preserve_type=True)), 0)
    s_I2 = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(acc_I_col2, preserve_type=True)), 0)
    Lprime = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(acc_Lprime, preserve_type=True)), 0)

    I = wp.mat33(
        s_I0[0], s_I1[0], s_I2[0],
        s_I0[1], s_I1[1], s_I2[1],
        s_I0[2], s_I1[2], s_I2[2],
    )

    dL = Lprime - target_L[group_id]
    omega_err = wp.inverse(I) @ dL

    # --- Phase 4: Apply angular correction ---
    p = lane
    while p < num_particles:
        idx = group_particles_flat[start_idx + p]
        r = x_out[idx] - com
        v_out[idx] = v_out[idx] - wp.cross(omega_err, r)
        x_out[idx] = x_out[idx] - wp.cross(omega_err, r) * dt
        p += bd


@wp.kernel
def solve_shape_matching_batch_tiled(
    particle_q: wp.array(dtype=wp.vec3),
    particle_q_rest: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    group_mass: wp.array(dtype=float),
    particle_mass: wp.array(dtype=float),
    group_particle_start: wp.array(dtype=wp.int32),
    group_particle_count: wp.array(dtype=wp.int32),
    group_particles_flat: wp.array(dtype=wp.int32),
    delta: wp.array(dtype=wp.vec3),
    P_b4_SM: wp.array(dtype=wp.vec3),
    L_b4_SM: wp.array(dtype=wp.vec3),
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

    # --- Cooperative reduction across the block ---
    t = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(acc_mx, preserve_type=True)), 0) / M
    t0 = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(acc_mx0, preserve_type=True)), 0) / M
    P = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(acc_p, preserve_type=True)), 0)
    vcom = P / M

    # --- Phase 1b: Angular momentum L = sum(cross(r, m * vrel)) ---
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

    P_b4_SM[group_id] = P
    L_b4_SM[group_id] = L

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
        sum_col0[0], sum_col1[0], sum_col2[0],
        sum_col0[1], sum_col1[1], sum_col2[1],
        sum_col0[2], sum_col1[2], sum_col2[2],
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
def old_solve_shape_matching_batch(
    particle_q: wp.array(dtype=wp.vec3),
    particle_q_rest: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    group_mass: wp.array(dtype=float),
    particle_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    group_particle_start: wp.array(dtype=wp.int32),
    group_particle_count: wp.array(dtype=wp.int32),
    group_particles_flat: wp.array(dtype=wp.int32),
    delta: wp.array(dtype=wp.vec3),
    P_b4_SM: wp.array(dtype=wp.vec3),
    L_b4_SM: wp.array(dtype=wp.vec3),
):
    """
    Solve shape matching constraints for a batch of groups.
    
    Args:
        particle_q: Current particle positions
        particle_q_rest: Rest particle positions
        particle_mass: Particle masses 
        group_particle_start: Start index of each group's particles in the flat array
        group_particle_count: Number of particles in each group
        group_particles_flat: Flattened array of all group particle indices
        delta: Output delta array to accumulate results
    """

    # Each thread handles one group
    group_id = wp.tid()

    start_idx = group_particle_start[group_id]
    num_particles = group_particle_count[group_id]


    M = group_mass[group_id]
    t = wp.vec3(0.0)
    t0 = wp.vec3(0.0)
    P = wp.vec3(0.0)
    L_origin = wp.vec3(0.0)

    for p in range(num_particles):
        idx = group_particles_flat[start_idx + p]
        if (particle_flags[idx] & ParticleFlags.ACTIVE) == 0:
            continue
        m = particle_mass[idx]
        x = particle_q[idx]
        x0 = particle_q_rest[idx]
        t += m * x
        t0 += m * x0
        v = particle_qd[idx]
        p_linear = m * v
        P += p_linear
        L_origin += wp.cross(x, p_linear)

    t = t / M
    t0 = t0 / M
    L = L_origin - wp.cross(t, P)
    P_b4_SM[group_id] = P
    L_b4_SM[group_id] = L

    # covariance A
    A = wp.mat33(0.0)
    for p in range(num_particles):
        idx = group_particles_flat[start_idx + p]
        if (particle_flags[idx] & ParticleFlags.ACTIVE) == 0:
            continue
        m = particle_mass[idx]
        x = particle_q[idx]
        x0 = particle_q_rest[idx]
        pi = x - t
        qi = x0 - t0
        A += wp.outer(pi, qi) * m

    # polar decomposition via SVD
    U = wp.mat33()
    S = wp.vec3()
    V = wp.mat33()
    wp.svd3(A, U, S, V)
    R = U @ wp.transpose(V)

    if (wp.determinant(R) < 0.0):
        U[:,2] = -U[:,2]
        R = U @ wp.transpose(V)

    for p in range(num_particles):
        idx = group_particles_flat[start_idx + p]
        if (particle_flags[idx] & ParticleFlags.ACTIVE) == 0:
            continue
        x0 = particle_q_rest[idx]
        x = particle_q[idx]
        goal = R @ (x0 - t0) + t
        dx = (goal - x)
        wp.atomic_add(delta, idx, dx)


@wp.kernel
def old_enforce_momemntum_conservation(
    x_pred: wp.array(dtype=wp.vec3),
    v_pred: wp.array(dtype=wp.vec3),
    group_mass: wp.array(dtype=float), 
    particle_flags: wp.array(dtype=wp.int32),
    particle_mass: wp.array(dtype=float),
    target_P: wp.array(dtype=wp.vec3),
    target_L: wp.array(dtype=wp.vec3),
    dt: float,
    group_particle_start: wp.array(dtype=wp.int32),
    group_particle_count: wp.array(dtype=wp.int32),
    group_particles_flat: wp.array(dtype=wp.int32),
    x_out: wp.array(dtype=wp.vec3),
    v_out: wp.array(dtype=wp.vec3),
):
    '''
    This kernel enforces momentum conservation after the shape matching algorithm is called. 
    x_pred and v_pred are the predicted positions and velocities after shape matching.
    x_out and v_out are the output positions and velocities after momentum correction.
    target_P and target_L are the linear and angular momentum before shape matching is called.
    By enforcing the momentum to match the target values, we ensure that the shape matching does not introduce any artificial momentum changes.
    Args:
        x_pred: Predicted particle positions
        v_pred: Predicted particle velocities
        target_P: Target linear momentum for each group
        target_L: Target angular momentum for each group
        x_out: Output particle positions after momentum correction
        v_out: Output particle velocities after momentum correction
    '''
    group_id = wp.tid()
    start_idx = group_particle_start[group_id]
    num_particles = group_particle_count[group_id]
    M = group_mass[group_id]

    # Compute current linear momentum
    Pprime = wp.vec3(0.0)
    for p in range(num_particles):
        idx = group_particles_flat[start_idx + p]
        if (particle_flags[idx] & ParticleFlags.ACTIVE) == 0:
            continue
        Pprime += particle_mass[idx] * v_pred[idx]

    # distribute linear momentum correction
    dv = (target_P[group_id] - Pprime) / M
    com = wp.vec3(0.0)
    vcom = wp.vec3(0.0)
    for p in range(num_particles):
        idx = group_particles_flat[start_idx + p]
        if (particle_flags[idx] & ParticleFlags.ACTIVE) == 0:
            continue
        v_out[idx] = v_pred[idx] + dv
        x_out[idx] = x_pred[idx] + dv * dt
        com += particle_mass[idx] * x_out[idx]
        vcom += particle_mass[idx] * v_out[idx]

    com = com / M # compute center of mass using corrected positions
    vcom = vcom / M # compute center of mass velocity
    
    identity = wp.mat33(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    )
    
    I = wp.mat33(0.0) # Inertia tensor
    Lprime = wp.vec3(0.0) # Angular momentum
    for p in range(num_particles):
        idx = group_particles_flat[start_idx + p]
        if (particle_flags[idx] & ParticleFlags.ACTIVE) == 0:
            continue
        m = particle_mass[idx]
        r = x_out[idx] - com
        r2 = wp.dot(r, r)
        I += m * (r2 * identity - wp.outer(r, r))
        vrel = v_out[idx] - vcom
        Lprime += wp.cross(r, m * vrel)

    dL = Lprime - target_L[group_id]
    omega_err = wp.inverse(I) @ dL

    for p in range(num_particles):
        idx = group_particles_flat[start_idx + p]
        if (particle_flags[idx] & ParticleFlags.ACTIVE) == 0:
            continue
        r = x_out[idx] - com
        v_out[idx] = v_out[idx] - wp.cross(omega_err, r)
        x_out[idx] = x_out[idx] - wp.cross(omega_err, r) * dt        