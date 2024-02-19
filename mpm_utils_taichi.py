import taichi as ti
from taichi_utils import *
import numpy as np
import math


# compute stress from F
# @ti.func
# def kirchoff_stress_FCR(
#     F: ti.math.mat3, U: ti.math.mat3, V: ti.math.mat3, J: float, mu: float, lam: float
# ):
#     # compute kirchoff stress for FCR model (remember tau = P F^T)
#     R = U * V.transpose()
#     id = ti.math.mat3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
#     return 2.0 * mu * (F - R) * F.transpose() + id * lam * J * (J - 1.0)


# @ti.func
# def kirchoff_stress_neoHookean(
#     F: ti.math.mat3, U: ti.math.mat3, V: ti.math.mat3, J: float, sig: ti.math.vec3, mu: float, lam: float
# ):
#     # compute kirchoff stress for FCR model (remember tau = P F^T)
#     b = ti.math.vec3(sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2])
#     b_hat = b - ti.math.vec3(
#         (b[0] + b[1] + b[2]) / 3.0,
#         (b[0] + b[1] + b[2]) / 3.0,
#         (b[0] + b[1] + b[2]) / 3.0,
#     )
#     tau = mu * J ** (-2.0 / 3.0) * b_hat + lam / 2.0 * (J * J - 1.0) * ti.math.vec3(
#         1.0, 1.0, 1.0
#     )
#     return (
#         U
#         * ti.math.mat3(tau[0], 0.0, 0.0, 0.0, tau[1], 0.0, 0.0, 0.0, tau[2])
#         * V.transpose()
#         * F.transpose()
#     )


# @ti.func
# def kirchoff_stress_StVK(
#     F: ti.math.mat3, U: ti.math.mat3, V: ti.math.mat3, sig: ti.math.vec3, mu: float, lam: float
# ):
#     sig = ti.math.vec3(
#         ti.max(sig[0], 0.01), ti.max(sig[1], 0.01), ti.max(sig[2], 0.01)
#     )  # add this to prevent NaN in extrem cases
#     epsilon = ti.math.vec3(ti.log(sig[0]), ti.log(sig[1]), ti.log(sig[2]))
#     log_sig_sum = ti.log(sig[0]) + ti.log(sig[1]) + ti.log(sig[2])
#     ONE = ti.math.vec3(1.0, 1.0, 1.0)
#     tau = 2.0 * mu * epsilon + lam * log_sig_sum * ONE
#     return (
#         U
#         * ti.math.mat3(tau[0], 0.0, 0.0, 0.0, tau[1], 0.0, 0.0, 0.0, tau[2])
#         * V.transpose()
#         * F.transpose()
#     )


@ti.func
def kirchoff_stress_drucker_prager(
    F: ti.math.mat3, U: ti.math.mat3, V: ti.math.mat3, sig: ti.math.mat3, mu: float, lam: float
):
    log_sig_sum = ti.log(sig[0, 0]) + ti.log(sig[1, 1]) + ti.log(sig[2, 2])
    center00 = 2.0 * mu * ti.log(sig[0, 0]) * (1.0 / sig[0, 0]) + lam * log_sig_sum * (
        1.0 / sig[0, 0]
    )
    center11 = 2.0 * mu * ti.log(sig[1, 1]) * (1.0 / sig[1, 1]) + lam * log_sig_sum * (
        1.0 / sig[1, 1]
    )
    center22 = 2.0 * mu * ti.log(sig[2, 2]) * (1.0 / sig[2, 2]) + lam * log_sig_sum * (
        1.0 / sig[2, 2]
    )
    center = ti.Matrix([[center00, 0.0, 0.0], [0.0, center11, 0.0], [0.0, 0.0, center22]])
    return U @ center @ V.transpose() @ F.transpose()


# @ti.func
# def von_mises_return_mapping(F_trial: ti.math.mat3, model: MPMModelStruct, b: int, p: int):
#     U = ti.math.mat3([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
#     V = ti.math.mat3([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
#     sig_old = ti.math.vec3(0.0)
#     ti.svd3(F_trial, U, sig_old, V)

#     sig = ti.math.vec3(
#         ti.max(sig_old[0], 0.01), ti.max(sig_old[1], 0.01), ti.max(sig_old[2], 0.01)
#     )  # add this to prevent NaN in extrem cases
#     epsilon = ti.math.vec3(ti.log(sig[0]), ti.log(sig[1]), ti.log(sig[2]))
#     temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

#     tau = 2.0 * model.mu[b, p] * epsilon + model.lam[b, p] * (
#         epsilon[0] + epsilon[1] + epsilon[2]
#     ) * ti.math.vec3(1.0, 1.0, 1.0)
#     sum_tau = tau[0] + tau[1] + tau[2]
#     cond = ti.math.vec3(
#         tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
#     )
#     if ti.math.length(cond) > model.yield_stress[b, p]:
#         epsilon_hat = epsilon - ti.math.vec3(temp, temp, temp)
#         epsilon_hat_norm = ti.math.length(epsilon_hat) + 1e-6
#         delta_gamma = epsilon_hat_norm - model.yield_stress[b, p] / (2.0 * model.mu[b, p])
#         epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
#         sig_elastic = ti.math.mat3(
#             ti.exp(epsilon[0]),
#             0.0,
#             0.0,
#             0.0,
#             ti.exp(epsilon[1]),
#             0.0,
#             0.0,
#             0.0,
#             ti.exp(epsilon[2]),
#         )
#         F_elastic = U * sig_elastic * ti.transpose(V)
#         if model.hardening == 1:
#             model.yield_stress[b, p] = (
#                 model.yield_stress[b, p] + 2.0 * model.mu[b, p] * model.xi * delta_gamma
#             )
#         return F_elastic
#     else:
#         return F_trial

# @ti.func
# def von_mises_return_mapping_with_damage(
#     F_trial: ti.math.mat3, model: MPMModelStruct, b: int, p: int
# ):
#     U = ti.math.mat3([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
#     V = ti.math.mat3([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
#     sig_old = ti.math.vec3(0.0)
#     ti.svd3(F_trial, U, sig_old, V)

#     sig = ti.math.vec3(
#         ti.max(sig_old[0], 0.01), ti.max(sig_old[1], 0.01), ti.max(sig_old[2], 0.01)
#     )  # add this to prevent NaN in extrem cases
#     epsilon = ti.math.vec3(ti.log(sig[0]), ti.log(sig[1]), ti.log(sig[2]))
#     temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

#     tau = 2.0 * model.mu[b, p] * epsilon + model.lam[b, p] * (
#         epsilon[0] + epsilon[1] + epsilon[2]
#     ) * ti.math.vec3(1.0, 1.0, 1.0)
#     sum_tau = tau[0] + tau[1] + tau[2]
#     cond = ti.math.vec3(
#         tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
#     )
#     if ti.math.length(cond) > model.yield_stress[b, p]:
#         if model.yield_stress[b, p] <= 0:
#             return F_trial
#         epsilon_hat = epsilon - ti.math.vec3(temp, temp, temp)
#         epsilon_hat_norm = ti.math.length(epsilon_hat) + 1e-6
#         delta_gamma = epsilon_hat_norm - model.yield_stress[b, p] / (2.0 * model.mu[b, p])
#         epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
#         model.yield_stress[b, p] = model.yield_stress[b, p] - model.softening * ti.math.length(
#             (delta_gamma / epsilon_hat_norm) * epsilon_hat
#         )
#         if model.yield_stress[b, p] <= 0:
#             model.mu[b, p] = 0.0
#             model.lam[b, p] = 0.0
#         sig_elastic = ti.math.mat3(
#             ti.exp(epsilon[0]),
#             0.0,
#             0.0,
#             0.0,
#             ti.exp(epsilon[1]),
#             0.0,
#             0.0,
#             0.0,
#             ti.exp(epsilon[2]),
#         )
#         F_elastic = U * sig_elastic * ti.transpose(V)
#         if model.hardening == 1:
#             model.yield_stress[b, p] = (
#                 model.yield_stress[b, p] + 2.0 * model.mu[b, p] * model.xi * delta_gamma
#             )
#         return F_elastic
#     else:
#         return F_trial


# # for toothpaste
# @ti.func
# def viscoplasticity_return_mapping_with_StVK(
#     F_trial: ti.math.mat3, model: MPMModelStruct, b: int, p: int, dt: float
# ):
#     U = ti.math.mat3([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
#     V = ti.math.mat3([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
#     sig_old = ti.math.vec3(0.0)
#     ti.svd3(F_trial, U, sig_old, V)

#     sig = ti.math.vec3(
#         ti.max(sig_old[0], 0.01), ti.max(sig_old[1], 0.01), ti.max(sig_old[2], 0.01)
#     )  # add this to prevent NaN in extrem cases
#     b_trial = ti.math.vec3(sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2])
#     epsilon = ti.math.vec3(ti.log(sig[0]), ti.log(sig[1]), ti.log(sig[2]))
#     trace_epsilon = epsilon[0] + epsilon[1] + epsilon[2]
#     epsilon_hat = epsilon - ti.math.vec3(
#         trace_epsilon / 3.0, trace_epsilon / 3.0, trace_epsilon / 3.0
#     )
#     s_trial = 2.0 * model.mu[b, p] * epsilon_hat
#     s_trial_norm = ti.math.length(s_trial)
#     y = s_trial_norm - ti.sqrt(2.0 / 3.0) * model.yield_stress[b, p]
#     if y > 0:
#         mu_hat = model.mu[b, p] * (b_trial[0] + b_trial[1] + b_trial[2]) / 3.0
#         s_new_norm = s_trial_norm - y / (
#             1.0 + model.plastic_viscosity / (2.0 * mu_hat * dt)
#         )
#         s_new = (s_new_norm / s_trial_norm) * s_trial
#         epsilon_new = 1.0 / (2.0 * model.mu[b, p]) * s_new + ti.math.vec3(
#             trace_epsilon / 3.0, trace_epsilon / 3.0, trace_epsilon / 3.0
#         )
#         sig_elastic = ti.math.mat3(
#             ti.exp(epsilon_new[0]),
#             0.0,
#             0.0,
#             0.0,
#             ti.exp(epsilon_new[1]),
#             0.0,
#             0.0,
#             0.0,
#             ti.exp(epsilon_new[2]),
#         )
#         F_elastic = U * sig_elastic * ti.transpose(V)
#         return F_elastic
#     else:
#         return F_trial


@ti.func
def sand_return_mapping(
    F_trial: ti.math.mat3,
    lam: float,
    mu: float,
    alpha: float,
) -> ti.math.mat3:
    U = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dt=ti.f32)
    V = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dt=ti.f32)
    sig = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dt=ti.f32)
    # sig = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
    U, sig, V = ti.svd(F_trial)

    epsilon = ti.Vector([
        ti.log(ti.max(ti.abs(sig[0, 0]), 1e-14)),
        ti.log(ti.max(ti.abs(sig[1, 1]), 1e-14)),
        ti.log(ti.max(ti.abs(sig[2, 2]), 1e-14)),
    ], dt=ti.f32)
    # sigma_out = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dt=ti.f32)
    tr = epsilon[0] + epsilon[1] + epsilon[2]  # + state.particle_Jp[p]
    epsilon_hat = epsilon - ti.Vector([tr / 3.0, tr / 3.0, tr / 3.0], dt=ti.f32)
    epsilon_hat_norm = ti.math.length(epsilon_hat)
    delta_gamma = (
        epsilon_hat_norm
        + (3.0 * lam + 2.0 * mu)
        / (2.0 * mu)
        * tr
        * alpha
    )

    F_elastic = F_trial

    if delta_gamma <= 0:
        F_elastic = F_trial

    if delta_gamma > 0 and tr > 0:
        F_elastic = U @ V.transpose()

    if delta_gamma > 0 and tr <= 0:
        H = epsilon - epsilon_hat * (delta_gamma / epsilon_hat_norm)
        s_new = ti.Matrix([
            [ti.exp(H[0]), 0., 0.],
            [0., ti.exp(H[1]), 0.],
            [0., 0., ti.exp(H[2])],
        ])
        F_elastic = U @ s_new @ V.transpose()

    return F_elastic


@ti.kernel
def compute_mu_lam_from_E_nu(
    E: ti.template(),  # type: ignore
    mu: ti.template(),  # type: ignore
    nu: ti.template(),  # type: ignore
    lam: ti.template(),  # type: ignore
):
    for b, p in E:
        mu[b, p] = E[b, p] / (2.0 * (1.0 + nu[b, p]))
        lam[b, p] = E[b, p] * nu[b, p]  / ((1.0 + nu[b, p]) * (1.0 - 2.0 * nu[b, p]))


@ti.kernel
def zero_grid(
    grid_m: ti.template(),  # type: ignore
    grid_v_in: ti.template(),  # type: ignore
    grid_v_out: ti.template(),  # type: ignore
):
    for b, grid_x, grid_y, grid_z in grid_m:
        grid_m[b, grid_x, grid_y, grid_z] = 0.0
        grid_v_in[b, grid_x, grid_y, grid_z] = ti.Vector([0.0, 0.0, 0.0])
        grid_v_out[b, grid_x, grid_y, grid_z] = ti.Vector([0.0, 0.0, 0.0])


@ti.func
def compute_dweight(
    inv_dx: float, w: ti.math.mat3, dw: ti.math.mat3, i: int, j: int, k: int
):
    dweight = ti.Vector([
        dw[0, i] * w[1, j] * w[2, k],
        w[0, i] * dw[1, j] * w[2, k],
        w[0, i] * w[1, j] * dw[2, k],
    ])
    return dweight * inv_dx


@ti.func
def update_cov(
    particle_cov: ti.template(),  # type: ignore
    b: int, p: int, grad_v: ti.math.mat3, dt: float
):
    cov_n = ti.Matrix.zero(ti.f32, 3, 3)
    cov_n[0, 0] = particle_cov[b, p][0]
    cov_n[0, 1] = particle_cov[b, p][1]
    cov_n[0, 2] = particle_cov[b, p][2]
    cov_n[1, 0] = particle_cov[b, p][1]
    cov_n[1, 1] = particle_cov[b, p][3]
    cov_n[1, 2] = particle_cov[b, p][4]
    cov_n[2, 0] = particle_cov[b, p][2]
    cov_n[2, 1] = particle_cov[b, p][4]
    cov_n[2, 2] = particle_cov[b, p][5]

    cov_np1 = cov_n + dt * (grad_v @ cov_n + cov_n @ grad_v.transpose())

    particle_cov[b, p][0] = cov_np1[0, 0]
    particle_cov[b, p][1] = cov_np1[0, 1]
    particle_cov[b, p][2] = cov_np1[0, 2]
    particle_cov[b, p][3] = cov_np1[1, 1]
    particle_cov[b, p][4] = cov_np1[1, 2]
    particle_cov[b, p][5] = cov_np1[2, 2]


@ti.kernel
def p2g_apic_with_stress(
    particle_stress: ti.template(),  # type: ignore
    particle_x: ti.template(),  # type: ignore
    particle_v: ti.template(),  # type: ignore
    particle_C: ti.template(),  # type: ignore
    particle_vol: ti.template(),  # type: ignore
    particle_mass: ti.template(),  # type: ignore
    particle_selection: ti.template(),  # type: ignore
    grid_v_in: ti.template(),  # type: ignore
    grid_m: ti.template(),  # type: ignore
    dt: float,
    dx: float,
    inv_dx: float,
    rpic_damping: float,
):
    # input given to p2g:   particle_stress
    #                       particle_x
    #                       particle_v
    #                       particle_C
    for b, p in particle_x:
        if particle_selection[b, p] == 0:
            stress = particle_stress[b, p]
            grid_pos = particle_x[b, p] * inv_dx
            base_pos_x = ti.floor(grid_pos[0] - 0.5)
            base_pos_y = ti.floor(grid_pos[1] - 0.5)
            base_pos_z = ti.floor(grid_pos[2] - 0.5)
            fx = grid_pos - ti.Vector([
                base_pos_x, base_pos_y, base_pos_z
            ], dt=ti.f32)
            wa = 1.5 - fx
            wb = fx - 1.0
            wc = fx - 0.5
            
            wa1 = wa * wa * 0.5
            wb1 = 0. - wb * wb + 0.75
            wc1 = wc * wc * 0.5
            w = ti.Matrix([
                [wa1[0], wb1[0], wc1[0]],
                [wa1[1], wb1[1], wc1[1]],
                [wa1[2], wb1[2], wc1[2]]
            ], dt=ti.f32)

            dw1 = fx - 1.5
            dw2 = -2.0 * (fx - 1.0)
            dw3 = fx - 0.5
            dw = ti.Matrix([
                [dw1[0], dw2[0], dw3[0]],
                [dw1[1], dw2[1], dw3[1]],
                [dw1[2], dw2[2], dw3[2]]
            ], dt=ti.f32)

            for i in range(0, 3):
                for j in range(0, 3):
                    for k in range(0, 3):
                        dpos = (
                            ti.Vector([i, j, k], dt=ti.f32) - fx
                        ) * dx
                        ix = int(base_pos_x + i + 1e-3)
                        iy = int(base_pos_y + j + 1e-3)
                        iz = int(base_pos_z + k + 1e-3)
                        weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                        dweight = compute_dweight(inv_dx, w, dw, i, j, k)
                        C = particle_C[b, p]
                        # if rpic = 0, standard apic
                        C = (1.0 - rpic_damping) * C + rpic_damping / 2.0 * (
                            C - C.transpose()
                        )
                        if rpic_damping < -0.001:
                            # standard pic
                            C = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

                        elastic_force = -particle_vol[b, p] * stress @ dweight
                        v_in_add = (
                            weight
                            * particle_mass[b, p]
                            * (particle_v[b, p] + C @ dpos)
                            + dt * elastic_force
                        )
                        grid_v_in[b, ix, iy, iz] += v_in_add
                        grid_m[b, ix, iy, iz] += weight * particle_mass[b, p]


# add gravity
@ti.kernel
def grid_normalization_and_gravity(
    grid_m: ti.template(),  # type: ignore
    grid_v_in: ti.template(),  # type: ignore
    grid_v_out: ti.template(),  # type: ignore
    gravitational_accelaration: ti.math.vec3, dt: float
):
    for b, grid_x, grid_y, grid_z in grid_m:
        if grid_m[b, grid_x, grid_y, grid_z] > 1e-15:
            v_out = grid_v_in[b, grid_x, grid_y, grid_z] * (
                1.0 / grid_m[b, grid_x, grid_y, grid_z]
            )
            # add gravity
            v_out = v_out + dt * gravitational_accelaration
            grid_v_out[b, grid_x, grid_y, grid_z] = v_out


@ti.kernel
def g2p(
    particle_x: ti.template(),  # type: ignore
    particle_v: ti.template(),  # type: ignore
    particle_C: ti.template(),  # type: ignore
    particle_F: ti.template(),  # type: ignore
    particle_F_trial: ti.template(),  # type: ignore
    particle_selection: ti.template(),  # type: ignore
    grid_v_out: ti.template(),  # type: ignore
    particle_cov: ti.template(),  # type: ignore
    inv_dx: float,
    update_cov_with_F: bool,
    dt: float
):
    for b, p in particle_x:
        if particle_selection[b, p] == 0:
            grid_pos = particle_x[b, p] * inv_dx
            base_pos_x = ti.math.floor(grid_pos[0] - 0.5)
            base_pos_y = ti.math.floor(grid_pos[1] - 0.5)
            base_pos_z = ti.math.floor(grid_pos[2] - 0.5)
            fx = grid_pos - ti.Vector([
                base_pos_x, base_pos_y, base_pos_z
            ], dt=ti.f32)
            wa = ti.Vector([1.5, 1.5, 1.5]) - fx
            wb = fx - ti.Vector([1.0, 1.0, 1.0])
            wc = fx - ti.Vector([0.5, 0.5, 0.5])

            wa1 = wa * wa * 0.5
            wb1 = 0. - wb * wb + 0.75
            wc1 = wc * wc * 0.5
            w = ti.Matrix([
                [wa1[0], wb1[0], wc1[0]],
                [wa1[1], wb1[1], wc1[1]],
                [wa1[2], wb1[2], wc1[2]]
            ], dt=ti.f32)

            dw1 = fx - 1.5
            dw2 = -2.0 * (fx - 1.0)
            dw3 = fx - 0.5
            dw = ti.Matrix([
                [dw1[0], dw2[0], dw3[0]],
                [dw1[1], dw2[1], dw3[1]],
                [dw1[2], dw2[2], dw3[2]]
            ], dt=ti.f32)

            new_v = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
            new_C = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dt=ti.f32)
            new_F = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dt=ti.f32)
            for i in range(0, 3):
                for j in range(0, 3):
                    for k in range(0, 3):
                        ix = int(base_pos_x + i)
                        iy = int(base_pos_y + j)
                        iz = int(base_pos_z + k)
                        dpos = ti.Vector([i, j, k], dt=ti.f32) - fx
                        weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                        grid_v = grid_v_out[b, ix, iy, iz]
                        new_v = new_v + grid_v * weight
                        new_C = new_C + grid_v.outer_product(dpos) * (
                            weight * inv_dx * 4.0
                        )
                        dweight = compute_dweight(inv_dx, w, dw, i, j, k)
                        new_F = new_F + grid_v.outer_product(dweight)

            particle_v[b, p] = new_v
            particle_x[b, p] = particle_x[b, p] + dt * new_v
            particle_C[b, p] = new_C
            I33 = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dt=ti.f32)
            F_tmp = (I33 + new_F * dt) @ particle_F[b, p]
            particle_F_trial[b, p] = F_tmp

            if update_cov_with_F:
                update_cov(particle_cov, b, p, new_F, dt)


# compute (Kirchhoff) stress = stress(returnMap(F_trial))
@ti.kernel
def compute_stress_from_F_trial(
    particle_F: ti.template(),  # type: ignore
    particle_F_trial: ti.template(),  # type: ignore
    particle_stress: ti.template(),  # type: ignore
    particle_selection: ti.template(),  # type: ignore
    lam: ti.template(),  # type: ignore
    mu: ti.template(),  # type: ignore
    alpha: float,  # type: ignore
    material: int,
    dt: float,
):
    for b, p in particle_F:
        if particle_selection[b, p] == 0:
            # # apply return mapping
            # if material == 1:  # metal
            #     particle_F[b, p] = von_mises_return_mapping(
            #         particle_F_trial[b, p], model, b, p
            #     )
            # elif material == 2:  # sand
            particle_F[b, p] = sand_return_mapping(
                particle_F_trial[b, p], lam[b, p], mu[b, p], alpha
            )
            # elif material == 3:  # visplas, with StVk+VM, no thickening
            #     particle_F[b, p] = viscoplasticity_return_mapping_with_StVK(
            #         particle_F_trial[b, p], model, b, p, dt
            #     )
            # elif material == 5:
            #     particle_F[b, p] = von_mises_return_mapping_with_damage(
            #         particle_F_trial[b, p], model, b, p
            #     )
            # else:  # elastic
            #     particle_F[b, p] = particle_F_trial[b, p]

            # also compute stress here
            J = ti.Matrix.determinant(particle_F[b, p])
            U = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            V = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            # sig = ti.Vector.zero(ti.f32, 3)
            sig = ti.Matrix.zero(ti.f32, 3, 3)
            stress = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            U, sig, V = ti.svd(particle_F[b, p])
            # if material == 0 or material == 5:
            #     stress = kirchoff_stress_FCR(
            #         particle_F[b, p], U, V, J, mu[b, p], lam[b, p]
            #     )
            # if material == 1:
            #     stress = kirchoff_stress_StVK(
            #         particle_F[b, p], U, V, sig, mu[b, p], lam[b, p]
            #     )
            if material == 2:
                stress = kirchoff_stress_drucker_prager(
                    particle_F[b, p], U, V, sig, mu[b, p], lam[b, p]
                )
            # if material == 3:
            #     # temporarily use stvk, subject to change
            #     stress = kirchoff_stress_StVK(
            #         particle_F[b, p], U, V, sig, mu[b, p], lam[b, p]
            #     )

            stress = (stress + stress.transpose()) / 2.0  # enfore symmetry
            particle_stress[b, p] = stress


# @ti.kernel
# def compute_cov_from_F(
#     particle_init_cov: ti.template(),  # type: ignore
#     particle_F_trial: ti.template(),  # type: ignore
#     particle_cov: ti.template(),  # type: ignore
# ):
#     for b, p in particle_init_cov:

#         F = particle_F_trial[b, p]

#         init_cov = ti.math.mat3(0.0)
#         init_cov[0, 0] = particle_init_cov[b, p * 6]
#         init_cov[0, 1] = particle_init_cov[b, p * 6 + 1]
#         init_cov[0, 2] = particle_init_cov[b, p * 6 + 2]
#         init_cov[1, 0] = particle_init_cov[b, p * 6 + 1]
#         init_cov[1, 1] = particle_init_cov[b, p * 6 + 3]
#         init_cov[1, 2] = particle_init_cov[b, p * 6 + 4]
#         init_cov[2, 0] = particle_init_cov[b, p * 6 + 2]
#         init_cov[2, 1] = particle_init_cov[b, p * 6 + 4]
#         init_cov[2, 2] = particle_init_cov[b, p * 6 + 5]

#         cov = F @ init_cov @ F.transpose()

#         particle_cov[b, p * 6] = cov[0, 0]
#         particle_cov[b, p * 6 + 1] = cov[0, 1]
#         particle_cov[b, p * 6 + 2] = cov[0, 2]
#         particle_cov[b, p * 6 + 3] = cov[1, 1]
#         particle_cov[b, p * 6 + 4] = cov[1, 2]
#         particle_cov[b, p * 6 + 5] = cov[2, 2]


# @ti.kernel
# def compute_R_from_F(
#     particle_F_trial: ti.template(),  # type: ignore
#     particle_R: ti.template(),  # type: ignore
# ):
#    for b, p in particle_F_trial:

#         F = particle_F_trial[b, p]

#         # polar svd decomposition
#         U = ti.math.mat3(0.0)
#         V = ti.math.mat3(0.0)
#         sig = ti.math.vec3(0.0)
#         ti.svd3(F, U, sig, V)

#         if ti.determinant(U) < 0.0:
#             U[0, 2] = -U[0, 2]
#             U[1, 2] = -U[1, 2]
#             U[2, 2] = -U[2, 2]

#         if ti.determinant(V) < 0.0:
#             V[0, 2] = -V[0, 2]
#             V[1, 2] = -V[1, 2]
#             V[2, 2] = -V[2, 2]

#         # compute rotation matrix
#         R = U * V.transpose()
#         particle_R[b, p] = R.transpose()

@ti.kernel
def add_damping_via_grid(
    grid_v_out: ti.template(),  # type: ignore
    scale: float
):
    for b, grid_x, grid_y, grid_z in grid_v_out:
        grid_v_out[b, grid_x, grid_y, grid_z] = (
            grid_v_out[b, grid_x, grid_y, grid_z] * scale
        )

# @ti.kernel
# def apply_additional_params(
#     state: MPMStateStruct,
#     model: MPMModelStruct,
#     params_modifier: MaterialParamsModifier,
# ):
#     b, p = ti.tid()
#     pos = state.particle_x[b, p]
#     if (
#         pos[0] > params_modifier.point[0] - params_modifier.size[0]
#         and pos[0] < params_modifier.point[0] + params_modifier.size[0]
#         and pos[1] > params_modifier.point[1] - params_modifier.size[1]
#         and pos[1] < params_modifier.point[1] + params_modifier.size[1]
#         and pos[2] > params_modifier.point[2] - params_modifier.size[2]
#         and pos[2] < params_modifier.point[2] + params_modifier.size[2]
#     ):
#         model.E[b, p] = params_modifier.E
#         model.nu[b, p] = params_modifier.nu
#         state.particle_density[b, p] = params_modifier.density


# @ti.kernel
# def selection_add_impulse_on_particles(state: MPMStateStruct, impulse_modifier: Impulse_modifier):
#     b, p = ti.tid()
#     offset = state.particle_x[b, p] - impulse_modifier.point
#     if (
#         ti.abs(offset[0]) < impulse_modifier.size[0]
#         and ti.abs(offset[1]) < impulse_modifier.size[1]
#         and ti.abs(offset[2]) < impulse_modifier.size[2]
#                 ):
#         impulse_modifier.mask[b, p] = 1 
#     else:
#         impulse_modifier.mask[b, p] = 0


# @ti.kernel
# def selection_enforce_particle_velocity_translation(state: MPMStateStruct, velocity_modifier: ParticleVelocityModifier):
#     b, p = ti.tid()
#     offset = state.particle_x[b, p] - velocity_modifier.point
#     if (
#         ti.abs(offset[0]) < velocity_modifier.size[0]
#         and ti.abs(offset[1]) < velocity_modifier.size[1]
#         and ti.abs(offset[2]) < velocity_modifier.size[2]
#                 ):
#         velocity_modifier.mask[b, p] = 1 
#     else:
#         velocity_modifier.mask[b, p] = 0

        
# @ti.kernel
# def selection_enforce_particle_velocity_cylinder(state: MPMStateStruct, velocity_modifier: ParticleVelocityModifier):
#     b, p = ti.tid()
#     offset = state.particle_x[b, p] - velocity_modifier.point

#     vertical_distance = ti.abs(ti.dot(offset, velocity_modifier.normal))

#     horizontal_distance = ti.math.length(offset - ti.dot(offset, velocity_modifier.normal) * velocity_modifier.normal)
#     if (
#         vertical_distance < velocity_modifier.half_height_and_radius[0]
#         and horizontal_distance < velocity_modifier.half_height_and_radius[1]
#                 ):
#         velocity_modifier.mask[b, p] = 1 
#     else:
#         velocity_modifier.mask[b, p] = 0

        