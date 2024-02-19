import taichi as ti
import torch


# @ti.dataclass
# class MPMModelStruct:
#     ####### essential #######
#     batch_size: int

#     grid_lim_x: float
#     grid_lim_y: float
#     grid_lim_z: float
#     n_grid_x: int
#     n_grid_y: int
#     n_grid_z: int

#     n_particles: int
#     dx: float
#     inv_dx: float
#     # mu: ti.types.ndarray(dtype=float, ndim=2)  # type: ignore
#     # lam: ti.types.ndarray(dtype=float, ndim=2)  # type: ignore
#     # E: ti.types.ndarray(dtype=float, ndim=2)  # type: ignore
#     # nu: ti.types.ndarray(dtype=float, ndim=2)  # type: ignore
#     material: int

#     ######## for plasticity ####
#     # yield_stress: ti.types.ndarray(dtype=float, ndim=2)  # type: ignore
#     friction_angle: float
#     alpha: float
#     # gravitational_accelaration: ti.math.vec3
#     hardening: float
#     xi: float
#     plastic_viscosity: float
#     softening: float

#     ####### for damping
#     rpic_damping: float
#     grid_v_damping_scale: float

#     ####### for PhysGaussian: covariance
#     update_cov_with_F: int

# @ti.dataclass
# class MPMStateStruct:
#     ###### essential #####
#     # particle
#     particle_x: ti.types.ndarray(dtype=ti.math.vec3, ndim=2)  # type: ignore # current position
#     particle_v: ti.types.ndarray(dtype=ti.math.vec3, ndim=2)  # type: ignore # particle velocity
#     particle_F: ti.types.ndarray(dtype=wp.mat33, ndim=2)  # type: ignore # particle elastic deformation gradient
#     particle_init_cov: ti.types.ndarray(dtype=float, ndim=2)  # type: ignore # initial covariance matrix
#     particle_cov: ti.types.ndarray(dtype=float, ndim=2)  # type: ignore # current covariance matrix
#     particle_F_trial: ti.types.ndarray(dtype=wp.mat33, ndim=2)  # type: ignore # apply return mapping on this to obtain elastic def grad
#     particle_R: ti.types.ndarray(dtype=wp.mat33, ndim=2)  # type: ignore # rotation matrix
#     particle_stress: ti.types.ndarray(dtype=wp.mat33, ndim=2)  # type: ignore # Kirchoff stress, elastic stress
#     particle_C: ti.types.ndarray(dtype=wp.mat33, ndim=2)  # type: ignore 
#     particle_vol: ti.types.ndarray(dtype=float, ndim=2)  # type: ignore # current volume
#     particle_mass: ti.types.ndarray(dtype=float, ndim=2)  # type: ignore # mass
#     particle_density: ti.types.ndarray(dtype=float, ndim=2)  # type: ignore # density
#     particle_Jp: ti.types.ndarray(dtype=float, ndim=2)  # type: ignore 

#     particle_selection: ti.types.ndarray(dtype=int, ndim=2)  # type: ignore # only particle_selection[p] = 0 will be simulated

#     # grid
#     grid_m: ti.types.ndarray(dtype=float, ndim=4) # type: ignore
#     grid_v_in: ti.types.ndarray(dtype=ti.math.vec3, ndim=4)  # type: ignore # grid node momentum/velocity
#     grid_v_out: ti.types.ndarray(dtype=ti.math.vec3, ndim=4)  # type: ignore # grid node momentum/velocity, after grid update

# for various boundary conditions
@ti.dataclass
class Dirichlet_collider:
    point: ti.math.vec3
    normal: ti.math.vec3
    direction: ti.math.vec3

    start_time: float
    end_time: float

    friction: float
    surface_type: int

    velocity: ti.math.vec3

    threshold: float
    reset: int
    index: int

    x_unit: ti.math.vec3
    y_unit: ti.math.vec3
    radius: float
    v_scale: float
    width: float
    height: float
    length: float
    R: float

    size: ti.math.vec3

    horizontal_axis_1: ti.math.vec3
    horizontal_axis_2: ti.math.vec3
    half_height_and_radius: ti.math.vec2

# @ti.dataclass
# class Impulse_modifier:
#     # this needs to be changed for each different BC!
#     point: ti.math.vec3
#     normal: ti.math.vec3
#     start_time: float
#     end_time: float
#     force: ti.math.vec3
#     forceTimesDt: ti.math.vec3
#     numsteps: int

#     # point: ti.math.vec3
#     size: ti.math.vec3
#     mask: ti.types.ndarray(dtype=int, ndim=2)  # type: ignore

# @ti.dataclass
# class MPMtailoredStruct:
#     # this needs to be changed for each different BC!
#     point: ti.math.vec3
#     normal: ti.math.vec3
#     start_time: float
#     end_time: float
#     friction: float
#     surface_type: int
#     velocity: ti.math.vec3
#     threshold: float
#     reset: int

#     point_rotate: ti.math.vec3
#     normal_rotate: ti.math.vec3
#     x_unit: ti.math.vec3
#     y_unit: ti.math.vec3
#     radius: float
#     v_scale: float
#     width: float
#     point_plane: ti.math.vec3
#     normal_plane: ti.math.vec3
#     velocity_plane: ti.math.vec3
#     threshold_plane: float

# @ti.dataclass
# class MaterialParamsModifier:
#     point: ti.math.vec3
#     size: ti.math.vec3
#     E: float
#     nu: float
#     density: float

# @ti.dataclass
# class ParticleVelocityModifier:
#     point: ti.math.vec3
#     normal: ti.math.vec3
#     half_height_and_radius: ti.math.vec2
#     rotation_scale: float
#     translation_scale: float

#     size: ti.math.vec3

#     horizontal_axis_1: ti.math.vec3
#     horizontal_axis_2: ti.math.vec3
    
#     start_time: float

#     end_time: float

#     velocity: ti.math.vec3

#     mask: ti.types.ndarray(dtype=int, ndim=2)  # type: ignore



@ti.kernel
def set_vec3_to_zero(target_array: ti.template()):  # type: ignore
    for bid, pid in target_array:
        target_array[bid, pid] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)


@ti.kernel
def set_mat33_to_identity(target_array: ti.template()):  # type: ignore
    for bid, pid in target_array:
        target_array[bid, pid] = ti.Matrix.identity(ti.f32, 3)


@ti.kernel
def add_identity_to_mat33(target_array: ti.template()):  # type: ignore
    for bid, pid in target_array:
        target_array[bid, pid] = target_array[bid, pid] + ti.Matrix.identity(ti.f32, 3)


@ti.kernel
def subtract_identity_to_mat33(target_array: ti.template()):  # type: ignore
    for bid, pid in target_array:
        target_array[bid, pid] = target_array[bid, pid] - ti.Matrix.identity(ti.f32, 3)


@ti.kernel
def add_vec3_to_vec3(
    first_array: ti.template(), second_array: ti.template()  # type: ignore
):
    for bid, pid in first_array:
        first_array[bid, pid] = first_array[bid, pid] + second_array[bid, pid]


@ti.kernel
def set_value_to_float_array(target_array: ti.template(), value: float):  # type: ignore
    for bid, pid in target_array:
        target_array[bid, pid] = value


@ti.kernel
def get_float_array_product(
    arrayA: ti.template(),  # type: ignore
    arrayB: ti.template(),  # type: ignore
    arrayC: ti.template(),  # type: ignore
):
    for bid, pid in arrayA:
        arrayC[bid, pid] = arrayA[bid, pid] * arrayB[bid, pid]


# def torch2warp_quat(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
#     assert t.is_contiguous()
#     if t.dtype != torch.float32 and t.dtype != torch.int32:
#         raise RuntimeError(
#             "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
#         )
#     assert t.shape[2] == 4
#     a = warp.types.array(
#         ptr=t.data_ptr(),
#         dtype=wp.quat,
#         shape=(t.shape[0], t.shape[1]),
#         copy=False,
#         owner=False,
#         requires_grad=t.requires_grad,
#         # device=t.device.type)
#         device=dvc,
#     )
#     a.tensor = t
#     return a

# def torch2warp_float(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
#     assert t.is_contiguous()
#     if t.dtype != torch.float32 and t.dtype != torch.int32:
#         raise RuntimeError(
#             "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
#         )
#     a = warp.types.array(
#         ptr=t.data_ptr(),
#         dtype=warp.types.float32,
#         shape=(t.shape[0], t.shape[1]),
#         copy=False,
#         owner=False,
#         requires_grad=t.requires_grad,
#         # device=t.device.type)
#         device=dvc,
#     )
#     a.tensor = t
#     return a

# def torch2warp_vec3(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
#     assert t.is_contiguous()
#     if t.dtype != torch.float32 and t.dtype != torch.int32:
#         raise RuntimeError(
#             "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
#         )
#     assert t.shape[2] == 3
#     a = warp.types.array(
#         ptr=t.data_ptr(),
#         dtype=warp.types.vector(3, dtype=float)  # type: ignore,
#         shape=(t.shape[0], t.shape[1]),
#         copy=False,
#         owner=False,
#         requires_grad=t.requires_grad,
#         # device=t.device.type)
#         device=dvc,
#     )
#     a.tensor = t
#     return a


# def torch2warp_mat33(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
#     assert t.is_contiguous()
#     if t.dtype != torch.float32 and t.dtype != torch.int32:
#         raise RuntimeError(
#             "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
#         )
#     assert t.shape[2] == 3
#     a = warp.types.array(
#         ptr=t.data_ptr(),
#         dtype=wp.mat33,
#         shape=(t.shape[0], t.shape[1]),
#         copy=False,
#         owner=False,
#         requires_grad=t.requires_grad,
#         # device=t.device.type)
#         device=dvc,
#     )
#     a.tensor = t
#     return a
