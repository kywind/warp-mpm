import warp as wp
import warp.torch
import torch


@wp.struct
class MPMModelStruct:
    ####### essential #######
    batch_size: int

    grid_lim_x: float
    grid_lim_y: float
    grid_lim_z: float
    n_grid_x: int
    n_grid_y: int
    n_grid_z: int

    n_particles: int
    dx: float
    inv_dx: float
    mu: wp.array(dtype=float, ndim=2)  # type: ignore
    lam: wp.array(dtype=float, ndim=2)  # type: ignore
    E: wp.array(dtype=float, ndim=2)  # type: ignore
    nu: wp.array(dtype=float, ndim=2)  # type: ignore
    material: int

    ######## for plasticity ####
    yield_stress: wp.array(dtype=float, ndim=2)  # type: ignore
    friction_angle: float
    alpha: float
    gravitational_accelaration: wp.vec3
    hardening: float
    xi: float
    plastic_viscosity: float
    softening: float

    ####### for damping
    rpic_damping: float
    grid_v_damping_scale: float

    ####### for PhysGaussian: covariance
    update_cov_with_F: int


@wp.struct
class MPMStateStruct:
    ###### essential #####
    # particle
    particle_x: wp.array(dtype=wp.vec3, ndim=2)  # type: ignore # current position
    particle_v: wp.array(dtype=wp.vec3, ndim=2)  # type: ignore # particle velocity
    particle_F: wp.array(dtype=wp.mat33, ndim=2)  # type: ignore # particle elastic deformation gradient
    particle_init_cov: wp.array(dtype=float, ndim=2)  # type: ignore # initial covariance matrix
    particle_cov: wp.array(dtype=float, ndim=2)  # type: ignore # current covariance matrix
    particle_F_trial: wp.array(dtype=wp.mat33, ndim=2)  # type: ignore # apply return mapping on this to obtain elastic def grad
    particle_R: wp.array(dtype=wp.mat33, ndim=2)  # type: ignore # rotation matrix
    particle_stress: wp.array(dtype=wp.mat33, ndim=2)  # type: ignore # Kirchoff stress, elastic stress
    particle_C: wp.array(dtype=wp.mat33, ndim=2)  # type: ignore 
    particle_vol: wp.array(dtype=float, ndim=2)  # type: ignore # current volume
    particle_mass: wp.array(dtype=float, ndim=2)  # type: ignore # mass
    particle_density: wp.array(dtype=float, ndim=2)  # type: ignore # density
    particle_Jp: wp.array(dtype=float, ndim=2)  # type: ignore 

    particle_selection: wp.array(dtype=int, ndim=2)  # type: ignore # only particle_selection[p] = 0 will be simulated

    # grid
    grid_m: wp.array(dtype=float, ndim=4) # type: ignore
    grid_v_in: wp.array(dtype=wp.vec3, ndim=4)  # type: ignore # grid node momentum/velocity
    grid_v_out: wp.array(dtype=wp.vec3, ndim=4)  # type: ignore # grid node momentum/velocity, after grid update


# for various boundary conditions
@wp.struct
class Dirichlet_collider:
    point: wp.vec3
    normal: wp.vec3
    direction: wp.vec3

    start_time: float
    end_time: float

    friction: float
    surface_type: int

    velocity: wp.vec3

    threshold: float
    reset: int
    index: int

    x_unit: wp.vec3
    y_unit: wp.vec3
    radius: float
    v_scale: float
    width: float
    height: float
    length: float
    R: float

    size: wp.vec3

    horizontal_axis_1: wp.vec3
    horizontal_axis_2: wp.vec3
    half_height_and_radius: wp.vec2
    


@wp.struct
class Impulse_modifier:
    # this needs to be changed for each different BC!
    point: wp.vec3
    normal: wp.vec3
    start_time: float
    end_time: float
    force: wp.vec3
    forceTimesDt: wp.vec3
    numsteps: int

    # point: wp.vec3
    size: wp.vec3
    mask: wp.array(dtype=int, ndim=2)  # type: ignore


@wp.struct
class MPMtailoredStruct:
    # this needs to be changed for each different BC!
    point: wp.vec3
    normal: wp.vec3
    start_time: float
    end_time: float
    friction: float
    surface_type: int
    velocity: wp.vec3
    threshold: float
    reset: int

    point_rotate: wp.vec3
    normal_rotate: wp.vec3
    x_unit: wp.vec3
    y_unit: wp.vec3
    radius: float
    v_scale: float
    width: float
    point_plane: wp.vec3
    normal_plane: wp.vec3
    velocity_plane: wp.vec3
    threshold_plane: float

@wp.struct
class MaterialParamsModifier:
    point: wp.vec3
    size: wp.vec3
    E: float
    nu: float
    density: float

@wp.struct
class ParticleVelocityModifier:
    point: wp.vec3
    normal: wp.vec3
    half_height_and_radius: wp.vec2
    rotation_scale: float
    translation_scale: float

    size: wp.vec3

    horizontal_axis_1: wp.vec3
    horizontal_axis_2: wp.vec3
    
    start_time: float

    end_time: float

    velocity: wp.vec3

    mask: wp.array(dtype=int, ndim=2)  # type: ignore



@wp.kernel
def set_vec3_to_zero(target_array: wp.array(dtype=wp.vec3, ndim=2)):  # type: ignore
    bid, pid = wp.tid()
    target_array[bid, pid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def set_mat33_to_identity(target_array: wp.array(dtype=wp.mat33, ndim=2)):  # type: ignore
    bid, pid = wp.tid()
    target_array[bid, pid] = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@wp.kernel
def add_identity_to_mat33(target_array: wp.array(dtype=wp.mat33, ndim=2)):  # type: ignore
    bid, pid = wp.tid()
    target_array[bid, pid] = wp.add(
        target_array[bid, pid], wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )


@wp.kernel
def subtract_identity_to_mat33(target_array: wp.array(dtype=wp.mat33, ndim=2)):  # type: ignore
    bid, pid = wp.tid()
    target_array[bid, pid] = wp.sub(
        target_array[bid, pid], wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )


@wp.kernel
def add_vec3_to_vec3(
    first_array: wp.array(dtype=wp.vec3, ndim=2), second_array: wp.array(dtype=wp.vec3, ndim=2)  # type: ignore
):
    bid, pid = wp.tid()
    first_array[bid, pid] = wp.add(first_array[bid, pid], second_array[bid, pid])


@wp.kernel
def set_value_to_float_array(target_array: wp.array(dtype=float, ndim=2), value: float):  # type: ignore
    bid, pid = wp.tid()
    target_array[bid, pid] = value


@wp.kernel
def get_float_array_product(
    arrayA: wp.array(dtype=float, ndim=2),  # type: ignore
    arrayB: wp.array(dtype=float, ndim=2),  # type: ignore
    arrayC: wp.array(dtype=float, ndim=2),  # type: ignore
):
    bid, pid = wp.tid()
    arrayC[bid, pid] = arrayA[bid, pid] * arrayB[bid, pid]


def torch2warp_quat(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[2] == 4
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.quat,
        shape=(t.shape[0], t.shape[1]),
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a

def torch2warp_float(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=warp.types.float32,
        shape=(t.shape[0], t.shape[1]),
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a

def torch2warp_vec3(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[2] == 3
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.vec3,
        shape=(t.shape[0], t.shape[1]),
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a


def torch2warp_mat33(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[2] == 3
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.mat33,
        shape=(t.shape[0], t.shape[1]),
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a
