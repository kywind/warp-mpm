import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from engine_utils_taichi import *
from taichi_utils import *
from mpm_utils_taichi import *


class MPM_Simulator_Taichi:

    def __init__(self, n_particles=10, batch_size=2, dx=0.01, grid_lim=[1.0, 1.0, 1.0], device="cuda:0"):
        self.initialize(n_particles, batch_size, dx, grid_lim, device=device)
        self.time_profile = {}


    def initialize(self, n_particles=10, batch_size=2, dx=0.01, grid_lim=[1.0, 1.0, 1.0], device="cuda:0"):
        self.n_particles = n_particles

        # self.mpm_model = MPMModelStruct()

        self.batch_size = batch_size

        self.grid_lim_x = grid_lim[0]
        self.grid_lim_y = grid_lim[1]
        self.grid_lim_z = grid_lim[2]
        self.n_grid_x = int(grid_lim[0] / dx)
        self.n_grid_y = int(grid_lim[1] / dx)
        self.n_grid_z = int(grid_lim[2] / dx)

        self.dx = float(dx)
        self.inv_dx = float(1.0 / dx)

        self.E = ti.field(shape=(batch_size, n_particles), dtype=ti.f32)
        self.nu = ti.field(shape=(batch_size, n_particles), dtype=ti.f32)
        self.mu = ti.field(shape=(batch_size, n_particles), dtype=ti.f32)
        self.lam = ti.field(shape=(batch_size, n_particles), dtype=ti.f32)

        self.update_cov_with_F = False

        # material is used to switch between different elastoplastic models. 0 is jelly
        self.material = 0

        # plasticity parameters
        self.plastic_viscosity = 0.0
        self.softening = 0.1
        self.yield_stress = ti.field(dtype=ti.f32, shape=(batch_size, n_particles))

        # frictional parameters
        self.friction_angle = 25.0
        sin_phi = ti.sin(self.friction_angle / 180.0 * 3.14159265)
        self.alpha = ti.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)

        self.gravitational_accelaration = ti.Vector([0.0, 0.0, 0.0])

        self.rpic_damping = 0.0  # 0.0 if no damping (apic). -1 if pic

        self.grid_v_damping_scale = 1.1  # no dampling if > 1.0

        # self.mpm_state = MPMStateStruct()

        self.particle_x = ti.Vector.field(3, dtype=ti.f32, shape=(batch_size, n_particles))  # current position

        self.particle_v = ti.Vector.field(3, dtype=ti.f32, shape=(batch_size, n_particles))  # particle velocity

        self.particle_F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(batch_size, n_particles))  # particle F elastic

        self.particle_R = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(batch_size, n_particles))  # particle R rotation

        self.particle_init_cov = ti.Vector.field(6, dtype=ti.f32, shape=(batch_size, n_particles))  # initial covariance matrix

        self.particle_cov = ti.Vector.field(6, dtype=ti.f32, shape=(batch_size, n_particles))  # current covariance matrix

        self.particle_F_trial = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(batch_size, n_particles))  # apply return mapping will yield

        self.particle_stress = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(batch_size, n_particles))

        self.particle_vol = ti.field(dtype=ti.f32, shape=(batch_size, n_particles))  # particle volume
        self.particle_mass = ti.field(dtype=ti.f32, shape=(batch_size, n_particles))  # particle mass
        self.particle_density = ti.field(dtype=ti.f32, shape=(batch_size, n_particles))
        self.particle_C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(batch_size, n_particles))
        self.particle_Jp = ti.field(dtype=ti.f32, shape=(batch_size, n_particles))

        self.particle_selection = ti.field(shape=(batch_size, n_particles), dtype=int)
        self.particle_selection.fill(0)

        self.grid_m = ti.field(
            dtype=ti.f32,
            shape=(batch_size, self.n_grid_x, self.n_grid_y, self.n_grid_z),
        )
        self.grid_v_in = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=(batch_size, self.n_grid_x, self.n_grid_y, self.n_grid_z),
        )
        self.grid_v_out = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=(batch_size, self.n_grid_x, self.n_grid_y, self.n_grid_z),
        )

        self.time = 0.0

        self.grid_postprocess = []
        self.collider_params = []
        self.modify_bc = []

        # self.tailored_struct_for_bc = MPMtailoredStruct()
        self.pre_p2g_operations = []
        self.impulse_params = []

        self.particle_velocity_modifiers = []
        self.particle_velocity_modifier_params = []


    # the h5 file should store particle initial position and volume.
    def load_from_sampling(self, sampling_h5, batch_size=2, dx=0.01, grid_lim=[1.0, 1.0, 1.0], fps=-1, device="cuda:0"):
        if not os.path.exists(sampling_h5):
            print("h5 file cannot be found at ", os.getcwd() + sampling_h5)
            exit()

        h5file = h5py.File(sampling_h5, "r")
        x, particle_volume = h5file["x"], h5file["particle_volume"]

        x = x[()].transpose()  # np vector of x # shape now is (n_particles, dim)

        print("Sampling particles are loaded from h5 file. Simulator is re-initialized for the correct n_particles")
        particle_volume = np.squeeze(particle_volume, 0)

        if fps > 0:
            from dgl.geometry import farthest_point_sampler
            particle_tensor = torch.from_numpy(x).float()[None, ...]
            fps_idx_tensor = farthest_point_sampler(particle_tensor, fps, start_idx=np.random.randint(0, x.shape[0]))[0]
            fps_idx = fps_idx_tensor.numpy().astype(np.int32)
            x = x[fps_idx]
            particle_volume = particle_volume[fps_idx]

        self.dim, self.n_particles = x.shape[1], x.shape[0]

        self.initialize(self.n_particles, batch_size, dx, grid_lim, device=device)

        x = np.array(x, dtype=np.float32)[None].repeat(batch_size, axis=0)
        particle_volume = np.array(particle_volume, dtype=np.float32)[None].repeat(batch_size, axis=0)

        self.particle_x.from_numpy(x)  # initialize warp array from np

        # initial velocity is default to zero
        set_vec3_to_zero(self.particle_v)
        # initial velocity is default to zero

        # initial deformation gradient is set to identity
        set_mat33_to_identity(self.particle_F_trial)
        # initial deformation gradient is set to identity

        self.particle_vol.from_numpy(particle_volume)

        print("Particles initialized from sampling file.")
        print("Total particles: ", self.n_particles)


    # shape of tensor_x is (bsz, n, 3); shape of tensor_volume is (bsz, n)
    def load_initial_data_from_torch(self, tensor_x, tensor_volume, tensor_cov=None, batch_size=2, dx=0.01, grid_lim=[1.0, 1.0, 1.0], device="cuda:0"):
        self.dim, self.n_particles = tensor_x.shape[2], tensor_x.shape[1]
        assert tensor_x.shape[0] == tensor_volume.shape[0]
        assert tensor_x.shape[1] == tensor_volume.shape[1]
        # assert tensor_x.shape[0] == tensor_cov.reshape(-1, 6).shape[0]

        batch_size_x = tensor_x.shape[0]
        assert batch_size_x == batch_size

        self.initialize(self.n_particles, batch_size, dx, grid_lim, device=device)

        self.import_particle_x_from_torch(tensor_x, device)
        self.particle_vol.from_numpy(
            tensor_volume.detach().clone().cpu().numpy().astype(np.float32)
        )
        if tensor_cov is not None:
            self.particle_init_cov.from_numpy(
                tensor_cov.reshape(batch_size, -1).detach().clone().cpu().numpy().astype(np.float32)
            )

            if self.update_cov_with_F:
                self.particle_cov = self.particle_init_cov

        # initial velocity is default to zero
        set_vec3_to_zero(self.particle_v)
        # initial velocity is default to zero

        # initial deformation gradient is set to identity
        set_mat33_to_identity(self.particle_F_trial)
        # initial trial deformation gradient is set to identity

        print("Particles initialized from torch data.")
        print("Total particles: ", self.n_particles)


    # must give density. mass will be updated as density * volume
    def set_parameters(self, device="cuda:0", **kwargs):
        self.set_parameters_dict(device, kwargs)


    def set_parameters_dict(self, kwargs={}, device="cuda:0"):
        if "material" in kwargs:
            if kwargs["material"] == "jelly":
                self.material = 0
            elif kwargs["material"] == "metal":
                self.material = 1
            elif kwargs["material"] == "sand":
                self.material = 2
            elif kwargs["material"] == "foam":
                self.material = 3
            elif kwargs["material"] == "snow":
                self.material = 4
            elif kwargs["material"] == "plasticine":
                self.material = 5
            else:
                raise TypeError("Undefined material type")

        if "grid_lim" in kwargs:
            grid_lim = kwargs["grid_lim"]
            self.grid_lim_x = grid_lim[0]
            self.grid_lim_y = grid_lim[1]
            self.grid_lim_z = grid_lim[2]
        if "dx" in kwargs:
            self.dx = kwargs["dx"]
            self.inv_dx = float(1.0 / self.dx)

        self.n_grid_x = int(self.grid_lim_x / self.dx)
        self.n_grid_y = int(self.grid_lim_y / self.dx)
        self.n_grid_z = int(self.grid_lim_z / self.dx)

        self.grid_m = ti.field(
            dtype=ti.f32,
            shape=(self.batch_size, self.n_grid_x, self.n_grid_y, self.n_grid_z),
        )
        self.grid_v_in = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=(self.batch_size, self.n_grid_x, self.n_grid_y, self.n_grid_z),
        )
        self.grid_v_out = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=(self.batch_size, self.n_grid_x, self.n_grid_y, self.n_grid_z),
        )

        if "E" in kwargs:
            set_value_to_float_array(self.E, kwargs["E"])
        if "nu" in kwargs:
            set_value_to_float_array(self.nu, kwargs["nu"])
        if "yield_stress" in kwargs:
            val = kwargs["yield_stress"]
            set_value_to_float_array(self.yield_stress, val)
        if "hardening" in kwargs:
            self.hardening = kwargs["hardening"]
        if "xi" in kwargs:
            self.xi = kwargs["xi"]
        if "friction_angle" in kwargs:
            self.friction_angle = kwargs["friction_angle"]
            sin_phi = ti.sin(self.friction_angle / 180.0 * 3.14159265)
            self.alpha = ti.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)

        if "g" in kwargs:
            self.gravitational_accelaration = ti.Vector([kwargs["g"][0], kwargs["g"][1], kwargs["g"][2]], dt=ti.f32)

        if "density" in kwargs:
            density_value = kwargs["density"]
            set_value_to_float_array(self.particle_density, density_value)
            get_float_array_product(
                self.particle_density,
                self.particle_vol,
                self.particle_mass,
            )
        if "rpic_damping" in kwargs:
            self.rpic_damping = kwargs["rpic_damping"]
        if "plastic_viscosity" in kwargs:
            self.plastic_viscosity = kwargs["plastic_viscosity"]
        if "softening" in kwargs:
            self.softening = kwargs["softening"]
        if "grid_v_damping_scale" in kwargs:
            self.grid_v_damping_scale = kwargs["grid_v_damping_scale"]

        if "additional_material_params" in kwargs:
            raise NotImplementedError("additional_material_params is not implemented yet.")
        #     for params in kwargs["additional_material_params"]:
        #         param_modifier = MaterialParamsModifier()
        #         param_modifier.point = wp.vec3(params["point"])
        #         param_modifier.size = wp.vec3(params["size"])
        #         param_modifier.density = params["density"]
        #         param_modifier.E = params["E"]
        #         param_modifier.nu = params["nu"]
        #         wp.launch(
        #             kernel=apply_additional_params,
        #             dim=(self.batch_size, self.n_particles),
        #             inputs=[self.mpm_state, self, param_modifier],
        #             device=device,
        #         )

        #     wp.launch(
        #         kernel=get_float_array_product,
        #         dim=(self.batch_size, self.n_particles),
        #         inputs=[
        #             self.mpm_state.particle_density,
        #             self.mpm_state.particle_vol,
        #             self.mpm_state.particle_mass,
        #         ],
        #         device=device,
        #     )


    def finalize_mu_lam(self, device="cuda:0"):
        compute_mu_lam_from_E_nu(self.E, self.mu, self.nu, self.lam) 

    def p2g2p(self, step, dt, device="cuda:0"):
        grid_size = (
            self.batch_size,
            self.n_grid_x,
            self.n_grid_y,
            self.n_grid_z,
        )
        zero_grid(self.grid_m, self.grid_v_in, self.grid_v_out)

        # apply pre-p2g operations on particles
        for k in range(len(self.pre_p2g_operations)):
            self.pre_p2g_operations[k](self.time, dt, self.impulse_params[k])

        # apply dirichlet particle v modifier
        for k in range(len(self.particle_velocity_modifiers)):
            self.particle_velocity_modifiers[k](self.time, self.particle_velocity_modifier_params[k])

        # compute stress = stress(returnMap(F_trial))
        # with wp.ScopedTimer(
        #     "compute_stress_from_F_trial",
        #     synchronize=True,
        #     print=False,
        #     dict=self.time_profile,
        # ):
        compute_stress_from_F_trial(
            self.particle_F,
            self.particle_F_trial,
            self.particle_stress,
            self.particle_selection,
            self.lam,
            self.mu,
            self.alpha,
            self.material,
            dt,
        )

        # p2g
        # with wp.ScopedTimer(
        #     "p2g",
        #     synchronize=True,
        #     print=False,
        #     dict=self.time_profile,
        # ):
        p2g_apic_with_stress(
            self.particle_stress,
            self.particle_x,
            self.particle_v,
            self.particle_C,
            self.particle_vol,
            self.particle_mass,
            self.particle_selection,
            self.grid_v_in,
            self.grid_m,
            dt,
            self.dx,
            self.inv_dx,
            self.rpic_damping,
        )  # apply p2g'

        # grid update
        # with wp.ScopedTimer(
        #     "grid_update", synchronize=True, print=False, dict=self.time_profile
        # ):
        grid_normalization_and_gravity(
            self.grid_m,
            self.grid_v_in,
            self.grid_v_out,
            self.gravitational_accelaration,
            dt,
        )

        if self.grid_v_damping_scale < 1.0:
            add_damping_via_grid(
                self.grid_v_out,
                self.grid_v_damping_scale, 
            )

        # apply BC on grid
        # with wp.ScopedTimer(
        #     "apply_BC_on_grid", synchronize=True, print=False, dict=self.time_profile
        # ):
        for k in range(len(self.grid_postprocess)):
            self.grid_postprocess[k](
                self.grid_v_out,
                self.collider_params[k].normal,
                self.collider_params[k].point,
                self.collider_params[k].start_time,
                self.collider_params[k].end_time,
                self.time,
                self.dx,
                self.collider_params[k].surface_type,
                self.collider_params[k].friction,
                dt,
            )
            if self.modify_bc[k] is not None:
                raise NotImplementedError("modify_bc is not implemented yet.")
                # self.modify_bc[k](self.time, dt, self.collider_params[k])

        # g2p
        # with wp.ScopedTimer(
        #     "g2p", synchronize=True, print=False, dict=self.time_profile
        # ):
        assert not self.update_cov_with_F
        g2p(
            self.particle_x,
            self.particle_v,
            self.particle_C,
            self.particle_F,
            self.particle_F_trial,
            self.particle_selection,
            self.grid_v_out,
            self.particle_cov,
            self.inv_dx,
            self.update_cov_with_F,
            dt,
        )  # x, v, C, F_trial are updated

        #### CFL check ####
        # particle_v = self.particle_v.numpy()
        # if np.max(np.abs(particle_v)) > self.mpm_model.dx / dt:
        #     print("max particle v: ", np.max(np.abs(particle_v)))
        #     print("max allowed  v: ", self.mpm_model.dx / dt)
        #     print("does not allow v*dt>dx")
        #     input()
        #### CFL check ####
        self.time = self.time + dt

    # set particle densities to all_particle_densities, 
    # def reset_densities_and_update_masses(self, all_particle_densities, device = "cuda:0"):
    #     all_particle_densities = all_particle_densities.clone().detach()
    #     self.particle_density = torch2warp_float(all_particle_densities, dvc=device)
    #     wp.launch(
    #             kernel=get_float_array_product,
    #             dim=(self.mpm_model.batch_size, self.n_particles),
    #             inputs=[
    #                 self.particle_density,
    #                 self.particle_vol,
    #                 self.particle_mass,
    #             ],
    #             device=device,
    #         )

    # clone = True makes a copy, not necessarily needed
    def import_particle_x_from_torch(self, tensor_x, clone=True, device="cuda:0"):
        if tensor_x is not None:
            if clone:
                tensor_x = tensor_x.clone().detach()
            self.particle_x.from_torch(tensor_x)
            # self.particle_x = torch2warp_vec3(tensor_x, dvc=device)

    # clone = True makes a copy, not necessarily needed
    def import_particle_v_from_torch(self, tensor_v, clone=True, device="cuda:0"):
        if tensor_v is not None:
            if clone:
                tensor_v = tensor_v.clone().detach()
            self.particle_v.from_torch(tensor_v)
            # self.particle_v = torch2warp_vec3(tensor_v, dvc=device)

    # clone = True makes a copy, not necessarily needed
    def import_particle_F_from_torch(self, tensor_F, clone=True, device="cuda:0"):
        if tensor_F is not None:
            if clone:
                tensor_F = tensor_F.clone().detach()
            tensor_F = torch.reshape(tensor_F, (self.mpm_model.batch_size, -1, 3, 3))  # arranged by rowmajor
            self.particle_F.from_torch(tensor_F)
            # self.particle_F = torch2warp_mat33(tensor_F, dvc=device)

    # clone = True makes a copy, not necessarily needed
    def import_particle_C_from_torch(self, tensor_C, clone=True, device="cuda:0"):
        if tensor_C is not None:
            if clone:
                tensor_C = tensor_C.clone().detach()
            tensor_C = torch.reshape(tensor_C, (self.mpm_model.batch_size, -1, 3, 3))  # arranged by rowmajor
            self.particle_C.from_torch(tensor_C)
            # self.particle_C = torch2warp_mat33(tensor_C, dvc=device)

    def export_particle_x_to_torch(self):
        return self.particle_x.to_torch()

    def export_particle_v_to_torch(self):
        return self.particle_v.to_torch()

    def export_particle_F_to_torch(self):
        F_tensor = self.particle_F.to_torch()
        F_tensor = F_tensor.reshape(self.batch_size, -1, 9)
        return F_tensor

    # def export_particle_R_to_torch(self, device="cuda:0"):
    #     compute_R_from_F(self.mpm_state, self)
    #     R_tensor = self.particle_R.to_torch()
    #     R_tensor = R_tensor.reshape(self.batch_size, -1, 9)
    #     return R_tensor

    def export_particle_C_to_torch(self):
        C_tensor = self.particle_C.to_torch()
        C_tensor = C_tensor.reshape(self.batch_size, -1, 9)
        return C_tensor

    # def export_particle_cov_to_torch(self, device="cuda:0"):
    #     if not self.update_cov_with_F:
    #         compute_cov_from_F(self.mpm_state, self)

    #     cov = self.particle_cov.to_torch()
    #     return cov

    def print_time_profile(self):
        print("MPM Time profile:")
        for key, value in self.time_profile.items():
            print(key, sum(value))

    # a surface specified by a point and the normal vector
    def add_surface_collider(
        self,
        point,
        normal,
        surface="sticky",
        friction=0.0,
        start_time=0.0,
        end_time=999.0,
    ):
        point = list(point)
        # Normalize normal
        normal_scale = 1.0 / ti.sqrt(float(sum(x**2 for x in normal)))
        normal = list(normal_scale * x for x in normal)

        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time

        collider_param.point = ti.Vector([point[0], point[1], point[2]])
        collider_param.normal = ti.Vector([normal[0], normal[1], normal[2]])

        if surface == "sticky" and friction != 0:
            raise ValueError("friction must be 0 on sticky surfaces.")
        if surface == "sticky":
            collider_param.surface_type = 0
        elif surface == "slip":
            collider_param.surface_type = 1
        elif surface == "cut":
            collider_param.surface_type = 11
        else:
            collider_param.surface_type = 2
        # frictional
        collider_param.friction = friction

        self.collider_params.append(collider_param)

        @ti.kernel
        def collide(
            grid_v_out: ti.template(),  # type: ignore
            normal: ti.math.vec3,
            point: ti.math.vec3,
            start_time: float,
            end_time: float,
            time: float,
            dx: float,
            surface_type: int,
            friction: float,
            dt: float,
        ):
            for b, grid_x, grid_y, grid_z in grid_v_out:
                if time >= start_time and time < end_time:
                    offset = ti.Vector([
                        float(grid_x) * dx - point[0],
                        float(grid_y) * dx - point[1],
                        float(grid_z) * dx - point[2],
                    ], dt=ti.f32)
                    n = ti.Vector([normal[0], normal[1], normal[2]], dt=ti.f32)
                    dotproduct = offset @ n

                    if dotproduct < 0.0:
                        if surface_type == 0:
                            grid_v_out[b, grid_x, grid_y, grid_z] = ti.Vector([
                                0.0, 0.0, 0.0
                            ])
                        elif surface_type == 11:
                            if (
                                float(grid_z) * dx < 0.4
                                or float(grid_z) * dx > 0.53
                            ):
                                grid_v_out[b, grid_x, grid_y, grid_z] = ti.Vector([
                                    0.0, 0.0, 0.0
                                ])
                            else:
                                v_in = grid_v_out[b, grid_x, grid_y, grid_z]
                                grid_v_out[b, grid_x, grid_y, grid_z] = (
                                    ti.Vector([v_in[0], 0.0, v_in[2]]) * 0.3
                                )
                        else:
                            v = grid_v_out[b, grid_x, grid_y, grid_z]
                            normal_component = v @ n
                            if surface_type == 1:
                                v = (
                                    v - normal_component * n
                                )  # Project out all normal component
                            else:
                                v = (
                                    v - ti.min(normal_component, 0.0) * n
                                )  # Project out only inward normal component
                            if normal_component < 0.0 and ti.math.length(v) > 1e-20:
                                v = ti.max(
                                    0.0, ti.math.length(v) + normal_component * friction
                                ) * ti.math.normalize(
                                    v
                                )  # apply friction here
                            grid_v_out[b, grid_x, grid_y, grid_z] = ti.Vector([
                                0.0, 0.0, 0.0
                            ])

        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)
