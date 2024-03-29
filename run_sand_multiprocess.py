
import warp as wp
from mpm_solver_warp import MPM_Simulator_WARP
from engine_utils import *
import torch
import time
import multiprocessing as mp


def run_sim_process(pid):
    wp.init()
    wp.config.verify_cuda = True
    dvc = "cuda:0"
    batch_size = 10

    print("PID:", pid, "started")
    mpm_solver = MPM_Simulator_WARP(n_particles=10, batch_size=batch_size, dx=0.01, device=dvc)
    print("PID:", pid, "initialized")

    # You can either load sampling data from an external h5 file, containing initial position (n,3) and particle_volume (n,)
    mpm_solver.load_from_sampling("sim_data/sand_column.h5", batch_size=batch_size, dx=0.01, device=dvc) 

    # Or load from torch tensor (also position and volume)
    # Here we borrow the data from h5, but you can use your own
    volume_tensor = torch.ones((batch_size, mpm_solver.n_particles)) * 2.5e-8  # (bsz, n)
    position_tensor = mpm_solver.export_particle_x_to_torch()  # (bsz, n, 3)

    mpm_solver.load_initial_data_from_torch(position_tensor, volume_tensor, batch_size=batch_size, dx=0.01, device=dvc)

    # Note: You must provide 'density=..' to set particle_mass = density * particle_volume

    material_params = {
        'E': 2000,
        'nu': 0.2,
        "material": "sand",
        'friction_angle': 35,
        'g': [0.0, 0.0, -4.0],
        "density": 200.0
    }
    mpm_solver.set_parameters_dict(material_params)

    mpm_solver.finalize_mu_lam() # set mu and lambda from the E and nu input

    mpm_solver.add_surface_collider((0.0, 0.0, 0.13), (0.0,0.0,1.0), 'sticky', 0.0)

    directory_to_save = f'./sim_results_{pid}'

    save_data_at_frame(mpm_solver, directory_to_save, 0, save_to_ply=True, save_to_h5=False)

    time1 = time.time()
    for k in range(1, 500):
        mpm_solver.p2g2p(k, 0.002, device=dvc)
        # save_data_at_frame(mpm_solver, directory_to_save, k, save_to_ply=True, save_to_h5=False)
    time2 = time.time()
    print("PID:", pid, ", Time for 500 iterations: ", time2-time1)


# extract the position, make some changes, load it back
# e.g. we shift the x position
# position = mpm_solver.export_particle_x_to_torch()
# position[..., 0] = position[..., 0] + 0.1
# mpm_solver.import_particle_x_from_torch(position)

# keep running sim
# for k in range(50, 1000):
#  
#     mpm_solver.p2g2p(k, 0.002, device=dvc)
#     save_data_at_frame(mpm_solver, directory_to_save, k, save_to_ply=True, save_to_h5=False)

# run_sim_process(0)

t1 = time.time()
bsz = 10
pool = mp.Pool(processes=bsz)
state_after_list = pool.map(run_sim_process, range(bsz))
pool.close()
pool.join()
t2 = time.time()
print("Total time for all processes: ", t2-t1)
