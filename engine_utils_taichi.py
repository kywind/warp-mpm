import numpy as np
import h5py
import os
import sys
import taichi as ti
import torch
import copy

def save_data_at_frame(mpm_solver, dir_name, frame, save_to_ply = True, save_to_h5 = False):
    os.umask(0)
    os.makedirs(dir_name, 0o777, exist_ok=True)
    
    fullfilename = dir_name + '/sim_' + str(frame).zfill(10) + '.h5'

    if save_to_ply:
        particle_position_to_ply(mpm_solver, fullfilename[:-2]+'ply')
    
    if save_to_h5:

        if os.path.exists(fullfilename): os.remove(fullfilename)
        newFile = h5py.File(fullfilename, "w")

        x_np = mpm_solver.particle_x.to_numpy().copy()[0].transpose() # x_np has shape (3, n_particles)
        newFile.create_dataset("x", data=x_np) # position

        currentTime = np.array([mpm_solver.time]).reshape(1,1)
        newFile.create_dataset("time", data=currentTime) # current time

        f_tensor_np = mpm_solver.particle_F.to_numpy().copy()[0].reshape(-1,9).transpose() # shape = (9, n_particles)
        newFile.create_dataset("f_tensor", data=f_tensor_np) # deformation grad

        v_np = mpm_solver.particle_v.to_numpy().copy()[0].transpose() # v_np has shape (3, n_particles)
        newFile.create_dataset("v", data=v_np) # particle velocity

        C_np = mpm_solver.particle_C.to_numpy().copy()[0].reshape(-1,9).transpose() # shape = (9, n_particles)
        newFile.create_dataset("C", data=C_np) # particle C
        print("save siumlation data at frame ", frame, " to ", fullfilename)

def particle_position_to_ply(mpm_solver, filename):
    # position is (n,3)
    if os.path.exists(filename):
        os.remove(filename)
    position = mpm_solver.particle_x.to_numpy().copy()
    num_particles = position.shape[0] * position.shape[1]
    bsz = position.shape[0]
    offset = np.arange(bsz)[:, None, None] * 1.0
    offset = np.concatenate([offset, np.zeros_like(offset), np.zeros_like(offset)], axis=-1)
    position = position + offset
    position = position.reshape(-1, 3)
    position = position.astype(np.float32)
    with open(filename, 'wb') as f: # write binary
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_particles}
property float x
property float y
property float z
end_header
"""
        f.write(str.encode(header))
        f.write(position.tobytes())
        print("write", filename)

def particle_position_tensor_to_ply(position_tensor, filename):
    # position is (n,3)
    if os.path.exists(filename):
        os.remove(filename)
    position = position_tensor.clone().detach().cpu().numpy()
    num_particles = (position).shape[0]
    position = position.astype(np.float32)
    with open(filename, 'wb') as f: # write binary
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_particles}
property float x
property float y
property float z
end_header
"""
        f.write(str.encode(header))
        f.write(position.tobytes())
        print("write", filename)
