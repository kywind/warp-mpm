import open3d as o3d
from time import sleep

frames = 500

vis = o3d.visualization.Visualizer()
vis.create_window()

vis_dir = 'sim_results_mls'

pcd = o3d.io.read_point_cloud(f'{vis_dir}/sim_0000000000.ply')
vis.add_geometry(pcd)
vis.poll_events()
vis.update_renderer()
for i in range(1, frames):
    pcd.points = o3d.io.read_point_cloud(f'{vis_dir}/sim_{i:010d}.ply').points
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    sleep(0.02)
