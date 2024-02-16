import open3d as o3d
from time import sleep

frames = 1000

vis = o3d.visualization.Visualizer()
vis.create_window()

pcd = o3d.io.read_point_cloud(f'sim_results/sim_0000000000.ply')
vis.add_geometry(pcd)
vis.poll_events()
vis.update_renderer()
for i in range(1, frames):
    pcd.points = o3d.io.read_point_cloud(f'sim_results/sim_{i:010d}.ply').points
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    sleep(0.02)