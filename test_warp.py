import warp as wp
import numpy as np

wp.init()

# num_points = 1024

# @wp.kernel
# def length(points: wp.array(dtype=wp.vec3),
#            lengths: wp.array(dtype=float)):

#     # thread index
#     tid = wp.tid()

#     # compute distance of each point from origin
#     lengths[tid] = wp.length(points[tid])


# # allocate an array of 3d points
# points = wp.array(np.random.rand(num_points, 3), dtype=wp.vec3)
# lengths = wp.zeros(num_points, dtype=float)

# # launch kernel
# wp.launch(kernel=length,
#           dim=len(points),
#           inputs=[points, lengths])

# print(lengths)

@wp.kernel
def set_vec3_to_zero(target_array: wp.array(dtype=wp.vec3)):  # type: ignore
    tid = wp.tid()
    target_array[tid] = wp.vec3(0.0, 0.0, 0.0)


points = wp.array(np.random.rand(1, 1024, 3), dtype=wp.vec3)
wp.launch(kernel=set_vec3_to_zero,
          dim=(1),
          inputs=[points])