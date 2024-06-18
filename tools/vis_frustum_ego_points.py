import os
import numpy
import numpy as np
import open3d as o3d

def create_custom_axes():
    '''
    Create custom axes with x(red), y(green), z(blue) lines.
    '''
    # Coordinates
    points = [[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 10]]
    lines = [[0, 1], [0, 2], [0, 3]]

    # Colors for each axis: x = red, y = green, z = blue
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # Create line set
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

path = "/media/gpal/8e78e258-6a68-4733-8ec2-b837743b11e6/workspace/github/bevfusion/assets/frustum.ego.nuscenes.points.txt"
# path = "/media/gpal/8e78e258-6a68-4733-8ec2-b837743b11e6/workspace/github/bevfusion/assets/frustum.ego.6v.points.txt"

points = np.loadtxt(path).reshape(-1, 3)
o3d_points = o3d.geometry.PointCloud()
o3d_points.points = o3d.utility.Vector3dVector(points)

# o3d.io.write_point_cloud(
#     "/media/gpal/8e78e258-6a68-4733-8ec2-b837743b11e6/workspace/github/bevfusion/assets/frustum.ego.6v.points.pcd",
#     o3d_points
# )

custom_axes = create_custom_axes()

vis = o3d.visualization.VisualizerWithKeyCallback() # 可以回调自定义的函数，比如用按键回调函数（颜色显示，法向量等等）
vis.create_window() # 创建窗口
# 设置背景颜色为天蓝色
vis.get_render_option().background_color = [0.53, 0.81, 0.92]
# 设置点的大小
vis.get_render_option().point_size = 4  # 调整这个值以改变显示的点的大小
vis.add_geometry(o3d_points)
vis.add_geometry(custom_axes)
vis.run()
vis.destroy_window()