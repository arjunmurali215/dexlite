import os
import random
from utils.hand_model_lite import HandModelMJCFLite
import numpy as np
import transforms3d
import torch
import trimesh


mesh_path = "/home/arjun/datasets/dexgraspnet/filtered_meshes"
data_path = "/home/arjun/backup/try1/post_optimized_grasps"


use_visual_mesh = True

hand_file = "mjcf/shadow_hand_vis.xml" if use_visual_mesh else "mjcf/shadow_hand_wrist_free.xml"
# "mjcf/shadow_hand_wrist_free.xml"

joint_names = [
    'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
    'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
    'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
    'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
    'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
]
translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']

hand_model = HandModelMJCFLite(
    hand_file,
    "mjcf/meshes")


grasp_code_list = []
for code in os.listdir(data_path):
    grasp_code_list.append(code[:-4])

grasp_code = random.choice(grasp_code_list)
# grasp_code = "core-bottle-1071fa4cddb2da2fc8724d5673a063a6"
grasp_data = np.load(
    os.path.join(data_path, grasp_code+".npy"), allow_pickle=True)
object_mesh_origin = trimesh.load(os.path.join(
    mesh_path, grasp_code, "coacd/decomposed.obj"))
print(grasp_code)

index = random.randint(0, len(grasp_data) - 1)
print(index)


qpos = grasp_data[index]['qpos']
rot = np.array(transforms3d.euler.euler2mat(
    *[qpos[name] for name in rot_names]))
rot = rot[:, :2].T.ravel().tolist()
hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [qpos[name]
                         for name in joint_names], dtype=torch.float, device="cpu").unsqueeze(0)
hand_model.set_parameters(hand_pose)
hand_mesh = hand_model.get_trimesh_data(0)
object_mesh = object_mesh_origin.copy().apply_scale(grasp_data[index]["scale"])

# Create visualization with origin, axes, and table plane
scene = trimesh.Scene()

# Add hand and object meshes
scene.add_geometry(hand_mesh, geom_name='hand')
scene.add_geometry(object_mesh, geom_name='object')

# Add coordinate axes at origin (RGB = XYZ)
axis_length = 0.1
axes = trimesh.creation.axis(origin_size=0.01, axis_length=axis_length)
# scene.add_geometry(axes, geom_name='axes')

# Add table plane (at z = bottom of object)
table_z = object_mesh.bounds[0, 2]  # bottom of scaled object
table_size = 0.3
table_plane = trimesh.creation.box(extents=[table_size, table_size, 0.002])
table_plane.apply_translation([0, 0, table_z - 0.001])  # just below object bottom
table_plane.visual.face_colors = [150, 150, 150, 180]  # gray, semi-transparent
# scene.add_geometry(table_plane, geom_name='table')

# Visualize ALL mesh vertices (same as E_table calculation in energy.py)
all_verts = []
for link_name in hand_model.mesh:
    link_verts = hand_model.mesh[link_name]['vertices']  # (n_verts, 3) in link frame
    # Transform to world frame using current FK status
    transformed = hand_model.current_status[link_name].transform_points(link_verts)
    if len(transformed.shape) == 2:
        transformed = transformed.unsqueeze(0)
    # Apply global rotation and translation
    transformed = transformed @ hand_model.global_rotation.transpose(1, 2) + hand_model.global_translation.unsqueeze(1)
    all_verts.append(transformed[0].detach().cpu().numpy())

all_pts = np.vstack(all_verts)

# Color points: red if below table, green if above
colors = np.zeros((len(all_pts), 4), dtype=np.uint8)
below_mask = all_pts[:, 2] < table_z
colors[below_mask] = [255, 0, 0, 255]      # red = below table (BAD)
colors[~below_mask] = [0, 255, 0, 255]     # green = above table (OK)

# Create point cloud
pts_cloud = trimesh.PointCloud(all_pts, colors=colors)
# scene.add_geometry(pts_cloud, geom_name='E_table_points')

# Print stats
n_below = below_mask.sum()
print(f"Hand mesh vertices: {len(all_pts)} total, {n_below} below table (red), {len(all_pts)-n_below} above (green)")
print(f"Table Z: {table_z:.4f}")
print(f"Min hand vertex Z: {all_pts[:, 2].min():.4f}")

# Create rotation callback for continuous slow rotation
def rotation_callback(scene_obj):
    """Continuously rotate the scene around the Z axis"""
    # Rotate all geometries by a small angle around Z axis
    angle_rad = np.radians(0.5)  # 0.5 degrees per frame for slow rotation
    rotation_matrix = trimesh.transformations.rotation_matrix(angle_rad, [0, 0, 1])
    
    for node_name in scene_obj.graph.nodes_geometry:
        current_transform = scene_obj.graph[node_name][0]
        new_transform = np.dot(rotation_matrix, current_transform)
        scene_obj.graph[node_name] = new_transform

# Show the scene with continuous rotation
scene.show(callback=rotation_callback, callback_period=0.03)
