"""
Test script to visualize associated point computation.
Loads a grasp, displays the hand + object, heading vector, and associated point.
"""

import os
import sys
import numpy as np
import trimesh
import torch
from scipy.spatial.transform import Rotation as R

# Add paths
sys.path.insert(0, '/home/arjun/backup/dexlite/DexGraspNet/grasp_generation')
from DexGraspNet.grasp_generation.utils.hand_model_lite import HandModelMJCFLite
import transforms3d

# Import dataset functions
from dataset import GraspDataset


def visualize_associated_point(obj_idx=0, grasp_idx=0):
    """
    Load a grasp and visualize the associated point computation.
    
    Args:
        obj_idx: Index of which object to load
        grasp_idx: Index of grasp within the object's grasp file
    """
    # Paths
    grasp_root = "/home/arjun/datasets/dexgraspnet/validated_grasps"
    mesh_root = "/home/arjun/datasets/dexgraspnet/filtered_meshes"
    
    # Get list of object codes
    import glob
    grasp_files = sorted(glob.glob(os.path.join(grasp_root, "*.npy")))
    
    if obj_idx >= len(grasp_files):
        print(f"Object index {obj_idx} out of range. Max: {len(grasp_files) - 1}")
        return
    
    grasp_file = grasp_files[obj_idx]
    obj_code = os.path.basename(grasp_file).replace(".npy", "")
    
    # Load all grasps for this object
    grasps = np.load(grasp_file, allow_pickle=True)
    
    if grasp_idx >= len(grasps):
        print(f"Grasp index {grasp_idx} out of range. Max: {len(grasps) - 1}")
        return
    
    grasp = grasps[grasp_idx]
    
    print(f"Object {obj_idx}/{len(grasp_files)}: {obj_code}")
    print(f"Grasp index: {grasp_idx} / {len(grasps)}")
    print(f"Scale: {grasp['scale']:.4f}")
    
    # Load mesh
    mesh_path = os.path.join(mesh_root, obj_code, "coacd", "decomposed.obj")
    if not os.path.exists(mesh_path):
        print(f"Mesh not found: {mesh_path}")
        return
    
    mesh = trimesh.load(mesh_path, force='mesh')
    object_mesh = mesh.copy().apply_scale(grasp['scale'])
    
    # Initialize dataset helper for the extract functions
    from dataset import GraspDataset
    dataset_helper = GraspDataset(grasp_root, mesh_root, num_points=1024)
    
    # Sample point cloud
    point_cloud_np, _ = trimesh.sample.sample_surface(mesh, 1024)
    point_cloud_np = point_cloud_np * grasp['scale']
    
    # =============== Setup hand model FIRST ===============
    joint_names = [
        'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
        'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
        'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
        'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
        'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
    ]
    translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
    rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
    
    # Change to grasp_generation directory to load hand model (uses relative paths)
    original_cwd = os.getcwd()
    os.chdir('/home/arjun/backup/dexlite/DexGraspNet/grasp_generation')
    
    hand_model = HandModelMJCFLite(
        "mjcf/shadow_hand_vis.xml",
        "mjcf/meshes"
    )
    
    # Change back to original directory
    os.chdir(original_cwd)
    
    # Convert hand params to hand model format and SET THE POSE
    qpos = grasp['qpos']
    rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in rot_names]))
    rot_6d = rot[:, :2].T.ravel().tolist()
    hand_pose = torch.tensor(
        [qpos[name] for name in translation_names] + rot_6d + [qpos[name] for name in joint_names],
        dtype=torch.float, device="cpu"
    ).unsqueeze(0)
    
    # SET HAND MODEL TO GRASP POSE BEFORE computing heading/tips
    hand_model.set_parameters(hand_pose)
    hand_mesh = hand_model.get_trimesh_data(0)
    
    # =============== NOW compute finger tips AFTER hand is in grasp pose ===============
    # Get actual finger tip positions from forward kinematics
    thumb_transform = hand_model.current_status['robot0:thdistal'].get_matrix()[0]
    middle_transform = hand_model.current_status['robot0:mfdistal'].get_matrix()[0]
    palm_transform = hand_model.current_status['robot0:palm'].get_matrix()[0]
    
    # Extract positions from transform matrices
    thumb_tip_local = thumb_transform[:3, 3]
    middle_tip_local = middle_transform[:3, 3]
    
    # Palm center is at the inertial center: offset [0.006, 0, 0.036] in palm frame
    palm_center_offset = torch.tensor([0.006, 0.0, 0.036], dtype=torch.float32, device=palm_transform.device)
    palm_center_local = palm_transform[:3, :3] @ palm_center_offset + palm_transform[:3, 3]
    
    # Transform to world coordinates
    global_rot = hand_model.global_rotation[0].cpu().numpy()
    global_trans = hand_model.global_translation[0].cpu().numpy()
    
    th_tip = (global_rot @ thumb_tip_local.cpu().numpy() + global_trans).astype(np.float32)
    mf_tip = (global_rot @ middle_tip_local.cpu().numpy() + global_trans).astype(np.float32)
    palm_center = (global_rot @ palm_center_local.cpu().numpy() + global_trans).astype(np.float32)
    
    # Midpoint between thumb and middle finger
    midpoint = (mf_tip + th_tip) / 2.0
    
    # Heading vector from palm to midpoint
    heading = midpoint - palm_center
    
    # Find associated point using the CORRECT heading vector
    associated_idx = dataset_helper.find_associated_point(point_cloud_np, heading, palm_center)
    associated_point = point_cloud_np[associated_idx]
    
    print(f"\nPalm center: {palm_center}")
    print(f"Thumb tip: {th_tip}")
    print(f"Middle tip: {mf_tip}")
    print(f"Midpoint: {midpoint}")
    print(f"Heading vector: {heading}")
    print(f"Associated point index: {associated_idx}")
    print(f"Associated point: {associated_point}")
    
    # =============== Create visualization scene ===============
    scene = trimesh.Scene()
    
    # Add hand and object
    scene.add_geometry(hand_mesh, geom_name='hand')
    scene.add_geometry(object_mesh, geom_name='object')
    
    # Add coordinate axes at origin
    axes = trimesh.creation.axis(origin_size=0.01, axis_length=0.1)
    scene.add_geometry(axes, geom_name='axes')
    
    # Add point cloud (small spheres)
    point_colors = np.ones((len(point_cloud_np), 4)) * [200, 200, 200, 100]  # Gray
    point_cloud_mesh = trimesh.PointCloud(point_cloud_np, colors=point_colors)
    scene.add_geometry(point_cloud_mesh, geom_name='point_cloud')
    
    # Add palm center sphere (blue)
    palm_sphere = trimesh.creation.icosphere(radius=0.008)
    palm_sphere.visual.face_colors = [0, 0, 255, 255]  # Blue
    palm_sphere.apply_translation(palm_center)
    scene.add_geometry(palm_sphere, geom_name='palm_center')
    
    # Add thumb tip sphere (magenta)
    thumb_sphere = trimesh.creation.icosphere(radius=0.007)
    thumb_sphere.visual.face_colors = [255, 0, 255, 255]  # Magenta
    thumb_sphere.apply_translation(th_tip)
    scene.add_geometry(thumb_sphere, geom_name='thumb_tip')
    
    # Add middle finger tip sphere (yellow)
    middle_sphere = trimesh.creation.icosphere(radius=0.007)
    middle_sphere.visual.face_colors = [255, 255, 0, 255]  # Yellow
    middle_sphere.apply_translation(mf_tip)
    scene.add_geometry(middle_sphere, geom_name='middle_tip')
    
    # Add midpoint sphere (orange)
    midpoint_sphere = trimesh.creation.icosphere(radius=0.009)
    midpoint_sphere.visual.face_colors = [255, 165, 0, 255]  # Orange
    midpoint_sphere.apply_translation(midpoint)
    scene.add_geometry(midpoint_sphere, geom_name='midpoint')
    
    # Add line from thumb to middle finger (magenta-yellow gradient)
    thumb_to_middle = trimesh.creation.cylinder(
        radius=0.002,
        segment=[th_tip, mf_tip]
    )
    thumb_to_middle.visual.face_colors = [255, 128, 128, 255]  # Pink
    scene.add_geometry(thumb_to_middle, geom_name='thumb_to_middle')
    
    # Add associated point sphere (red)
    assoc_sphere = trimesh.creation.icosphere(radius=0.010)
    assoc_sphere.visual.face_colors = [255, 0, 0, 255]  # Red
    assoc_sphere.apply_translation(associated_point)
    scene.add_geometry(assoc_sphere, geom_name='associated_point')
    
    # Add heading vector as arrow (green)
    # Scale heading to make it visible
    heading_length = 0.15
    heading_end = palm_center + heading * heading_length
    
    # Create cylinder for arrow shaft
    arrow_shaft = trimesh.creation.cylinder(
        radius=0.002,
        segment=[palm_center, heading_end]
    )
    arrow_shaft.visual.face_colors = [0, 255, 0, 255]  # Green
    scene.add_geometry(arrow_shaft, geom_name='heading_arrow')
    
    # Create cone for arrow head
    arrow_head = trimesh.creation.cone(radius=0.005, height=0.015)
    # Orient cone along heading direction
    z_axis = np.array([0, 0, 1])
    if not np.allclose(heading, z_axis):
        rotation_axis = np.cross(z_axis, heading)
        rotation_angle = np.arccos(np.dot(z_axis, heading))
        rotation_matrix = trimesh.transformations.rotation_matrix(
            rotation_angle, rotation_axis
        )
        arrow_head.apply_transform(rotation_matrix)
    arrow_head.apply_translation(heading_end)
    arrow_head.visual.face_colors = [0, 255, 0, 255]  # Green
    scene.add_geometry(arrow_head, geom_name='heading_arrow_head')
    
    # Add line from palm to associated point (cyan, dashed-like by using many small segments)
    line_points = np.linspace(palm_center, associated_point, 20)
    for i in range(len(line_points) - 1):
        if i % 2 == 0:  # Create dashed effect
            seg = trimesh.creation.cylinder(
                radius=0.001,
                segment=[line_points[i], line_points[i+1]]
            )
            seg.visual.face_colors = [0, 255, 255, 200]  # Cyan
            scene.add_geometry(seg, geom_name=f'palm_to_assoc_{i}')
    
    # Print legend
    print("\n" + "=" * 60)
    print("VISUALIZATION LEGEND:")
    print("=" * 60)
    print("  Blue sphere     : Palm center")
    print("  Magenta sphere  : Thumb tip")
    print("  Yellow sphere   : Middle finger tip")
    print("  Orange sphere   : Midpoint (thumb-middle)")
    print("  Pink line       : Thumb tip → Middle finger tip")
    print("  Red sphere      : Associated point (closest along heading)")
    print("  Green arrow     : Heading vector (palm → midpoint)")
    print("  Cyan dashed line: Connection from palm to associated point")
    print("  Gray points     : Sampled point cloud from object")
    print("=" * 60)
    
    # Show scene
    scene.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize associated point computation')
    parser.add_argument('--obj_idx', type=int, default=0,
                        help='Index of object to load (0 to 2938)')
    parser.add_argument('--grasp_idx', type=int, default=0,
                        help='Index of grasp within object')
    
    args = parser.parse_args()
    
    visualize_associated_point(obj_idx=args.obj_idx, grasp_idx=args.grasp_idx)
