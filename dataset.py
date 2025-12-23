"""
Grasp Dataset Loader for DexGraspNet validated grasps.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh
from scipy.spatial.transform import Rotation as R
import transforms3d

# Import hand model
import sys
sys.path.append('/home/arjun/backup/dexlite/DexGraspNet/grasp_generation')
from DexGraspNet.grasp_generation.utils.hand_model_lite import HandModelMJCFLite


class GraspDataset(Dataset):
    def __init__(self, grasp_root, mesh_root, num_points=1024, augment=False):
        """
        Args:
            grasp_root: Path to validated grasps (.npy files)
            mesh_root: Path to mesh files (should have coacd/decomposed.obj)
            num_points: Number of points to sample from mesh
            augment: Whether to apply data augmentation
        """
        self.grasp_root = grasp_root
        self.mesh_root = mesh_root
        self.num_points = num_points
        self.augment = augment
        
        # Load all grasp files
        self.samples = []
        grasp_files = glob.glob(os.path.join(grasp_root, "*.npy"))
        
        for grasp_file in grasp_files:
            obj_code = os.path.basename(grasp_file).replace(".npy", "")
            mesh_path = os.path.join(mesh_root, obj_code, "coacd", "decomposed.obj")
            
            if not os.path.exists(mesh_path):
                continue
                
            grasps = np.load(grasp_file, allow_pickle=True)
            
            for grasp in grasps:
                self.samples.append({
                    'obj_code': obj_code,
                    'mesh_path': mesh_path,
                    'grasp': grasp
                })
        
        print(f"Loaded {len(self.samples)} grasp samples from {len(grasp_files)} objects")
        
        # Cache for meshes to avoid reloading
        self.mesh_cache = {}
        
        # Initialize hand model for FK (needs to be in correct directory)
        original_cwd = os.getcwd()
        os.chdir('/home/arjun/backup/dexlite/DexGraspNet/grasp_generation')
        self.hand_model = HandModelMJCFLite("mjcf/shadow_hand_vis.xml", "mjcf/meshes")
        os.chdir(original_cwd)
        
        # Joint names for conversion
        self.joint_names = [
            'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
            'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
            'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
            'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
            'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
        ]
        self.translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
        self.rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
    
    def __len__(self):
        return len(self.samples)
    
    def load_mesh(self, mesh_path):
        """Load and cache mesh."""
        if mesh_path not in self.mesh_cache:
            self.mesh_cache[mesh_path] = trimesh.load(mesh_path, force='mesh')
        return self.mesh_cache[mesh_path]
    
    def sample_point_cloud(self, mesh, scale, num_points):
        """Sample point cloud from mesh surface."""
        # Sample points on the surface
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        points = points * scale  # Apply scale
        return points.astype(np.float32)
    
    def extract_hand_params(self, grasp):
        """
        Extract 28D hand parameter vector from grasp.
        Order: [tx, ty, tz, rx, ry, rz, joint1, ..., joint22]
        """
        qpos = grasp['qpos']
        
        # Translation (3D)
        trans = np.array([
            qpos.get('WRJTx', 0.0),
            qpos.get('WRJTy', 0.0),
            qpos.get('WRJTz', 0.0)
        ], dtype=np.float32)
        
        # Rotation as euler angles (3D)
        rot = np.array([
            qpos.get('WRJRx', 0.0),
            qpos.get('WRJRy', 0.0),
            qpos.get('WRJRz', 0.0)
        ], dtype=np.float32)
        
        # Joint angles (22D) - in order
        joint_names = [
            'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
            'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
            'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
            'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
            'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
        ]
        joints = np.array([qpos.get(name, 0.0) for name in joint_names], dtype=np.float32)
        
        # Concatenate: [3 + 3 + 22] = 28D
        hand_params = np.concatenate([trans, rot, joints])
        return hand_params
    
    def compute_heading_vector(self, grasp):
        """
        Compute heading vector from palm center to midpoint between thumb tip and middle finger tip.
        Uses actual hand model forward kinematics.
        
        Args:
            grasp: Grasp dictionary with 'qpos' field
            
        Returns:
            heading_vector: (3,) normalized direction vector
            palm_center: (3,) palm center position
        """
        qpos = grasp['qpos']
        
        # Convert to hand model format
        rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in self.rot_names]))
        rot_6d = rot[:, :2].T.ravel().tolist()
        hand_pose = torch.tensor(
            [qpos[name] for name in self.translation_names] + rot_6d + [qpos[name] for name in self.joint_names],
            dtype=torch.float, device="cpu"
        ).unsqueeze(0)
        
        # Set hand model to grasp pose
        self.hand_model.set_parameters(hand_pose)
        
        # Get finger tips using FK from current_status
        # Thumb tip link is 'robot0:thdistal', middle finger is 'robot0:mfdistal'
        thumb_transform = self.hand_model.current_status['robot0:thdistal'].get_matrix()[0]
        middle_transform = self.hand_model.current_status['robot0:mfdistal'].get_matrix()[0]
        palm_transform = self.hand_model.current_status['robot0:palm'].get_matrix()[0]
        
        # Extract positions from transform matrices
        thumb_tip_local = thumb_transform[:3, 3]
        middle_tip_local = middle_transform[:3, 3]
        
        # Palm center is at the inertial center: offset [0.006, 0, 0.036] in palm frame
        palm_center_offset = torch.tensor([0.006, 0.0, 0.036], dtype=torch.float32, device=palm_transform.device)
        palm_center_local = palm_transform[:3, :3] @ palm_center_offset + palm_transform[:3, 3]
        
        # Transform to world coordinates
        global_rot = self.hand_model.global_rotation[0].cpu().numpy()
        global_trans = self.hand_model.global_translation[0].cpu().numpy()
        
        thumb_tip = (global_rot @ thumb_tip_local.cpu().numpy() + global_trans).astype(np.float32)
        middle_tip = (global_rot @ middle_tip_local.cpu().numpy() + global_trans).astype(np.float32)
        palm_center = (global_rot @ palm_center_local.cpu().numpy() + global_trans).astype(np.float32)
        
        # Midpoint between thumb and middle finger
        midpoint = (thumb_tip + middle_tip) / 2.0
        
        # Heading vector from palm to midpoint
        heading = midpoint - palm_center
        heading = heading / (np.linalg.norm(heading) + 1e-8)  # Normalize
        
        return heading, palm_center
    
    def find_associated_point(self, point_cloud, heading, palm_center):
        """
        Find the closest point on the object along the heading direction.
        
        Args:
            point_cloud: (N, 3) point cloud
            heading: (3,) normalized heading vector
            palm_center: (3,) palm center position
        
        Returns:
            associated_point_idx: index of the associated point
        """
        # Project all points onto the heading vector
        vectors_to_points = point_cloud - palm_center  # (N, 3)
        projections = np.dot(vectors_to_points, heading)  # (N,)
        
        # Only consider points in the forward direction
        valid_mask = projections > 0
        
        if not valid_mask.any():
            # Fallback: closest point overall
            distances = np.linalg.norm(point_cloud - palm_center, axis=1)
            return np.argmin(distances)
        
        # Among valid points, find closest to the heading ray
        valid_indices = np.where(valid_mask)[0]
        valid_points = point_cloud[valid_indices]
        
        # Distance to ray = ||(p - palm_center) - proj * heading||
        valid_vectors = valid_points - palm_center
        valid_projs = projections[valid_indices]
        projected_points = palm_center + valid_projs[:, None] * heading
        distances_to_ray = np.linalg.norm(valid_points - projected_points, axis=1)
        
        # Find closest to ray
        closest_idx = valid_indices[np.argmin(distances_to_ray)]
        
        return closest_idx
    
    def __getitem__(self, idx):
        """
        Returns:
            point_cloud: (num_points, 3) tensor
            hand_params: (28,) tensor
            associated_point_idx: scalar tensor
            scale: scalar tensor
        """
        sample = self.samples[idx]
        grasp = sample['grasp']
        
        # Load mesh and sample point cloud
        mesh = self.load_mesh(sample['mesh_path'])
        scale = grasp['scale']
        point_cloud = self.sample_point_cloud(mesh, scale, self.num_points)
        
        # Extract hand parameters
        hand_params = self.extract_hand_params(grasp)
        
        # Compute heading vector and find associated point (using FK with actual grasp pose)
        heading, palm_center = self.compute_heading_vector(grasp)
        associated_point_idx = self.find_associated_point(point_cloud, heading, palm_center)
        
        # Convert to tensors
        point_cloud = torch.from_numpy(point_cloud).float()
        hand_params = torch.from_numpy(hand_params).float()
        associated_point_idx = torch.tensor(associated_point_idx, dtype=torch.long)
        scale = torch.tensor(scale, dtype=torch.float32)
        
        return {
            'point_cloud': point_cloud,
            'hand_params': hand_params,
            'associated_point_idx': associated_point_idx,
            'scale': scale,
            'obj_code': sample['obj_code']
        }
