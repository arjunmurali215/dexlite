"""
Synthesis script to generate novel grasps using trained CVAE model.
Instead of reconstructing from encoder latent space, samples z ~ N(0,I) for synthesis.
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
import trimesh

# Add DexGraspNet to path
sys.path.insert(0, '/home/arjun/backup/dexlite/DexGraspNet/grasp_generation')

from dataset import GraspDataset
from networks.pointnet import DexPointNet
from networks.cvae import DexSimpleCVAE


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize models
    pointnet = DexPointNet(output_dim=256).to(device)
    cvae = DexSimpleCVAE(n_frames=1, latent_dim=256).to(device)
    
    # Load weights
    pointnet.load_state_dict(checkpoint['pointnet_state'])
    cvae.load_state_dict(checkpoint['cvae_state'])
    
    # Set to eval mode
    pointnet.eval()
    cvae.eval()
    
    print("Model loaded successfully")
    return pointnet, cvae, checkpoint.get('epoch', 0)


def get_default_qpos():
    """Get default qpos dictionary with zero values."""
    joint_names = [
        'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
        'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
        'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
        'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
        'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
    ]
    translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
    rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
    
    qpos = {}
    for name in translation_names + rot_names + joint_names:
        qpos[name] = 0.0
    return qpos


def reconstruct_qpos(hand_params, template_qpos=None):
    """
    Convert hand parameters back to qpos dictionary format.
    
    Args:
        hand_params: (28,) numpy array [tx, ty, tz, rx, ry, rz, joint1...joint22]
        template_qpos: Template qpos dict (optional, uses default if None)
    
    Returns:
        qpos dictionary
    """
    joint_names = [
        'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
        'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
        'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
        'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
        'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
    ]
    translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
    rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
    
    # Start with template or default qpos
    if template_qpos is not None:
        qpos = template_qpos.copy()
    else:
        qpos = get_default_qpos()
    
    # Update with synthesized values
    for i, name in enumerate(translation_names):
        qpos[name] = float(hand_params[i])
    
    for i, name in enumerate(rot_names):
        qpos[name] = float(hand_params[3 + i])
    
    for i, name in enumerate(joint_names):
        qpos[name] = float(hand_params[6 + i])
    
    return qpos


def load_mesh_and_sample_points(mesh_path, num_points=1024):
    """Load mesh and sample point cloud."""
    mesh = trimesh.load(mesh_path, force='mesh')
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points.astype(np.float32)


def synthesize_grasps(args):
    """Main synthesis function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    pointnet, cvae, epoch = load_model(args.checkpoint, device)
    print(f"Loaded model from epoch {epoch}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Load dataset for point cloud and associated point information
    print(f"\nLoading dataset from {args.grasp_root}")
    dataset = GraspDataset(
        grasp_root=args.grasp_root,
        mesh_root=args.mesh_root,
        num_points=args.num_points
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Group samples by object code
    print("\nGrouping samples by object...")
    object_to_samples = {}
    for idx, sample in enumerate(dataset.samples):
        obj_code = sample['obj_code']
        if obj_code not in object_to_samples:
            object_to_samples[obj_code] = []
        object_to_samples[obj_code].append(idx)
    
    print(f"Found {len(object_to_samples)} unique objects")
    
    # Storage for synthesized grasps (organized by object)
    synthesized_grasps = {obj_code: [] for obj_code in object_to_samples.keys()}
    
    print("\nSynthesizing grasps...")
    print(f"Generating {args.num_samples} samples per input")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            # Move to device
            point_cloud = batch['point_cloud'].to(device)
            associated_idx = batch['associated_point_idx'].to(device)
            obj_codes = batch['obj_code']
            scales = batch['scale'].cpu().numpy()
            
            batch_size = point_cloud.size(0)
            
            # Get PointNet features (same as infer.py)
            point_cloud_t = point_cloud.transpose(1, 2)
            global_feat, local_feat = pointnet(point_cloud_t, associated_idx)
            
            # Generate multiple samples per input if requested
            for sample_iter in range(args.num_samples):
                # SYNTHESIS MODE: Sample z from N(0, I) instead of using encoder
                # This is the key difference from infer.py
                synth_hand = cvae.inference(global_feat, local_feat)
                
                # Convert back to numpy
                synth_hand_np = synth_hand.squeeze(1).cpu().numpy()  # (B, 28)
                
                for i in range(len(obj_codes)):
                    obj_code = obj_codes[i]
                    hand_params = synth_hand_np[i]
                    scale = scales[i]
                    
                    # Use default qpos template (no ground truth needed for synthesis)
                    qpos = reconstruct_qpos(hand_params)
                    
                    # Create grasp dictionary
                    synthesized_grasp = {
                        'qpos': qpos,
                        'scale': float(scale)
                    }
                    
                    synthesized_grasps[obj_code].append(synthesized_grasp)
    
    # Save synthesized grasps (one .npy file per object)
    print("\nSaving synthesized grasps...")
    for obj_code, grasps in tqdm(synthesized_grasps.items(), desc="Saving files"):
        if len(grasps) > 0:
            output_path = os.path.join(args.output_dir, f"{obj_code}.npy")
            np.save(output_path, np.array(grasps, dtype=object), allow_pickle=True)
    
    print(f"\nSynthesis complete!")
    print(f"Synthesized {sum(len(g) for g in synthesized_grasps.values())} grasps")
    print(f"Saved to {args.output_dir}")
    
    # Print statistics
    print("\nStatistics:")
    print(f"  Total objects: {len(synthesized_grasps)}")
    grasps_per_obj = [len(g) for g in synthesized_grasps.values()]
    if grasps_per_obj:
        print(f"  Grasps per object (avg): {np.mean(grasps_per_obj):.1f}")
        print(f"  Grasps per object (min/max): {min(grasps_per_obj)}/{max(grasps_per_obj)}")


def main():
    parser = argparse.ArgumentParser(description='Synthesize novel grasps using trained CVAE model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--grasp_root', type=str,
                        default='/home/arjun/datasets/dexgraspnet/validated_grasps',
                        help='Path to input validated grasps (for point clouds and associated points)')
    parser.add_argument('--mesh_root', type=str,
                        default='/home/arjun/datasets/dexgraspnet/filtered_meshes',
                        help='Path to mesh files')
    parser.add_argument('--output_dir', type=str,
                        default='./synthesized_grasps',
                        help='Output directory for synthesized grasps')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Number of points to sample from mesh')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of grasp samples to generate per input')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 70)
    print("DexGrasp CVAE Synthesis (Novel Grasp Generation)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Input grasps: {args.grasp_root}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Samples per input: {args.num_samples}")
    print("=" * 70)
    
    synthesize_grasps(args)


if __name__ == '__main__':
    main()
