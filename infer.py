"""
Inference script to generate grasps using trained CVAE model.
Processes all validated grasps through the model and saves outputs.
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob

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


def reconstruct_qpos(hand_params, original_qpos):
    """
    Convert hand parameters back to qpos dictionary format.
    
    Args:
        hand_params: (28,) numpy array [tx, ty, tz, rx, ry, rz, joint1...joint22]
        original_qpos: Original qpos dict for reference
    
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
    
    # Start with original qpos to preserve any other keys
    qpos = original_qpos.copy()
    
    # Update with reconstructed values
    for i, name in enumerate(translation_names):
        qpos[name] = float(hand_params[i])
    
    for i, name in enumerate(rot_names):
        qpos[name] = float(hand_params[3 + i])
    
    for i, name in enumerate(joint_names):
        qpos[name] = float(hand_params[6 + i])
    
    return qpos


def generate_grasps(args):
    """Main inference function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    pointnet, cvae, epoch = load_model(args.checkpoint, device)
    print(f"Loaded model from epoch {epoch}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Load dataset
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
    
    # Storage for generated grasps (organized by object)
    generated_grasps = {obj_code: [] for obj_code in object_to_samples.keys()}
    
    print("\nGenerating grasps...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            # Move to device
            point_cloud = batch['point_cloud'].to(device)
            hand_params_gt = batch['hand_params'].unsqueeze(1).to(device)
            associated_idx = batch['associated_point_idx'].to(device)
            obj_codes = batch['obj_code']
            scales = batch['scale'].cpu().numpy()
            
            # Forward pass
            point_cloud_t = point_cloud.transpose(1, 2)
            global_feat, local_feat = pointnet(point_cloud_t, associated_idx)
            
            if args.sample:
                # Sample from latent space
                recon_hand, mu, logvar = cvae(hand_params_gt, global_feat, local_feat)
                # Optionally: sample multiple times per input
                # z = cvae.reparameterize(mu, logvar)
                # recon_hand = cvae.decode(z, global_feat, local_feat)
            else:
                # Reconstruction mode (deterministic)
                recon_hand, mu, logvar = cvae(hand_params_gt, global_feat, local_feat)
            
            # Convert back to numpy and store
            recon_hand_np = recon_hand.squeeze(1).cpu().numpy()  # (B, 28)
            
            for i in range(len(obj_codes)):
                obj_code = obj_codes[i]
                hand_params = recon_hand_np[i]
                scale = scales[i]
                
                # Get original grasp for qpos template
                sample_idx = batch_idx * args.batch_size + i
                if sample_idx < len(dataset.samples):
                    original_grasp = dataset.samples[sample_idx]['grasp']
                    
                    # Reconstruct qpos dictionary
                    qpos = reconstruct_qpos(hand_params, original_grasp['qpos'])
                    
                    # Create grasp dictionary
                    generated_grasp = {
                        'qpos': qpos,
                        'scale': float(scale)
                    }
                    
                    generated_grasps[obj_code].append(generated_grasp)
    
    # Save generated grasps (one .npy file per object)
    print("\nSaving generated grasps...")
    for obj_code, grasps in tqdm(generated_grasps.items(), desc="Saving files"):
        if len(grasps) > 0:
            output_path = os.path.join(args.output_dir, f"{obj_code}.npy")
            np.save(output_path, np.array(grasps, dtype=object), allow_pickle=True)
    
    print(f"\nGeneration complete!")
    print(f"Generated {sum(len(g) for g in generated_grasps.values())} grasps")
    print(f"Saved to {args.output_dir}")
    
    # Print statistics
    print("\nStatistics:")
    print(f"  Total objects: {len(generated_grasps)}")
    print(f"  Grasps per object (avg): {np.mean([len(g) for g in generated_grasps.values()]):.1f}")
    print(f"  Grasps per object (min/max): {min(len(g) for g in generated_grasps.values())}/{max(len(g) for g in generated_grasps.values())}")


def main():
    parser = argparse.ArgumentParser(description='Generate grasps using trained CVAE model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--grasp_root', type=str,
                        default='/home/arjun/datasets/dexgraspnet/validated_grasps',
                        help='Path to input validated grasps')
    parser.add_argument('--mesh_root', type=str,
                        default='/home/arjun/datasets/dexgraspnet/filtered_meshes',
                        help='Path to mesh files')
    parser.add_argument('--output_dir', type=str,
                        default='./generated_grasps',
                        help='Output directory for generated grasps')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Number of points to sample from mesh')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--sample', action='store_true',
                        help='Sample from latent space (vs deterministic reconstruction)')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 70)
    print("DexGrasp CVAE Inference")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Input grasps: {args.grasp_root}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sample mode: {args.sample}")
    print("=" * 70)
    
    generate_grasps(args)


if __name__ == '__main__':
    main()
