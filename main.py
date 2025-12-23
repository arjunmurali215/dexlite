"""
Main entry point for training DexGrasp CVAE model.
"""

import argparse
from train import GraspTrainer


def main():
    parser = argparse.ArgumentParser(description='Train DexGrasp CVAE')
    parser.add_argument('--grasp_root', type=str, 
                        default='/home/arjun/datasets/dexgraspnet/validated_grasps',
                        help='Path to validated grasp .npy files')
    parser.add_argument('--mesh_root', type=str,
                        default='/home/arjun/datasets/dexgraspnet/filtered_meshes',
                        help='Path to mesh files')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Number of points to sample from mesh')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of dataloader workers')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./runs',
                        help='Directory for TensorBoard logs')
    
    # Loss weights
    parser.add_argument('--w_kl', type=float, default=1e-4,
                        help='Weight for KL divergence loss')
    parser.add_argument('--w_recon', type=float, default=1.0,
                        help='Weight for reconstruction loss')
    parser.add_argument('--w_dis', type=float, default=1e-4,
                        help='Weight for contact distance energy')
    parser.add_argument('--w_pen', type=float, default=1e-4,
                        help='Weight for penetration energy')
    parser.add_argument('--w_spen', type=float, default=1e-5,
                        help='Weight for self-penetration energy')
    parser.add_argument('--w_joints', type=float, default=1e-6,
                        help='Weight for joint limit energy')
    parser.add_argument('--w_fc', type=float, default=1.0,
                        help='Weight for force closure energy')
    
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create config dict
    config = {
        'grasp_root': args.grasp_root,
        'mesh_root': args.mesh_root,
        'num_points': args.num_points,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'lr': args.lr,
        'save_every': args.save_every,
        'w_kl': args.w_kl,
        'w_recon': args.w_recon,
        'w_dis': args.w_dis,
        'w_pen': args.w_pen,
        'w_spen': args.w_spen,
        'w_joints': args.w_joints,
        'w_fc': args.w_fc
    }
    
    # Print configuration
    print("=" * 70)
    print("DexGrasp CVAE Training")
    print("=" * 70)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 70)
    
    # Initialize trainer
    trainer = GraspTrainer(config)
    
    # Setup dataloaders
    trainer.setup_dataloaders(train_split=0.9)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resuming from epoch {start_epoch}")
    
    # Train
    trainer.train(num_epochs=args.num_epochs, save_dir=args.save_dir, log_dir=args.log_dir)


if __name__ == '__main__':
    main()
