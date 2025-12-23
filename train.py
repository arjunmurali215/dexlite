"""
Training script for DexGrasp CVAE model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from tqdm import tqdm
import numpy as np

# Add DexGraspNet to path for hand/object models
sys.path.insert(0, '/home/arjun/backup/dexlite/DexGraspNet/grasp_generation')

from dataset import GraspDataset
from networks.pointnet import DexPointNet
from networks.cvae import DexSimpleCVAE
from DexGraspNet.grasp_generation.utils.hand_model import HandModel
from DexGraspNet.grasp_generation.utils.object_model import ObjectModel


class GraspTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.pointnet = DexPointNet(output_dim=256).to(self.device)
        self.cvae = DexSimpleCVAE(n_frames=1, latent_dim=256).to(self.device)
        
        # Initialize hand model for energy computation (need to be in correct directory)
        original_cwd = os.getcwd()
        os.chdir('/home/arjun/backup/dexlite/DexGraspNet/grasp_generation')
        self.hand_model = HandModel(
            mjcf_path='mjcf/shadow_hand_wrist_free.xml',
            mesh_path='mjcf/meshes',
            contact_points_path='mjcf/contact_points.json',
            penetration_points_path='mjcf/penetration_points.json',
            n_surface_points=1024,
            device=self.device
        )
        os.chdir(original_cwd)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.pointnet.parameters()) + list(self.cvae.parameters()),
            lr=config['lr']
        )
        
        # Loss weights
        self.w_kl = config.get('w_kl', 0.001)
        self.w_recon = config.get('w_recon', 1.0)
        self.w_dis = config.get('w_dis', 100.0)
        self.w_pen = config.get('w_pen', 100.0)
        self.w_spen = config.get('w_spen', 10.0)
        self.w_joints = config.get('w_joints', 1.0)
        self.w_fc = config.get('w_fc', 1.0)
        
        # TensorBoard writer
        self.writer = None
        
    def setup_dataloaders(self, train_split=0.9):
        """Create train and validation dataloaders."""
        full_dataset = GraspDataset(
            grasp_root=self.config['grasp_root'],
            mesh_root=self.config['mesh_root'],
            num_points=self.config['num_points']
        )
        
        # Split dataset
        train_size = int(train_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    def compute_energy_terms(self, hand_params_batch, point_clouds, scales):
        """
        Compute energy terms for a batch of hand poses.
        
        Args:
            hand_params_batch: (B, 28) tensor of hand parameters [tx, ty, tz, rx, ry, rz, joints(22)]
            point_clouds: (B, N, 3) tensor of object point clouds
            scales: (B,) tensor of object scales
        
        Returns:
            Dictionary of energy terms
        """
        batch_size = hand_params_batch.shape[0]
        
        # Convert hand parameters from euler to 6D rotation
        # hand_params format: [tx, ty, tz, rx, ry, rz, joint1, ..., joint22] (28D)
        # hand_model expects: [tx, ty, tz, rot6d(6), joint1, ..., joint22] (31D)
        import transforms3d
        from DexGraspNet.grasp_generation.utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
        
        translation = hand_params_batch[:, :3]  # (B, 3)
        euler_angles = hand_params_batch[:, 3:6]  # (B, 3)
        joint_angles = hand_params_batch[:, 6:]  # (B, 22)
        
        # Convert euler to rotation matrix, then to 6D representation
        rot_matrices = []
        for i in range(batch_size):
            euler = euler_angles[i].detach().cpu().numpy()
            R = transforms3d.euler.euler2mat(euler[0], euler[1], euler[2])
            rot_matrices.append(torch.tensor(R, dtype=torch.float32, device=self.device))
        
        rot_matrices = torch.stack(rot_matrices)  # (B, 3, 3)
        rot_6d = rot_matrices[:, :, :2].transpose(1, 2).reshape(batch_size, 6)  # (B, 6)
        
        # Concatenate to get hand model format
        hand_params_model = torch.cat([translation, rot_6d, joint_angles], dim=1)  # (B, 31)
        
        # Set hand poses in hand model
        self.hand_model.set_parameters(hand_params_model)
        
        # Compute energy terms following energy.py implementation
        
        # E_joints: Joint limit violation (same as energy.py)
        joints = hand_params_model[:, 9:]  # (B, 22)
        E_joints = torch.sum((joints > self.hand_model.joints_upper) * (joints - self.hand_model.joints_upper), dim=-1) + \
                   torch.sum((joints < self.hand_model.joints_lower) * (self.hand_model.joints_lower - joints), dim=-1)
        
        # E_spen: Self-penetration (same as energy.py)
        E_spen = self.hand_model.self_penetration()  # (B,)
        
        # E_pen: Penetration of object surface points into hand (same as energy.py)
        # object_surface_points should be (B, num_samples, 3)
        # hand_model.cal_distance expects this and returns distances (B, num_samples)
        distances = self.hand_model.cal_distance(point_clouds)  # (B, N)
        distances = torch.clamp(distances, min=0)  # Only positive (penetration)
        E_pen = distances.sum(-1)  # (B,) - sum over all points
        
        # E_dis: Contact point distance to object
        # In energy.py: distance = object_model.cal_distance(hand_model.contact_points)
        # This gives signed distance from contact points to object surface
        # Since we don't have contact_points during training, we'll use hand surface points as proxy
        # and compute their signed distance to the object (approximated via point cloud)
        
        hand_surface = self.hand_model.get_surface_points()  # (B, n_surf, 3)
        
        # For each surface point, find distance to nearest object point
        # This approximates the signed distance (negative = outside, positive = inside)
        E_dis_list = []
        for i in range(batch_size):
            surf_pts = hand_surface[i]  # (n_surf, 3)
            obj_pts = point_clouds[i]  # (N, 3)
            
            # Distance to nearest object point
            dists = torch.cdist(surf_pts.unsqueeze(0), obj_pts.unsqueeze(0))[0]  # (n_surf, N)
            min_dists, _ = torch.min(dists, dim=1)  # (n_surf,)
            
            # Sum of absolute distances: torch.sum(distance.abs(), dim=-1)
            E_dis_list.append(torch.sum(min_dists.abs()))
        
        E_dis = torch.stack(E_dis_list)  # (B,)
        
        # E_fc: Force closure (placeholder - requires contact normals and grasp matrix)
        E_fc = torch.zeros(batch_size, device=self.device)
        
        energies = {
            'E_dis': E_dis,
            'E_pen': E_pen,
            'E_spen': E_spen,
            'E_joints': E_joints,
            'E_fc': E_fc
        }
        
        return energies
    
    def compute_loss(self, batch, recon_hand, mu, logvar):
        """
        Compute total loss including VAE loss and energy terms.
        
        Args:
            batch: Input batch dictionary
            recon_hand: (B, 1, 28) reconstructed hand parameters
            mu: (B, latent_dim) mean of latent distribution
            logvar: (B, latent_dim) log variance of latent distribution
        
        Returns:
            total_loss, loss_dict
        """
        gt_hand = batch['hand_params'].unsqueeze(1).to(self.device)  # (B, 1, 28)
        point_clouds = batch['point_cloud'].to(self.device)  # (B, N, 3)
        scales = batch['scale'].to(self.device)  # (B,)
        
        # 1. VAE Reconstruction Loss
        recon_loss = nn.functional.mse_loss(recon_hand, gt_hand)
        
        # 2. KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]
        
        # 3. Energy Terms (on reconstructed hand)
        recon_hand_flat = recon_hand.squeeze(1)  # (B, 28)
        energies = self.compute_energy_terms(recon_hand_flat, point_clouds, scales)
        
        E_dis = energies['E_dis'].mean()
        E_pen = energies['E_pen'].mean()
        E_spen = energies['E_spen'].mean()
        E_joints = energies['E_joints'].mean()
        E_fc = energies['E_fc'].mean()
        
        # Total Loss
        total_loss = (
            self.w_recon * recon_loss +
            self.w_kl * kl_loss +
            self.w_dis * E_dis +
            self.w_pen * E_pen +
            self.w_spen * E_spen +
            self.w_joints * E_joints +
            self.w_fc * E_fc
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item(),
            'E_dis': E_dis.item(),
            'E_pen': E_pen.item(),
            'E_spen': E_spen.item(),
            'E_joints': E_joints.item(),
            'E_fc': E_fc.item()
        }
        
        return total_loss, loss_dict
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.pointnet.train()
        self.cvae.train()
        
        epoch_losses = {key: [] for key in ['total', 'recon', 'kl', 'E_dis', 'E_pen', 'E_spen', 'E_joints', 'E_fc']}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            point_cloud = batch['point_cloud'].to(self.device)  # (B, N, 3)
            hand_params = batch['hand_params'].unsqueeze(1).to(self.device)  # (B, 1, 28)
            associated_idx = batch['associated_point_idx'].to(self.device)  # (B,)
            
            # Forward pass through PointNet
            # Transpose for PointNet: (B, 3, N)
            point_cloud_t = point_cloud.transpose(1, 2)
            global_feat, local_feat = self.pointnet(point_cloud_t, associated_idx)
            
            # Forward pass through CVAE
            recon_hand, mu, logvar = self.cvae(hand_params, global_feat, local_feat)
            
            # Compute loss
            loss, loss_dict = self.compute_loss(batch, recon_hand, mu, logvar)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.pointnet.parameters()) + list(self.cvae.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()
            
            # Log losses
            for key, val in loss_dict.items():
                epoch_losses[key].append(val)
            
            # Log to TensorBoard every 100 batches
            if self.writer is not None and batch_idx % 100 == 0:
                global_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Batch/train_loss', loss_dict['total'], global_step)
                self.writer.add_scalar('Batch/train_recon', loss_dict['recon'], global_step)
                self.writer.add_scalar('Batch/train_kl', loss_dict['kl'], global_step)
                self.writer.add_scalar('Batch/E_dis', loss_dict['E_dis'], global_step)
                self.writer.add_scalar('Batch/E_pen', loss_dict['E_pen'], global_step)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss_dict['total'],
                'recon': loss_dict['recon'],
                'kl': loss_dict['kl']
            })
        
        # Average losses
        avg_losses = {key: np.mean(vals) for key, vals in epoch_losses.items()}
        return avg_losses
    
    def validate(self):
        """Validate on validation set."""
        self.pointnet.eval()
        self.cvae.eval()
        
        val_losses = {key: [] for key in ['total', 'recon', 'kl', 'E_dis', 'E_pen', 'E_spen', 'E_joints', 'E_fc']}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                point_cloud = batch['point_cloud'].to(self.device)
                hand_params = batch['hand_params'].unsqueeze(1).to(self.device)
                associated_idx = batch['associated_point_idx'].to(self.device)
                
                point_cloud_t = point_cloud.transpose(1, 2)
                global_feat, local_feat = self.pointnet(point_cloud_t, associated_idx)
                
                recon_hand, mu, logvar = self.cvae(hand_params, global_feat, local_feat)
                
                _, loss_dict = self.compute_loss(batch, recon_hand, mu, logvar)
                
                for key, val in loss_dict.items():
                    val_losses[key].append(val)
        
        avg_losses = {key: np.mean(vals) for key, vals in val_losses.items()}
        return avg_losses
    
    def save_checkpoint(self, epoch, save_dir):
        """Save model checkpoint."""
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'pointnet_state': self.pointnet.state_dict(),
            'cvae_state': self.cvae.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config
        }
        path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.pointnet.load_state_dict(checkpoint['pointnet_state'])
        self.cvae.load_state_dict(checkpoint['cvae_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch']
    
    def train(self, num_epochs, save_dir='./checkpoints', log_dir='./runs'):
        """Main training loop."""
        print(f"\nStarting training for {num_epochs} epochs...")
        
        # Initialize TensorBoard writer
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_losses = self.train_epoch(epoch)
            
            # Validate
            val_losses = self.validate()
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train_total', train_losses['total'], epoch)
            self.writer.add_scalar('Loss/val_total', val_losses['total'], epoch)
            self.writer.add_scalar('Loss/train_recon', train_losses['recon'], epoch)
            self.writer.add_scalar('Loss/train_kl', train_losses['kl'], epoch)
            self.writer.add_scalar('Loss/val_recon', val_losses['recon'], epoch)
            self.writer.add_scalar('Loss/val_kl', val_losses['kl'], epoch)
            
            # Log energy terms
            for key in ['E_dis', 'E_pen', 'E_spen', 'E_joints', 'E_fc']:
                self.writer.add_scalar(f'Energy/train_{key}', train_losses[key], epoch)
                self.writer.add_scalar(f'Energy/val_{key}', val_losses[key], epoch)
            
            # Print summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_losses['total']:.4f} | Val Loss: {val_losses['total']:.4f}")
            print(f"  Recon: {train_losses['recon']:.4f} | KL: {train_losses['kl']:.4f}")
            print(f"  E_dis: {train_losses['E_dis']:.4f} | E_pen: {train_losses['E_pen']:.4f}")
            print(f"  E_spen: {train_losses['E_spen']:.4f} | E_joints: {train_losses['E_joints']:.4f}")
            
            # Save checkpoint
            if epoch % self.config['save_every'] == 0:
                self.save_checkpoint(epoch, save_dir)
            
            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                os.makedirs(save_dir, exist_ok=True)
                checkpoint = {
                    'epoch': epoch,
                    'pointnet_state': self.pointnet.state_dict(),
                    'cvae_state': self.cvae.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'config': self.config
                }
                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pt'))
                print(f"  → Saved best model (val_loss={best_val_loss:.4f})")
        
        self.writer.close()
        print("\nTraining complete!")