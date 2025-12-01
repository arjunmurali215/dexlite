import torch
import torch.nn as nn
import torch.nn.functional as F

class DexPointNet(nn.Module):
    def __init__(self, output_dim=256):
        super(DexPointNet, self).__init__()
        
        # Based on Table VI: PointNet Layer Sizes (3, 64, 128, 1024, 256)
        # We implement these as 1D Convolutions (kernel size 1)
        # This acts as a Shared MLP applied to every point identically.
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.conv4 = nn.Conv1d(1024, output_dim, 1) # Compresses to 256

        # Batch Normalization for stability (Standard in PointNet)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(output_dim)

    def forward(self, x, local_point_indices=None):
        """
        Args:
            x: Point Cloud tensor of shape (Batch_Size, 3, Num_Points)
            local_point_indices: Indices of the 'associated points' (Batch_Size,)
        Returns:
            global_feat: (Batch_Size, 256)
            local_feat:  (Batch_Size, 256) or None
        """
        # 1. Layer 1: 3 -> 64
        x = F.relu(self.bn1(self.conv1(x)))
        
        # 2. Layer 2: 64 -> 128
        x = F.relu(self.bn2(self.conv2(x)))
        
        # 3. Layer 3: 128 -> 1024
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 4. Layer 4: 1024 -> 256 (Final Feature Projection)
        point_features = self.bn4(self.conv4(x)) # Shape: (B, 256, N)

        global_feat = torch.max(point_features, 2, keepdim=False)[0] 
        # Shape: (B, 256)

        local_feat = None
        if local_point_indices is not None:
            batch_size = x.size(0)
            idx = local_point_indices.view(batch_size, 1, 1).expand(-1, 256, -1)
            
            # Gather: Extract the column corresponding to the point index
            local_feat = torch.gather(point_features, 2, idx).squeeze(2)
            # Shape: (B, 256)

        return global_feat, local_feat