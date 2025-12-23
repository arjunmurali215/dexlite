import torch
import torch.nn as nn
import torch.nn.functional as F

class DexSimpleCVAE(nn.Module):
    def __init__(self, n_frames=1, latent_dim=256):
        """
        DexSimple CVAE for Shadow Hand.
        
        Args:
            n_frames (int): Number of frames in the trajectory (Default 1 for single pose).
            latent_dim (int): Size of the latent space (Paper uses 256).
        """
        super(DexSimpleCVAE, self).__init__()

        # -----------------------------------------------------------
        # 1. DIMENSION CALCULATIONS
        # -----------------------------------------------------------
        # Shadow Hand Params: 3 (Trans) + 3 (Rot) + 22 (Joints) = 28 
        self.hand_dof = 28 
        self.input_dim = n_frames * self.hand_dof
        
        # Condition Params: 256 (Global f_obj) + 256 (Local f_p) = 512
        self.condition_dim = 512 
        
        self.latent_dim = latent_dim

        # -----------------------------------------------------------
        # 2. ENCODER (The "Training" Path)
        # Input: Flattened Hand Trajectory + Conditions
        # Layers: (256, 512, 256) -> Latent 
        # -----------------------------------------------------------
        encoder_input_size = self.input_dim + self.condition_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Output layers for Mean and Log-Variance
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # -----------------------------------------------------------
        # 3. DECODER (The "Generation" Path)
        # Input: Latent Vector + Conditions
        # Layers: (256, 512, 256) -> Reconstruction 
        # -----------------------------------------------------------
        decoder_input_size = latent_dim + self.condition_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            # Final projection back to hand parameter space
            nn.Linear(256, self.input_dim) 
        )

    def reparameterize(self, mu, logvar):
        """
        Standard VAE Reparameterization Trick.
        z = mu + sigma * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, hand_traj, global_feat, local_feat=None):
        """
        Args:
            hand_traj:  (Batch, N_Frames, 28) - Ground Truth Hand Poses
            global_feat: (Batch, 256) - From PointNet
            local_feat:  (Batch, 256) - From PointNet (Optional)
        """
        batch_size = hand_traj.size(0)

        # -----------------------------------------------------------
        # A. PREPARE CONDITIONS (Concatenate Embeddings)
        # -----------------------------------------------------------
        # Handle optional local feature (Zero-masking if missing)
        if local_feat is None:
            local_feat = torch.zeros_like(global_feat)
            
        # Combine Global and Local features [cite: 195]
        # Shape: (Batch, 512)
        condition_vector = torch.cat([global_feat, local_feat], dim=1)

        # -----------------------------------------------------------
        # B. ENCODE
        # -----------------------------------------------------------
        # Flatten input trajectory: (Batch, N*28)
        flat_input = hand_traj.view(batch_size, -1)
        
        # Concat Input + Condition for Encoder
        enc_input = torch.cat([flat_input, condition_vector], dim=1)
        
        # Pass through Encoder MLP
        enc_hidden = self.encoder(enc_input)
        mu = self.fc_mu(enc_hidden)
        logvar = self.fc_logvar(enc_hidden)

        # -----------------------------------------------------------
        # C. SAMPLE
        # -----------------------------------------------------------
        z = self.reparameterize(mu, logvar)

        # -----------------------------------------------------------
        # D. DECODE
        # -----------------------------------------------------------
        # Concat Latent z + Condition for Decoder
        dec_input = torch.cat([z, condition_vector], dim=1)
        
        # Pass through Decoder MLP
        reconstruction_flat = self.decoder(dec_input)
        
        # Reshape back to trajectory format
        reconstruction = reconstruction_flat.view(batch_size, -1, self.hand_dof)

        return reconstruction, mu, logvar

    def inference(self, global_feat, local_feat=None):
        """
        Generation Mode: No hand input required.
        """
        batch_size = global_feat.size(0)
        
        # 1. Prepare Conditions
        if local_feat is None:
            local_feat = torch.zeros_like(global_feat)
        condition_vector = torch.cat([global_feat, local_feat], dim=1)
        
        # 2. Sample Random Noise z ~ N(0, I)
        z = torch.randn(batch_size, self.latent_dim).to(global_feat.device)
        
        # 3. Decode
        dec_input = torch.cat([z, condition_vector], dim=1)
        reconstruction_flat = self.decoder(dec_input)
        
        return reconstruction_flat.view(batch_size, -1, self.hand_dof)