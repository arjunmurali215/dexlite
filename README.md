# DexLite: Simplified Dexterous Grasp Generation

A lightweight implementation of dexterous grasp synthesis for the Shadow Hand, inspired by the methods from **[Dex1B: Learning with 1B Demonstrations for Dexterous Manipulation](https://arxiv.org/abs/2506.17198)** by Ye et al.

## Overview

This repository implements a **Conditional Variational Autoencoder (CVAE)** combined with **PointNet** to generate dexterous grasps for robotic manipulation. The model learns to synthesize Shadow Hand configurations conditioned on object geometry, enabling grasp generation for diverse objects without restriction to tabletop scenarios.

### Key Features

- **PointNet-based Object Encoding**: Processes 3D point clouds to extract both global and local geometric features
- **Conditional VAE Architecture**: Generates diverse grasp poses conditioned on object features
- **Physics-Informed Training**: Incorporates multiple energy terms (contact, penetration, force closure) for physically plausible grasps
- **Single-Pose Generation**: Focuses on static grasp pose synthesis rather than full trajectories
- **Shadow Hand Model**: 28-DOF parametrization (3 translation + 3 rotation + 22 joints)

## Architecture

```
Object Point Cloud (1024 points)
         ↓
    PointNet Encoder
         ↓
  Global Features (256) + Local Features (256)
         ↓
    CVAE (Latent: 256)
         ↓
  Hand Parameters (28-DOF)
```

### Model Components

1. **PointNet**: Extracts object geometry features from point clouds
   - Global feature: 256-dim vector representing the entire object
   - Local features: 256-dim per-point features for contact modeling

2. **CVAE**: Conditional variational autoencoder
   - Encoder: [256, 512, 256] → Latent (256)
   - Decoder: [512, 256, 128] → Hand params (28)
   - Conditions: Concatenated global and local object features (512-dim)

3. **Hand Model**: Shadow Hand with 22 actuated joints
   - 6 DOF pose (translation + rotation)
   - 22 joint angles (finger articulation)

## Training

The model is trained with a multi-objective loss function:

```
L_total = w_recon * L_recon + w_kl * L_kl + 
          w_dis * E_dis + w_pen * E_pen + 
          w_spen * E_spen + w_joints * E_joints + 
          w_fc * E_fc
```

Where:
- **L_recon**: Reconstruction loss (MSE between input and output hand params)
- **L_kl**: KL divergence (regularizes latent space)
- **E_dis**: Contact distance energy (encourages contact with object)
- **E_pen**: Penetration energy (penalizes hand-object interpenetration)
- **E_spen**: Self-penetration energy (prevents hand self-collision)
- **E_joints**: Joint limit energy (keeps joints within valid ranges)
- **E_fc**: Force closure energy (promotes stable grasps)

### Training the Model

```bash
python main.py \
    --grasp_root /path/to/validated_grasps \
    --mesh_root /path/to/meshes \
    --batch_size 32 \
    --num_epochs 100 \
    --lr 1e-5 \
    --save_dir ./checkpoints \
    --log_dir ./runs
```

**Dataset**: This implementation was trained on a filtered version of **DexGraspNet**, without restriction to tabletop-only grasps. The dataset should contain:
- Validated grasp files (`.npy` format with Shadow Hand configurations)
- Object meshes (`.obj` format with CoACD decomposition)

### Key Training Arguments

- `--w_kl`: KL divergence weight (default: 1e-4)
- `--w_recon`: Reconstruction loss weight (default: 1.0)
- `--w_dis`: Contact distance weight (default: 1e-4)
- `--w_pen`: Penetration weight (default: 1e-4)
- `--w_spen`: Self-penetration weight (default: 1e-5)
- `--w_joints`: Joint limits weight (default: 1e-6)
- `--w_fc`: Force closure weight (default: 1.0)

## Inference

### Grasp Reconstruction

Reconstruct grasps by encoding existing grasps and decoding them back:

```bash
python infer.py \
    --checkpoint checkpoints/model_epoch_100.pt \
    --grasp_root /path/to/validated_grasps \
    --mesh_root /path/to/meshes \
    --output_dir ./reconstructed_grasps \
    --batch_size 64
```

### Grasp Synthesis

Generate novel grasps by sampling from the latent space:

```bash
python synthesize.py \
    --checkpoint checkpoints/model_epoch_100.pt \
    --mesh_root /path/to/meshes \
    --output_dir ./synthesized_grasps \
    --num_samples 10 \
    --batch_size 32
```

This samples `z ~ N(0, I)` from the latent space and decodes conditioned on object features to generate diverse grasp candidates.

## Post-Processing

After generation, grasps can be refined using gradient-based optimization:

```bash
python post_optimize.py \
    --grasp_file ./synthesized_grasps/object_code.npy \
    --mesh_path /path/to/object.obj \
    --output_file ./optimized_grasps/object_code.npy \
    --num_iterations 100
```

This performs energy minimization to improve grasp quality based on the physics-based energy terms.

## Project Structure

```
dexlite/
├── main.py                 # Training entry point
├── train.py               # Training loop and loss computation
├── dataset.py             # Grasp dataset loader
├── infer.py               # Grasp reconstruction from trained model
├── synthesize.py          # Novel grasp generation via sampling
├── post_optimize.py       # Gradient-based grasp refinement
├── networks/
│   ├── cvae.py           # Conditional VAE architecture
│   └── pointnet.py       # PointNet encoder
├── data_utils/
│   ├── stability_filter.py
│   └── test_grasp_mujoco.py
└── DexGraspNet/          # DexGraspNet utilities (hand model, physics)
```

## Requirements

```bash
# Core dependencies
torch>=2.0.0
numpy
trimesh
scipy
tensorboard
tqdm

# For hand model and physics
mujoco
transforms3d
pytorch-kinematics
```

## Citation

This implementation is inspired by the Dex1B paper:

```bibtex
@article{ye2025dex1b,
  title={Dex1B: Learning with 1B Demonstrations for Dexterous Manipulation},
  author={Ye, Jianglong and Wang, Keyi and Yuan, Chengjing and Yang, Ruihan and Li, Yiquan and Zhu, Jiyue and Qin, Yuzhe and Zou, Xueyan and Wang, Xiaolong},
  journal={arXiv preprint arXiv:2506.17198},
  year={2025}
}
```

Training dataset: **DexGraspNet** (Wang et al., 2023) - A filtered version was used without tabletop restrictions.

## License

This project is provided for research purposes. Please refer to the original papers for their respective licenses.

## Notes

- This is a **simplified implementation** focused on single-pose grasp generation
- The original Dex1B work encompasses broader aspects of dexterous manipulation with 1B demonstrations
- Physics simulation uses MuJoCo through the Shadow Hand model
- The model architecture follows CVAE principles but is adapted for this specific grasp synthesis task
