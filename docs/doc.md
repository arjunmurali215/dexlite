---
title: "DexLite Grasp Synthesis"
date: "2025-12-20"
excerpt: "Learning-based dexterous grasp synthesis with physics-informed optimization."
coverImage: "./assets/dexlite.mp4"
---

# DexLite: Learning-Based Dexterous Grasp Synthesis with Physics-Informed Optimization

> **Note:** This project is an implementation of the grasp synthesis methodology presented in the paper **Dex1B: Learning with 1B Demonstrations for Dexterous Manipulation** [1]. The neural network architecture, energy functions, and optimization procedures described herein are based on their published work, adapted and extended for this implementation.

**Abstract:** This report presents DexLite, a learning-based approach for synthesizing dexterous grasps for a high-DOF robotic hand (Shadow Hand). The system combines conditional variational autoencoders (CVAE) with PointNet-based object encoding to generate grasp configurations, followed by physics-informed post-optimization to ensure grasp quality. I leverage a curated subset of DexGraspNet grasps as seed data, filtered through stability analysis and physics simulation validation.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Optimization-Based Grasp Synthesis (DexGraspNet Pipeline)](#2-optimization-based-grasp-synthesis-dexgraspnet-pipeline)
3. [Seed Dataset Preparation](#3-seed-dataset-preparation)
4. [Neural Network Architecture](#4-neural-network-architecture)
5. [Training Losses and Objectives](#5-training-losses-and-objectives)
6. [Dataset Expansion and Grasp Generation](#6-dataset-expansion-and-grasp-generation)
7. [Grasp Synthesis (Inference)](#7-grasp-synthesis-inference)
8. [Post-Optimization](#8-post-optimization)
9. [Results](#9-results)
10. [Future Work](#10-future-work)

---

## 1. Introduction

Dexterous manipulation with high-DOF robotic hands remains one of the most challenging problems in robotics. Unlike parallel-jaw grippers with 1-2 DOF, the Shadow Hand possesses 24 actuated degrees of freedom, enabling human-like dexterity but dramatically increasing the search space for grasp planning.

![Example Synthesized Grasp](assets/dexlite.mp4)

### 1.1 Problem Formulation

Given an object represented by a point cloud $\mathcal{P} \in \mathbb{R}^{N \times 3}$, our goal is to find a hand configuration $\mathbf{q} \in \mathbb{R}^{28}$ that results in a stable, force-closure grasp. The hand configuration consists of:

$$\mathbf{q} = [\mathbf{t}, \boldsymbol{\theta}_{\text{rot}}, \boldsymbol{\theta}_{\text{joints}}]$$

where:
- $\mathbf{t} \in \mathbb{R}^3$: Wrist translation
- $\boldsymbol{\theta}_{\text{rot}} \in \mathbb{R}^3$: Wrist rotation (Euler angles)
- $\boldsymbol{\theta}_{\text{joints}} \in \mathbb{R}^{22}$: Joint angles for the Shadow Hand

### 1.2 System Overview

![System Pipeline](assets/system_pipeline.svg)

The DexLite pipeline consists of four main stages:

1. **Seed Dataset Preparation**: Filter and validate grasps from DexGraspNet
2. **Training**: Learn grasp distribution using CVAE conditioned on object features
3. **Synthesis**: Generate novel grasps by sampling from learned latent space
4. **Post-Optimization**: Refine generated grasps using physics-based optimization

---

## 2. Optimization-Based Grasp Synthesis (DexGraspNet Pipeline)

The original DexGraspNet approach uses differentiable simulation to synthesize grasps through gradient-based optimization. This section details the mathematical formulation of all energy terms.

### 2.1 Overall Energy Function

The total energy function to be minimized is:

$$E_{\text{total}} = E_{\text{fc}} + w_{\text{dis}} E_{\text{dis}} + w_{\text{pen}} E_{\text{pen}} + w_{\text{spen}} E_{\text{spen}} + w_{\text{joints}} E_{\text{joints}}$$

where the default weights are:
- $w_{\text{dis}} = 100.0$ (contact distance weight)
- $w_{\text{pen}} = 100.0$ (object penetration weight)
- $w_{\text{spen}} = 10.0$ (self-penetration weight)
- $w_{\text{joints}} = 1.0$ (joint limit weight)

### 2.2 Contact Distance Energy ($E_{\text{dis}}$)

This energy encourages contact points on the hand to be close to the object surface. Given $K$ contact points $\{\mathbf{c}_k\}_{k=1}^{K}$ on the hand and a signed distance function $\phi(\cdot)$ to the object surface:

$$E_{\text{dis}} = \sum_{k=1}^{K} |\phi(\mathbf{c}_k)|$$

The signed distance $\phi(\mathbf{c}_k)$ is computed using a pre-computed SDF or nearest-neighbor queries to the object surface. Minimizing this energy drives contact points toward the object surface.

### 2.3 Force Closure Energy ($E_{\text{fc}}$)

Force closure ensures the grasp can resist external wrenches. This is formulated using the grasp matrix $\mathbf{G}$. For contact points $\{\mathbf{c}_k\}$ with surface normals $\{\mathbf{n}_k\}$, the grasp matrix maps contact forces to object wrenches:

$$\mathbf{G} = \begin{bmatrix} \mathbf{I}_3 & \mathbf{I}_3 & \cdots & \mathbf{I}_3 \\ [\mathbf{c}_1]_\times & [\mathbf{c}_2]_\times & \cdots & [\mathbf{c}_K]_\times \end{bmatrix}$$

where $[\mathbf{c}_k]_\times$ is the skew-symmetric matrix for cross product:

$$[\mathbf{c}]_\times = \begin{bmatrix} 0 & -c_z & c_y \\ c_z & 0 & -c_x \\ -c_y & c_x & 0 \end{bmatrix}$$

The transformation matrix to convert contact normals to the grasp matrix space is:

$$\mathbf{T} = \begin{bmatrix} 0 & 0 & 0 & 0 & 0 & -1 & 0 & 1 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 & -1 & 0 & 0 \\ 0 & -1 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \end{bmatrix}$$

The force closure energy is then:

$$E_{\text{fc}} = \|\mathbf{n}^T \mathbf{G}\|_F^2$$

where $\mathbf{n}$ is the concatenated contact normal vector. This penalizes configurations where normals are poorly aligned with the grasp matrix null space.

### 2.4 Object Penetration Energy ($E_{\text{pen}}$)

To prevent the hand from penetrating the object, the algorithm samples $M$ points $\{\mathbf{p}_m\}$ from the object surface and computes their signed distance to the hand:

$$E_{\text{pen}} = \sum_{m=1}^{M} \max(0, d_{\text{hand}}(\mathbf{p}_m) + 1)$$

where $d_{\text{hand}}(\mathbf{p}_m)$ is the signed distance from point $\mathbf{p}_m$ to the hand surface. The $+1$ offset ensures a margin. Only positive distances (penetrations) are penalized.

### 2.5 Self-Penetration Energy ($E_{\text{spen}}$)

The hand has multiple links that should not interpenetrate. For each pair of potentially colliding links $(i, j)$, the algorithm computes:

$$E_{\text{spen}} = \sum_{(i,j) \in \mathcal{C}} \max(0, -d_{ij})$$

where $d_{ij}$ is the minimum distance between links $i$ and $j$, and $\mathcal{C}$ is the set of collision-checkable link pairs.

### 2.6 Joint Limit Energy ($E_{\text{joints}}$)

The Shadow Hand has physical joint limits $[\mathbf{q}^{-}, \mathbf{q}^{+}]$. The joint limit violation is:

$$E_{\text{joints}} = \sum_{i=1}^{22} \max(0, q_i - q_i^{+}) + \max(0, q_i^{-} - q_i)$$

This creates a soft constraint that penalizes configurations exceeding joint limits.

### 2.7 Simulated Annealing Optimization

The optimization uses a modified RMSProp with simulated annealing for escaping local minima:

**RMSProp Update:**
$$\mathbf{v}_t = \mu \mathbf{v}_{t-1} + (1-\mu)(\nabla E)^2$$
$$\mathbf{q}_{t+1} = \mathbf{q}_t - \frac{\alpha}{\sqrt{\mathbf{v}_t} + \epsilon} \nabla E$$

**Annealing Acceptance:**
$$P(\text{accept}) = \exp\left(\frac{E_{\text{old}} - E_{\text{new}}}{T}\right)$$

where temperature $T$ decays as:
$$T_k = T_0 \cdot \gamma^{\lfloor k/n_{\text{period}} \rfloor}$$

with $T_0 = 18$, $\gamma = 0.95$, and $n_{\text{period}} = 30$.

---

## 3. Seed Dataset Preparation

Due to computational constraints (RTX 4050 GPU), generating a seed dataset from scratch through optimization was infeasible. Instead, I curated a high-quality subset from the existing DexGraspNet dataset through a two-stage filtering process.

### 3.1 Object Stability Filtering

Not all objects in DexGraspNet are suitable for tabletop grasping. Objects that cannot stably rest on a table (e.g., spheres, thin rods) are filtered out using PyBullet physics simulation.

#### 3.1.1 Stability Test Protocol

1. **Scene Setup**: Create a flat plane (table surface) with realistic friction ($\mu = 1.0$)

2. **Object Placement**: Place the object slightly above the table with a small initial perturbation:
   - Position: $[0, 0, 1.0]$ meters (to allow settling)
   - Orientation: Small random rotation $[0.1, 0.1, 0.1]$ radians (Euler)

3. **Simulation**: Run for 480 timesteps (2 seconds at 240Hz)

4. **Stability Criteria**: An object passes if after settling:
   
   **Position criterion** - Lateral displacement $< 0.05$ m:
   $$\|\mathbf{p}_{xy}^{\text{end}} - \mathbf{p}_{xy}^{\text{start}}\|_2 < 0.05$$
   
   **Orientation criterion** - Angular change $< 7°$:
   $$2 \arccos(|\langle \mathbf{q}^{\text{start}}, \mathbf{q}^{\text{end}} \rangle|) < 7°$$

#### 3.1.2 Filtering Results

| Category | Count |
|----------|-------|
| Total Objects (DexGraspNet) | ~5,500 |
| Objects Passing Stability Test | ~3,000 |
| Rejection Rate | ~45% |

Objects failing stability tests include thin plates, spheres, and objects with high centers of mass.

### 3.2 Grasp Validation in MuJoCo

Even with stable objects, not all grasps from DexGraspNet are physically realizable. I validate each grasp using MuJoCo physics simulation with a lift test.

#### 3.2.1 Simulation Setup

The MuJoCo scene includes:
- **Hand Model**: Shadow Hand with capsule collision primitives (more stable than mesh-mesh collision)
- **Object Model**: Convex decomposition meshes from CoACD for accurate collision
- **Physics Parameters**:
  - Timestep: 0.01s
  - Friction: 1.0
  - Contact margin: 0.001m

#### 3.2.2 Lift Test Protocol

1. **Initialization Phase** (1.0s): Stabilize the grasp with position-controlled fingers
   - Actuator gains: $k_p = 100$, $k_v = 5$
   - Apply slight finger tightening bias: $+0.07$ rad on flexion joints

2. **Lift Phase** (2.0s): Move the hand upward at constant velocity
   - Lift distance: 1.0m
   - Velocity: 0.5 m/s
   - Hand follows mocap body via weld constraint

3. **Success Criterion**: Object lift ratio $> 0.7$
   
   $$r_{\text{lift}} = \frac{z_{\text{object}}^{\text{end}} - z_{\text{object}}^{\text{start}}}{z_{\text{hand}}^{\text{end}} - z_{\text{hand}}^{\text{start}}} > 0.7$$

#### 3.2.3 Validation Results

| Metric | Value |
|--------|-------|
| Total Grasps Tested | ~550,000 |
| Grasps Passing Lift Test | ~495,000 |
| Pass Rate | ~90% |

The high pass rate indicates DexGraspNet grasps are generally high quality, with failures primarily due to:
- Edge cases with thin objects
- Grasps relying on friction beyond simulation limits
- Minor penetration issues

---

## 4. Neural Network Architecture

![Neural Network Architecture](assets/network_architecture.svg)

### 4.1 System Overview

The learning system consists of two main components:
1. **PointNet**: Extracts geometric features from object point clouds
2. **CVAE**: Generates grasp configurations conditioned on object features

### 4.2 PointNet Feature Extractor

PointNet processes the object point cloud to extract both global shape features and local point features.

#### 4.2.1 Architecture

![PointNet Architecture](assets/pointnet_architecture.svg)

| Layer | Input Dim | Output Dim | Operation |
|-------|-----------|------------|-----------|
| Conv1 | 3 | 64 | 1D Conv + BN + ReLU |
| Conv2 | 64 | 128 | 1D Conv + BN + ReLU |
| Conv3 | 128 | 1024 | 1D Conv + BN + ReLU |
| Conv4 | 1024 | 256 | 1D Conv + BN |

**Input**: Point cloud $\mathcal{P} \in \mathbb{R}^{B \times 3 \times N}$ (batch × channels × points)

**Processing**:

For each point $\mathbf{p}_i \in \mathbb{R}^3$:
$$\mathbf{f}_i^{(1)} = \text{ReLU}(\text{BN}(\text{Conv1d}(\mathbf{p}_i))) \in \mathbb{R}^{64}$$
$$\mathbf{f}_i^{(2)} = \text{ReLU}(\text{BN}(\text{Conv1d}(\mathbf{f}_i^{(1)}))) \in \mathbb{R}^{128}$$
$$\mathbf{f}_i^{(3)} = \text{ReLU}(\text{BN}(\text{Conv1d}(\mathbf{f}_i^{(2)}))) \in \mathbb{R}^{1024}$$
$$\mathbf{f}_i^{(4)} = \text{BN}(\text{Conv1d}(\mathbf{f}_i^{(3)})) \in \mathbb{R}^{256}$$

**Global Feature** (via symmetric max pooling):
$$\mathbf{f}_{\text{global}} = \max_{i \in [1,N]} \mathbf{f}_i^{(4)} \in \mathbb{R}^{256}$$

**Local Feature** (at associated point index $j$):
$$\mathbf{f}_{\text{local}} = \mathbf{f}_j^{(4)} \in \mathbb{R}^{256}$$

#### 4.2.2 Significance

- **Permutation Invariance**: The max-pooling operation makes the global feature invariant to point ordering
- **Local Feature**: Captures geometry at the specific grasp approach point
- **Multi-Scale**: Hierarchical feature extraction captures both fine details and global shape

### 4.3 Conditional Variational Autoencoder (CVAE)

The CVAE learns the distribution of grasp configurations conditioned on object features.

#### 4.3.1 Architecture

![CVAE Architecture](assets/cvae_architecture.svg)

**Encoder** (Training only):
| Layer | Input Dim | Output Dim |
|-------|-----------|------------|
| FC1 | 28 + 512 = 540 | 256 |
| FC2 | 256 | 512 |
| FC3 | 512 | 256 |
| FC_μ | 256 | 256 |
| FC_σ | 256 | 256 |

**Decoder** (Training and Inference):
| Layer | Input Dim | Output Dim |
|-------|-----------|------------|
| FC1 | 256 + 512 = 768 | 256 |
| FC2 | 256 | 512 |
| FC3 | 512 | 256 |
| FC_out | 256 | 28 |

#### 4.3.2 Mathematical Formulation

**Condition Vector**:
$$\mathbf{c} = [\mathbf{f}_{\text{global}}; \mathbf{f}_{\text{local}}] \in \mathbb{R}^{512}$$

**Encoder** (approximate posterior $q_\phi(\mathbf{z}|\mathbf{q}, \mathbf{c})$):
$$\mathbf{h}_{\text{enc}} = \text{MLP}_{\text{enc}}([\mathbf{q}; \mathbf{c}])$$
$$\boldsymbol{\mu} = W_\mu \mathbf{h}_{\text{enc}} + \mathbf{b}_\mu$$
$$\log \boldsymbol{\sigma}^2 = W_\sigma \mathbf{h}_{\text{enc}} + \mathbf{b}_\sigma$$

**Reparameterization Trick**:
$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

**Decoder** (likelihood $p_\theta(\mathbf{q}|\mathbf{z}, \mathbf{c})$):
$$\hat{\mathbf{q}} = \text{MLP}_{\text{dec}}([\mathbf{z}; \mathbf{c}])$$

### 4.4 Associated Point Computation

The "associated point" determines which object surface point is most relevant to a grasp, providing local geometric context.

#### 4.4.1 Heading Vector Computation

Using forward kinematics (FK) on the hand model:

1. Compute fingertip positions:
   $$\mathbf{p}_{\text{thumb}} = \text{FK}(\mathbf{q})[\text{thdistal}]$$
   $$\mathbf{p}_{\text{middle}} = \text{FK}(\mathbf{q})[\text{mfdistal}]$$

2. Compute palm center (with offset $[0.006, 0, 0.036]$ in palm frame):
   $$\mathbf{p}_{\text{palm}} = R_{\text{palm}} \cdot [0.006, 0, 0.036]^T + \mathbf{t}_{\text{palm}}$$

3. Heading vector from palm to grasp center:
   $$\mathbf{h} = \frac{(\mathbf{p}_{\text{thumb}} + \mathbf{p}_{\text{middle}})/2 - \mathbf{p}_{\text{palm}}}{\|(\mathbf{p}_{\text{thumb}} + \mathbf{p}_{\text{middle}})/2 - \mathbf{p}_{\text{palm}}\|}$$

#### 4.4.2 Associated Point Selection

Find the point on the object surface closest to the heading ray:

$$j^* = \arg\min_{j: \mathbf{h}^T(\mathbf{p}_j - \mathbf{p}_{\text{palm}}) > 0} \|(\mathbf{p}_j - \mathbf{p}_{\text{palm}}) - ((\mathbf{p}_j - \mathbf{p}_{\text{palm}})^T \mathbf{h})\mathbf{h}\|$$

This finds the object point that lies closest to the ray from the palm center in the grasping direction.

---

## 5. Training Losses and Objectives

The CVAE training combines a standard VAE loss with physics-informed energy terms that penalize implausible grasp configurations.

### 5.1 Reconstruction Loss

The reconstruction loss measures how well the decoder reconstructs the input grasp configuration:

$$\mathcal{L}_{\text{recon}} = \text{MSE}(\mathbf{q}, \hat{\mathbf{q}}) = \frac{1}{B} \sum_{b=1}^{B} \|\mathbf{q}_b - \hat{\mathbf{q}}_b\|_2^2$$

where $\mathbf{q}$ is the ground truth grasp and $\hat{\mathbf{q}}$ is the reconstructed grasp.

### 5.2 KL Divergence Loss

The KL divergence regularizes the approximate posterior to be close to a standard normal prior:

$$\mathcal{L}_{\text{KL}} = -\frac{1}{2B} \sum_{b=1}^{B} \sum_{i=1}^{d_z} \left(1 + \log \sigma_{b,i}^2 - \mu_{b,i}^2 - \sigma_{b,i}^2\right)$$

where $d_z = 256$ is the latent dimension, $\mu$ is the encoder mean, and $\sigma^2 = \exp(\log\sigma^2)$ is the variance.

### 5.3 Energy-Based Losses

The following energy terms are computed on the reconstructed grasp $\hat{\mathbf{q}}$ and penalize physically implausible configurations. These follow the exact formulations from DexGraspNet's `energy.py`:

**Contact Distance Energy ($E_{\text{dis}}$)**:

$$E_{\text{dis}} = \sum_{k=1}^{K} |d(\mathbf{c}_k, \mathcal{S})|$$

where $\mathbf{c}_k$ are contact points on the hand and $d(\mathbf{c}_k, \mathcal{S})$ is the signed distance from contact point $k$ to the object surface $\mathcal{S}$.

**Force Closure Energy ($E_{\text{fc}}$)**:

$$E_{\text{fc}} = \|\mathbf{n}^T \mathbf{G}\|_F^2$$

where $\mathbf{n}$ is the concatenated contact normal vector $(1 \times 3K)$ and $\mathbf{G}$ is the grasp matrix $(3K \times 6)$:

$$\mathbf{G} = \begin{bmatrix} \mathbf{I}_3 & \mathbf{I}_3 & \cdots & \mathbf{I}_3 \\ [\mathbf{c}_1]_\times & [\mathbf{c}_2]_\times & \cdots & [\mathbf{c}_K]_\times \end{bmatrix}^T$$

**Joint Limit Energy ($E_{\text{joints}}$)**:

$$E_{\text{joints}} = \sum_{i=1}^{22} \mathbf{1}[\theta_i > \theta_i^{+}](\theta_i - \theta_i^{+}) + \sum_{i=1}^{22} \mathbf{1}[\theta_i < \theta_i^{-}](\theta_i^{-} - \theta_i)$$

where $\theta_i^{+}$ and $\theta_i^{-}$ are upper and lower joint limits respectively.

**Object Penetration Energy ($E_{\text{pen}}$)**:

$$E_{\text{pen}} = \left\| \max(0, d_{\text{hand}}(\mathbf{p}_m) + 1) \right\|_2$$

where $d_{\text{hand}}(\mathbf{p}_m)$ is the signed distance from object surface point $\mathbf{p}_m$ to the hand. The $+1$ offset provides a safety margin.

**Self-Penetration Energy ($E_{\text{spen}}$)**:

$$E_{\text{spen}} = \text{hand\_model.self\_penetration()}$$

This is computed internally by the hand model, checking for interpenetration between finger links.

### 5.4 Total Training Loss

$$\mathcal{L}_{\text{total}} = w_{\text{recon}} \mathcal{L}_{\text{recon}} + w_{\text{KL}} \mathcal{L}_{\text{KL}} + w_{\text{fc}} E_{\text{fc}} + w_{\text{dis}} E_{\text{dis}} + w_{\text{pen}} E_{\text{pen}} + w_{\text{spen}} E_{\text{spen}} + w_{\text{joints}} E_{\text{joints}}$$

**Default Weights**:
| Weight | Value | Purpose |
|--------|-------|---------|
| $w_{\text{recon}}$ | 1.0 | Reconstruction fidelity |
| $w_{\text{KL}}$ | 0.001 | Latent regularization |
| $w_{\text{fc}}$ | 1.0 | Force closure quality |
| $w_{\text{dis}}$ | 100.0 | Contact proximity |
| $w_{\text{pen}}$ | 100.0 | Object collision avoidance |
| $w_{\text{spen}}$ | 10.0 | Self-collision avoidance |
| $w_{\text{joints}}$ | 1.0 | Joint feasibility |

---

## 6. Dataset Expansion and Grasp Generation

### 6.1 Training Data Flow

![Training Pipeline](assets/training_pipeline.png)

### 6.2 Data Augmentation (Optional)

To improve generalization, the following augmentations can be applied:

1. **Random Rotation**: Apply random SO(3) rotation to both object and grasp
2. **Point Jittering**: Add small Gaussian noise to point cloud
3. **Random Sampling**: Different point cloud samples each epoch

---

## 7. Grasp Synthesis (Inference)

### 7.1 Synthesis Pipeline

During inference, the pipeline bypasses the encoder and samples directly from the prior:

![Inference Pipeline](assets/inference_pipeline.png)

### 7.2 Multiple Grasp Generation

For a single object, multiple diverse grasps can be generated by:

1. Sampling different latent codes: $\mathbf{z}_1, \mathbf{z}_2, \ldots, \mathbf{z}_K \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
2. Using different associated points from the point cloud
3. Each $(\mathbf{z}_i, \mathbf{f}_{\text{local}}^{(j)})$ pair produces a unique grasp

### 7.3 Grasp Output Format

The synthesized grasp is converted back to the qpos dictionary format:

```python
qpos = {
    'WRJTx': t_x,      # Wrist translation X
    'WRJTy': t_y,      # Wrist translation Y
    'WRJTz': t_z,      # Wrist translation Z
    'WRJRx': r_x,      # Wrist rotation X (Euler)
    'WRJRy': r_y,      # Wrist rotation Y (Euler)
    'WRJRz': r_z,      # Wrist rotation Z (Euler)
    'robot0:FFJ3': ...,  # First finger abduction
    'robot0:FFJ2': ...,  # First finger MCP
    'robot0:FFJ1': ...,  # First finger PIP
    'robot0:FFJ0': ...,  # First finger DIP
    # ... (22 joint angles total)
}
```

---

## 8. Post-Optimization

Post-optimization is **critical** for converting raw network outputs into physically valid grasps. The neural network provides a good initialization, but fine-tuning is necessary to achieve contact and avoid penetrations.

### 8.1 Importance of Post-Optimization

| Without Post-Opt | With Post-Opt |
|-----------------|---------------|
| ~60% contact achieved | ~95% contact achieved |
| Frequent penetrations | Minimal penetrations |
| ~50% grasp success | ~79% grasp success |

The network learns the general grasp distribution but cannot precisely achieve the sub-millimeter accuracy needed for contact.

### 8.2 Post-Optimization Algorithm

The same optimization procedure from DexGraspNet is applied, but initialized from network output instead of random:

```
Algorithm: Post-Optimization
─────────────────────────────────────────
Input: Initial grasp q₀ (from CVAE)
       Object model O
       Iterations N = 200
       
1. Initialize hand model with q₀
2. Initialize contact point indices randomly
3. for i = 1 to N do:
   4. Compute energy E(q, O)
   5. Compute gradients ∇E
   6. Update q using RMSProp
   7. Randomly switch contact point indices
   8. Apply annealing acceptance criterion
   9. if accepted: q ← q_new
10. return optimized q
```

### 8.3 Optimization Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_iter` | 200 | Optimization iterations |
| `step_size` | 0.005 | Initial step size |
| `starting_temperature` | 18 | Initial annealing temperature |
| `temperature_decay` | 0.95 | Temperature decay rate |
| `annealing_period` | 30 | Steps between temperature updates |
| `switch_possibility` | 0.5 | Probability of switching contact points |
| `mu` | 0.98 | RMSProp momentum |

### 8.4 Benefits of Network Initialization

Starting from network output vs. random initialization:

1. **Faster Convergence**: ~3x fewer iterations needed
2. **Better Local Minima**: Network provides semantically meaningful starting point
3. **Higher Success Rate**: Avoids bad local minima that trap random starts
4. **Consistent Quality**: More predictable optimization outcomes

---

## 9. Results

### 9.3 Grasp Synthesis Quality

| Metric | Before Post-Opt | After Post-Opt |
|--------|-----------------|----------------|
| MuJoCo Lift Test Success | ~50% | ~79% |

---

## 10. Future Work

1. **Incorporating Graspness**: Find graspness across points on the object to decide best direction to grasp from

2. **Multi-Hand Grasps**: Extend for dual hand grasps for bigger objects

3. **Real-World Transfer**

---

## References

1. Jianglong Ye, Keyi Wang, et al. "Dex1B: Learning with 1B Demonstrations for Dexterous Manipulation."

2. Wang, R., Zhang, J., et al. "DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation." ICRA 2023.

3. Qi, C.R., Su, H., et al. "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation." CVPR 2017.

6. Todorov, E., Erez, T., Tassa, Y. "MuJoCo: A physics engine for model-based control." IROS 2012.

---

<!-- ## Appendix A: Shadow Hand Joint Configuration

| Joint Name | DOF | Range (rad) | Description |
|------------|-----|-------------|-------------|
| FFJ3 | 1 | [-0.349, 0.349] | First finger abduction |
| FFJ2 | 1 | [0, 1.571] | First finger MCP |
| FFJ1 | 1 | [0, 1.571] | First finger PIP |
| FFJ0 | 1 | [0, 1.571] | First finger DIP |
| MFJ3 | 1 | [-0.349, 0.349] | Middle finger abduction |
| MFJ2 | 1 | [0, 1.571] | Middle finger MCP |
| MFJ1 | 1 | [0, 1.571] | Middle finger PIP |
| MFJ0 | 1 | [0, 1.571] | Middle finger DIP |
| RFJ3 | 1 | [-0.349, 0.349] | Ring finger abduction |
| RFJ2 | 1 | [0, 1.571] | Ring finger MCP |
| RFJ1 | 1 | [0, 1.571] | Ring finger PIP |
| RFJ0 | 1 | [0, 1.571] | Ring finger DIP |
| LFJ4 | 1 | [0, 0.785] | Little finger metacarpal |
| LFJ3 | 1 | [-0.349, 0.349] | Little finger abduction |
| LFJ2 | 1 | [0, 1.571] | Little finger MCP |
| LFJ1 | 1 | [0, 1.571] | Little finger PIP |
| LFJ0 | 1 | [0, 1.571] | Little finger DIP |
| THJ4 | 1 | [-1.047, 1.047] | Thumb rotation |
| THJ3 | 1 | [0, 1.222] | Thumb MCP |
| THJ2 | 1 | [-0.209, 0.209] | Thumb flexion |
| THJ1 | 1 | [-0.524, 0.524] | Thumb PIP |
| THJ0 | 1 | [-1.571, 0] | Thumb DIP |

**Total DOF**: 22 (joints) + 6 (wrist pose) = 28

---

## Appendix B: Energy Function Implementation Details

### B.1 Signed Distance Computation

For object penetration energy, the implementation uses TorchSDF for differentiable SDF computation:

```python
# Precompute face vertices for SDF
face_verts = index_vertices_by_faces(vertices, faces)

# Compute SDF at query points
sdf_values = compute_sdf(query_points, face_verts)
```

### B.2 Contact Normal Estimation

Contact normals are estimated from the object mesh using barycentric interpolation:

$$\mathbf{n}_c = \frac{\alpha \mathbf{n}_1 + \beta \mathbf{n}_2 + \gamma \mathbf{n}_3}{\|\alpha \mathbf{n}_1 + \beta \mathbf{n}_2 + \gamma \mathbf{n}_3\|}$$

where $(\alpha, \beta, \gamma)$ are barycentric coordinates and $(\mathbf{n}_1, \mathbf{n}_2, \mathbf{n}_3)$ are vertex normals of the containing triangle.

--- -->