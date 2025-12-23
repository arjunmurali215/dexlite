"""
Grasp Physics Test Tool for DexGraspNet grasps in MuJoCo.

Loads grasps from filtered_grasps and tests them with physics simulation.
Uses CAPSULE primitives for hand collision (more stable than mesh-mesh).
Uses convex decomposition meshes for object collision.
"""

import mujoco
import mujoco.viewer
import numpy as np
import os
import glob
import time
from scipy.spatial.transform import Rotation as R

# ================= CONFIGURATION =================
MESH_ROOT = "/home/arjun/datasets/dexgraspnet/filtered_meshes"
GRASP_ROOT = "/home/arjun/backup/dexlite/try1/post_optimized_grasps"
VALIDATED_ROOT = "/home/arjun/backup/dexlite/try1/validated_grasps"
HAND_MESH_DIR = "/home/arjun/backup/dexlite/DexGraspNet/grasp_generation/mjcf/meshes/"
# =================================================

# Joint name mapping from dataset to MuJoCo model
JOINT_MAP = {
    'robot0:FFJ3': 'robot0:FFJ3',
    'robot0:FFJ2': 'robot0:FFJ2',
    'robot0:FFJ1': 'robot0:FFJ1',
    'robot0:FFJ0': 'robot0:FFJ0',
    'robot0:MFJ3': 'robot0:MFJ3',
    'robot0:MFJ2': 'robot0:MFJ2',
    'robot0:MFJ1': 'robot0:MFJ1',
    'robot0:MFJ0': 'robot0:MFJ0',
    'robot0:RFJ3': 'robot0:RFJ3',
    'robot0:RFJ2': 'robot0:RFJ2',
    'robot0:RFJ1': 'robot0:RFJ1',
    'robot0:RFJ0': 'robot0:RFJ0',
    'robot0:LFJ4': 'robot0:LFJ4',
    'robot0:LFJ3': 'robot0:LFJ3',
    'robot0:LFJ2': 'robot0:LFJ2',
    'robot0:LFJ1': 'robot0:LFJ1',
    'robot0:LFJ0': 'robot0:LFJ0',
    'robot0:THJ4': 'robot0:THJ4',
    'robot0:THJ3': 'robot0:THJ3',
    'robot0:THJ2': 'robot0:THJ2',
    'robot0:THJ1': 'robot0:THJ1',
    'robot0:THJ0': 'robot0:THJ0',
}


def find_convex_pieces(mesh_dir: str) -> list:
    """Find all convex piece OBJ files in the coacd directory."""
    pieces = glob.glob(os.path.join(mesh_dir, "coacd_convex_piece_*.obj"))
    return sorted(pieces)


def find_matched_objects():
    """Find objects that have both grasp files and convex mesh files."""
    npy_files = glob.glob(os.path.join(GRASP_ROOT, "*.npy"))
    matched = []
    for npy_path in npy_files:
        code = os.path.basename(npy_path).replace(".npy", "")
        mesh_dir = os.path.join(MESH_ROOT, code, "coacd")
        convex_pieces = find_convex_pieces(mesh_dir)
        # Also check for decomposed.obj for visual
        visual_mesh = os.path.join(mesh_dir, "decomposed.obj")
        if convex_pieces and os.path.exists(visual_mesh):
            matched.append({
                'code': code,
                'npy': npy_path,
                'mesh_dir': mesh_dir,
                'convex_pieces': convex_pieces,
                'visual_mesh': visual_mesh
            })
    return sorted(matched, key=lambda x: x['code'])


def build_scene_xml(obj_data: dict, scale: float, hand_pos: list, hand_quat: list) -> str:
    """
    Build a complete MuJoCo XML scene with physics enabled.
    
    Uses convex decomposition meshes for object collision.
    Uses CAPSULE primitives for hand collision (more stable than mesh-mesh).
    Hand is fixed at the grasp pose position (no freejoint).
    Object has full physics (mass, inertia, freejoint).
    
    Args:
        obj_data: Dict with mesh paths
        scale: Object scale factor
        hand_pos: [x, y, z] position of hand wrist
        hand_quat: [w, x, y, z] quaternion for hand orientation
    """
    visual_mesh = os.path.abspath(obj_data['visual_mesh'])
    convex_pieces = [os.path.abspath(p) for p in obj_data['convex_pieces']]
    
    # Build mesh assets for convex pieces
    mesh_assets = []
    for i, piece_path in enumerate(convex_pieces):
        mesh_assets.append(
            f'<mesh name="obj_collision_{i}" file="{piece_path}" scale="{scale} {scale} {scale}"/>'
        )
    # Visual mesh
    mesh_assets.append(
        f'<mesh name="obj_visual" file="{visual_mesh}" scale="{scale} {scale} {scale}"/>'
    )
    mesh_assets_str = "\n        ".join(mesh_assets)
    
    # Build collision geoms for object (all convex pieces)
    obj_collision_geoms = []
    for i in range(len(convex_pieces)):
        obj_collision_geoms.append(
            f'<geom name="obj_col_{i}" type="mesh" mesh="obj_collision_{i}" '
            f'rgba="0.2 0.8 0.3 1" contype="1" conaffinity="1"/>'
        )
    obj_collision_str = "\n            ".join(obj_collision_geoms)
    
    # Build actuators for finger joints (position control to hold grasp)
    # Higher kp values for stronger grip, with velocity damping
    actuators = []
    for joint_name in JOINT_MAP.values():
        actuators.append(
            f'<position name="act_{joint_name}" joint="{joint_name}" kp="100" kv="5"/>'
        )
    actuators_str = "\n        ".join(actuators)
    
    # Format hand position and quaternion
    hand_pos_str = f"{hand_pos[0]} {hand_pos[1]} {hand_pos[2]}"
    hand_quat_str = f"{hand_quat[0]} {hand_quat[1]} {hand_quat[2]} {hand_quat[3]}"
    
    # Use smaller capsule radii than original to avoid initial penetration
    xml = f'''<mujoco model="GraspPhysicsTest">
    <compiler angle="radian" meshdir="{HAND_MESH_DIR}"/>
    <option gravity="0 0 -9.81" timestep="0.01" iterations="30" integrator="implicitfast"/>
    <default>
        <joint limited="true" damping="1.0" armature="0.001"/>
        <geom friction="2 0.5 0.2" condim="6" margin="0.001" solimp="0.9 0.95 0.001 0.5 2" solref="0.01 1"/>
    </default>
    
    <!-- Contact offset -->
    <size njmax="2000" nconmax="1000"/>
    
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0.1 0.1 0.2" width="512" height="512"/>
        <texture name="groundtex" type="2d" builtin="checker" rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3" width="512" height="512"/>
        <material name="groundmat" texture="groundtex" texrepeat="10 10"/>
        <material name="hand_mat" rgba="0.9 0.85 0.8 1" specular="0.3" shininess="0.3"/>
        
        <!-- Hand meshes (visual only) -->
        <mesh name="robot0:palm" file="palm.obj" scale="0.001 0.001 0.001"/>
        <mesh name="robot0:knuckle" file="knuckle.obj" scale="0.001 0.001 0.001"/>
        <mesh name="robot0:F3" file="F3.obj" scale="0.001 0.001 0.001"/>
        <mesh name="robot0:F2" file="F2.obj" scale="0.001 0.001 0.001"/>
        <mesh name="robot0:F1" file="F1.obj" scale="0.001 0.001 0.001"/>
        <mesh name="robot0:lfmetacarpal" file="lfmetacarpal.obj" scale="0.001 0.001 0.001"/>
        <mesh name="robot0:TH3_z" file="TH3_z.obj" scale="0.001 0.001 0.001"/>
        <mesh name="robot0:TH2_z" file="TH2_z.obj" scale="0.001 0.001 0.001"/>
        <mesh name="robot0:TH1_z" file="TH1_z.obj" scale="0.001 0.001 0.001"/>
        
        <!-- Object meshes -->
        {mesh_assets_str}
    </asset>
    
    <worldbody>
        <light pos="0 0 2" dir="0 0 -1" diffuse="1 1 1"/>
        <light pos="1 1 1.5" dir="-0.5 -0.5 -1" diffuse="0.5 0.5 0.5"/>
        
        <!-- Ground plane -->
        <geom name="ground" type="plane" size="2 2 0.1" pos="0 0 -0.5" material="groundmat" contype="1" conaffinity="1"/>
        
        <!-- Object with physics -->
        <body name="object" pos="0 0 0">
            <freejoint name="object_joint"/>
            <inertial pos="0 0 0" mass="0.1" diaginertia="0.001 0.001 0.001"/>
            <!-- Visual mesh -->
            <geom name="obj_visual" type="mesh" mesh="obj_visual" contype="0" conaffinity="0" rgba="0.2 0.8 0.3 0.5"/>
            <!-- Collision meshes -->
            {obj_collision_str}
        </body>
        
        <!-- Mocap body to control hand position kinematically -->
        <body name="hand_mocap" pos="{hand_pos_str}" quat="{hand_quat_str}" mocap="true">
            <geom type="box" size="0.01 0.01 0.01" rgba="1 0 0 0" contype="0" conaffinity="0"/>
        </body>
        
        <!-- Hand at grasp position (uses CAPSULE collision primitives) -->
        <body name="robot0:palm" pos="{hand_pos_str}" quat="{hand_quat_str}">
            <freejoint name="hand_root"/>
            <inertial pos="0 0 0.04" mass="1.0" diaginertia="0.001 0.001 0.001"/>
            <geom name="palm_visual" type="mesh" mesh="robot0:palm" material="hand_mat" contype="0" conaffinity="0"/>
            <geom name="palm_col0" type="box" size="0.028 0.009 0.045" pos="0.011 0 0.038" contype="1" conaffinity="1" rgba="0.8 0.7 0.6 0.3"/>
            <geom name="palm_col1" type="box" size="0.009 0.009 0.022" pos="-0.032 0 0.014" contype="1" conaffinity="1" rgba="0.8 0.7 0.6 0.3"/>
            
            <!-- First Finger -->
            <body name="robot0:ffknuckle" pos="0.033 0 0.095">
                <inertial pos="0 0 0" mass="0.008" diaginertia="1e-05 1e-05 1e-05"/>
                <joint name="robot0:FFJ3" axis="0 1 0" range="-0.349 0.349"/>
                <geom type="mesh" mesh="robot0:knuckle" material="hand_mat" contype="0" conaffinity="0"/>
                <body name="robot0:ffproximal">
                    <inertial pos="0 0 0.02" mass="0.014" diaginertia="1e-05 1e-05 1e-05"/>
                    <joint name="robot0:FFJ2" axis="1 0 0" range="0 1.571"/>
                    <geom type="mesh" mesh="robot0:F3" material="hand_mat" contype="0" conaffinity="0"/>
                    <geom name="ff_prox_col" type="capsule" size="0.010 0.020" pos="0 0 0.0225" contype="1" conaffinity="1" rgba="0.8 0.7 0.6 0.3"/>
                    <body name="robot0:ffmiddle" pos="0 0 0.045">
                        <inertial pos="0 0 0.012" mass="0.012" diaginertia="1e-05 1e-05 1e-05"/>
                        <joint name="robot0:FFJ1" axis="1 0 0" range="0 1.571"/>
                        <geom type="mesh" mesh="robot0:F2" material="hand_mat" contype="0" conaffinity="0"/>
                        <geom name="ff_mid_col" type="capsule" size="0.008 0.011" pos="0 0 0.0125" contype="1" conaffinity="1" rgba="0.8 0.7 0.6 0.3"/>
                        <body name="robot0:ffdistal" pos="0 0 0.025">
                            <inertial pos="0 0 0.015" mass="0.01" diaginertia="1e-05 1e-05 1e-05"/>
                            <joint name="robot0:FFJ0" axis="1 0 0" range="0 1.571"/>
                            <geom type="mesh" mesh="robot0:F1" material="hand_mat" contype="0" conaffinity="0"/>
                            <geom name="ff_dist_col" type="capsule" size="0.008 0.010" pos="0 0 0.012" contype="1" conaffinity="1" rgba="0.8 0.7 0.6 0.3"/>
                        </body>
                    </body>
                </body>
            </body>
            
            <!-- Middle Finger -->
            <body name="robot0:mfknuckle" pos="0.011 0 0.099">
                <inertial pos="0 0 0" mass="0.008" diaginertia="1e-05 1e-05 1e-05"/>
                <joint name="robot0:MFJ3" axis="0 1 0" range="-0.349 0.349"/>
                <geom type="mesh" mesh="robot0:knuckle" material="hand_mat" contype="0" conaffinity="0"/>
                <body name="robot0:mfproximal">
                    <inertial pos="0 0 0.02" mass="0.014" diaginertia="1e-05 1e-05 1e-05"/>
                    <joint name="robot0:MFJ2" axis="1 0 0" range="0 1.571"/>
                    <geom type="mesh" mesh="robot0:F3" material="hand_mat" contype="0" conaffinity="0"/>
                    <geom name="mf_prox_col" type="capsule" size="0.010 0.020" pos="0 0 0.0225" contype="1" conaffinity="1" rgba="0.8 0.7 0.6 0.3"/>
                    <body name="robot0:mfmiddle" pos="0 0 0.045">
                        <inertial pos="0 0 0.012" mass="0.012" diaginertia="1e-05 1e-05 1e-05"/>
                        <joint name="robot0:MFJ1" axis="1 0 0" range="0 1.571"/>
                        <geom type="mesh" mesh="robot0:F2" material="hand_mat" contype="0" conaffinity="0"/>
                        <geom name="mf_mid_col" type="capsule" size="0.008 0.011" pos="0 0 0.0125" contype="1" conaffinity="1" rgba="0.8 0.7 0.6 0.3"/>
                        <body name="robot0:mfdistal" pos="0 0 0.025">
                            <inertial pos="0 0 0.015" mass="0.01" diaginertia="1e-05 1e-05 1e-05"/>
                            <joint name="robot0:MFJ0" axis="1 0 0" range="0 1.571"/>
                            <geom type="mesh" mesh="robot0:F1" material="hand_mat" contype="0" conaffinity="0"/>
                            <geom name="mf_dist_col" type="capsule" size="0.008 0.010" pos="0 0 0.012" contype="1" conaffinity="1" rgba="0.8 0.7 0.6 0.3"/>
                        </body>
                    </body>
                </body>
            </body>
            
            <!-- Ring Finger -->
            <body name="robot0:rfknuckle" pos="-0.011 0 0.095">
                <inertial pos="0 0 0" mass="0.008" diaginertia="1e-05 1e-05 1e-05"/>
                <joint name="robot0:RFJ3" axis="0 1 0" range="-0.349 0.349"/>
                <geom type="mesh" mesh="robot0:knuckle" material="hand_mat" contype="0" conaffinity="0"/>
                <body name="robot0:rfproximal">
                    <inertial pos="0 0 0.02" mass="0.014" diaginertia="1e-05 1e-05 1e-05"/>
                    <joint name="robot0:RFJ2" axis="1 0 0" range="0 1.571"/>
                    <geom type="mesh" mesh="robot0:F3" material="hand_mat" contype="0" conaffinity="0"/>
                    <geom name="rf_prox_col" type="capsule" size="0.010 0.020" pos="0 0 0.0225" contype="1" conaffinity="1" rgba="0.8 0.7 0.6 0.3"/>
                    <body name="robot0:rfmiddle" pos="0 0 0.045">
                        <inertial pos="0 0 0.012" mass="0.012" diaginertia="1e-05 1e-05 1e-05"/>
                        <joint name="robot0:RFJ1" axis="1 0 0" range="0 1.571"/>
                        <geom type="mesh" mesh="robot0:F2" material="hand_mat" contype="0" conaffinity="0"/>
                        <geom name="rf_mid_col" type="capsule" size="0.008 0.011" pos="0 0 0.0125" contype="1" conaffinity="1" rgba="0.8 0.7 0.6 0.3"/>
                        <body name="robot0:rfdistal" pos="0 0 0.025">
                            <inertial pos="0 0 0.015" mass="0.01" diaginertia="1e-05 1e-05 1e-05"/>
                            <joint name="robot0:RFJ0" axis="1 0 0" range="0 1.571"/>
                            <geom type="mesh" mesh="robot0:F1" material="hand_mat" contype="0" conaffinity="0"/>
                            <geom name="rf_dist_col" type="capsule" size="0.008 0.010" pos="0 0 0.012" contype="1" conaffinity="1" rgba="0.8 0.7 0.6 0.3"/>
                        </body>
                    </body>
                </body>
            </body>
            
            <!-- Little Finger -->
            <body name="robot0:lfmetacarpal" pos="-0.017 0 0.044">
                <inertial pos="-0.01 0 0.01" mass="0.02" diaginertia="1e-05 1e-05 1e-05"/>
                <joint name="robot0:LFJ4" axis="0.570977 0 0.820966" range="0 0.785"/>
                <geom type="mesh" mesh="robot0:lfmetacarpal" pos="-0.016 0 -0.023" material="hand_mat" contype="0" conaffinity="0"/>
                <geom name="lf_meta_col" type="box" size="0.008 0.009 0.022" pos="-0.0165 0 0.01" contype="1" conaffinity="1" rgba="0.8 0.7 0.6 0.3"/>
                <body name="robot0:lfknuckle" pos="-0.017 0 0.044">
                    <inertial pos="0 0 0" mass="0.008" diaginertia="1e-05 1e-05 1e-05"/>
                    <joint name="robot0:LFJ3" axis="0 1 0" range="-0.349 0.349"/>
                    <geom type="mesh" mesh="robot0:knuckle" material="hand_mat" contype="0" conaffinity="0"/>
                    <body name="robot0:lfproximal">
                        <inertial pos="0 0 0.02" mass="0.014" diaginertia="1e-05 1e-05 1e-05"/>
                        <joint name="robot0:LFJ2" axis="1 0 0" range="0 1.571"/>
                        <geom type="mesh" mesh="robot0:F3" material="hand_mat" contype="0" conaffinity="0"/>
                        <geom name="lf_prox_col" type="capsule" size="0.010 0.020" pos="0 0 0.0225" contype="1" conaffinity="1" rgba="0.8 0.7 0.6 0.3"/>
                        <body name="robot0:lfmiddle" pos="0 0 0.045">
                            <inertial pos="0 0 0.012" mass="0.012" diaginertia="1e-05 1e-05 1e-05"/>
                            <joint name="robot0:LFJ1" axis="1 0 0" range="0 1.571"/>
                            <geom type="mesh" mesh="robot0:F2" material="hand_mat" contype="0" conaffinity="0"/>
                            <geom name="lf_mid_col" type="capsule" size="0.008 0.011" pos="0 0 0.0125" contype="1" conaffinity="1" rgba="0.8 0.7 0.6 0.3"/>
                            <body name="robot0:lfdistal" pos="0 0 0.025">
                                <inertial pos="0 0 0.015" mass="0.01" diaginertia="1e-05 1e-05 1e-05"/>
                                <joint name="robot0:LFJ0" axis="1 0 0" range="0 1.571"/>
                                <geom type="mesh" mesh="robot0:F1" material="hand_mat" contype="0" conaffinity="0"/>
                                <geom name="lf_dist_col" type="capsule" size="0.008 0.010" pos="0 0 0.012" contype="1" conaffinity="1" rgba="0.8 0.7 0.6 0.3"/>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
            
            <!-- Thumb -->
            <body name="robot0:thbase" pos="0.034 -0.009 0.029" quat="0.923956 0 0.382499 0">
                <inertial pos="0 0 0" mass="0.01" diaginertia="1e-05 1e-05 1e-05"/>
                <joint name="robot0:THJ4" axis="0 0 -1" range="-1.047 1.047"/>
                <body name="robot0:thproximal">
                    <inertial pos="0 0 0.015" mass="0.02" diaginertia="1e-05 1e-05 1e-05"/>
                    <joint name="robot0:THJ3" axis="1 0 0" range="0 1.222"/>
                    <geom type="mesh" mesh="robot0:TH3_z" material="hand_mat" contype="0" conaffinity="0"/>
                    <geom name="th_prox_col" type="capsule" size="0.012 0.016" pos="0 0 0.019" contype="1" conaffinity="1" rgba="0.8 0.7 0.6 0.3"/>
                    <body name="robot0:thhub" pos="0 0 0.038">
                        <inertial pos="0 0 0" mass="0.005" diaginertia="1e-05 1e-05 1e-05"/>
                        <joint name="robot0:THJ2" axis="1 0 0" range="-0.209 0.209"/>
                        <body name="robot0:thmiddle">
                            <inertial pos="0 0 0.015" mass="0.02" diaginertia="1e-05 1e-05 1e-05"/>
                            <joint name="robot0:THJ1" axis="0 1 0" range="-0.524 0.524"/>
                            <geom type="mesh" mesh="robot0:TH2_z" material="hand_mat" contype="0" conaffinity="0"/>
                            <geom name="th_mid_col" type="capsule" size="0.011 0.014" pos="0 0 0.016" contype="1" conaffinity="1" rgba="0.8 0.7 0.6 0.3"/>
                            <body name="robot0:thdistal" pos="0 0 0.032">
                                <inertial pos="0 0 0.015" mass="0.02" diaginertia="1e-05 1e-05 1e-05"/>
                                <joint name="robot0:THJ0" axis="0 1 0" range="-1.571 0"/>
                                <geom type="mesh" mesh="robot0:TH1_z" material="hand_mat" contype="0" conaffinity="0"/>
                                <geom name="th_dist_col" type="capsule" size="0.009 0.011" pos="0 0 0.013" contype="1" conaffinity="1" rgba="0.8 0.7 0.6 0.3"/>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
    
    <equality>
        <!-- Weld hand to mocap body - hand will follow mocap position -->
        <weld body1="hand_mocap" body2="robot0:palm" solref="0.001 1" solimp="0.9 0.95 0.001"/>
    </equality>
    
    <actuator>
        {actuators_str}
    </actuator>
</mujoco>'''
    
    return xml


def extract_hand_pose(grasp: dict):
    """
    Extract hand wrist position and orientation from grasp data.
    
    Returns:
        hand_pos: [x, y, z] position
        hand_quat: [w, x, y, z] quaternion (MuJoCo format)
    """
    qpos = grasp['qpos']
    
    # Extract wrist translation
    tx = qpos.get('WRJTx', 0.0)
    ty = qpos.get('WRJTy', 0.0)
    tz = qpos.get('WRJTz', 0.0)
    
    # Extract wrist rotation (euler angles in radians)
    rx = qpos.get('WRJRx', 0.0)
    ry = qpos.get('WRJRy', 0.0)
    rz = qpos.get('WRJRz', 0.0)
    
    # Convert euler angles to quaternion
    rot = R.from_euler('xyz', [rx, ry, rz])
    quat = rot.as_quat()  # Returns [x, y, z, w]
    # MuJoCo uses [w, x, y, z] format
    quat_mujoco = [quat[3], quat[0], quat[1], quat[2]]
    
    return [tx, ty, tz], quat_mujoco


# Grasp tightening bias (radians) - close fingers slightly more
GRASP_TIGHTEN_BIAS = 0.07  # ~4 degrees


def apply_grasp_pose(model: mujoco.MjModel, data: mujoco.MjData, grasp: dict):
    """
    Apply grasp finger joint angles and set actuator targets.
    
    The hand wrist is already positioned in the XML, so we only set finger joints here.
    Applies a small tightening bias to close fingers slightly more than original grasp.
    
    Returns the target joint positions for the actuators.
    """
    qpos = grasp['qpos']
    
    # Flexion joints (close inward) - these get positive bias
    flexion_joints = ['FFJ2', 'FFJ1', 'FFJ0', 'MFJ2', 'MFJ1', 'MFJ0', 
                      'RFJ2', 'RFJ1', 'RFJ0', 'LFJ2', 'LFJ1', 'LFJ0',
                      'THJ3', 'THJ1']
    
    # Set finger joint angles and collect actuator targets
    actuator_targets = {}
    for dataset_name, model_name in JOINT_MAP.items():
        if dataset_name in qpos:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, model_name)
            if joint_id != -1:
                joint_adr = model.jnt_qposadr[joint_id]
                joint_val = qpos[dataset_name]
                
                # Apply tightening bias to flexion joints
                joint_suffix = dataset_name.split(':')[-1]
                if joint_suffix in flexion_joints:
                    # Get joint limits
                    jnt_range = model.jnt_range[joint_id]
                    # Add bias but clamp to joint limits
                    joint_val = min(joint_val + GRASP_TIGHTEN_BIAS, jnt_range[1])
                
                data.qpos[joint_adr] = joint_val
                actuator_targets[model_name] = joint_val
    
    return actuator_targets


def set_actuator_targets(model: mujoco.MjModel, data: mujoco.MjData, targets: dict):
    """Set actuator control values to maintain grasp pose."""
    for joint_name, target in targets.items():
        act_name = f"act_{joint_name}"
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        if act_id != -1:
            data.ctrl[act_id] = target


class GraspBrowser:
    def __init__(self):
        self.objects = find_matched_objects()
        if not self.objects:
            raise ValueError("No matching objects found! Check MESH_ROOT and GRASP_ROOT paths.")
        
        print(f"Found {len(self.objects)} objects with matching meshes and grasps.")
        
        self.current_obj_idx = 0
        self.current_grasp_idx = 0
        self.grasps_cache = None
    
    def load_grasps(self):
        """Load grasps for current object."""
        obj = self.objects[self.current_obj_idx]
        if self.grasps_cache is None:
            self.grasps_cache = np.load(obj['npy'], allow_pickle=True)
        return self.grasps_cache
    
    def get_current_info(self) -> str:
        """Get info string for current state."""
        obj = self.objects[self.current_obj_idx]
        grasps = self.load_grasps()
        return f"Object [{self.current_obj_idx + 1}/{len(self.objects)}]: {obj['code']} | Grasp [{self.current_grasp_idx + 1}/{len(grasps)}]"
    
    def run(self):
        """Main physics simulation loop."""
        print("\n" + "=" * 70)
        print("DexGraspNet Grasp Physics Tester")
        print("=" * 70)
        print("Controls (in terminal after closing viewer):")
        print("  n - next grasp")
        print("  p - previous grasp")
        print("  N - next object")
        print("  P - previous object")
        print("  g <num> - go to grasp number")
        print("  o <num> - go to object number")
        print("  q - quit")
        print("=" * 70)
        print("In viewer: Watch if the hand maintains grasp under gravity!")
        print("=" * 70 + "\n")
        
        while True:
            # Load current grasp
            grasps = self.load_grasps()
            grasp = grasps[self.current_grasp_idx]
            obj = self.objects[self.current_obj_idx]
            
            print(f"\n{self.get_current_info()}")
            print(f"Scale: {grasp['scale']:.4f}")
            print(f"Convex pieces: {len(obj['convex_pieces'])}")
            
            # Extract hand pose from grasp data
            hand_pos, hand_quat = extract_hand_pose(grasp)
            
            # Build scene XML with hand at grasp position
            try:
                xml = build_scene_xml(obj, grasp['scale'], hand_pos, hand_quat)
                model = mujoco.MjModel.from_xml_string(xml)
                data = mujoco.MjData(model)
            except Exception as e:
                print(f"Error loading scene: {e}")
                import traceback
                traceback.print_exc()
                print("Skipping to next object...")
                self.current_obj_idx = (self.current_obj_idx + 1) % len(self.objects)
                self.current_grasp_idx = 0
                self.grasps_cache = None
                continue
            
            # Apply finger joint angles and get actuator targets
            actuator_targets = apply_grasp_pose(model, data, grasp)
            
            # Set actuator controls to maintain grasp
            set_actuator_targets(model, data, actuator_targets)
            
            # Forward kinematics to update positions
            mujoco.mj_forward(model, data)
            
            # Track object and hand position for grasp success evaluation
            obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
            hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0:palm")
            initial_obj_pos = data.xpos[obj_body_id].copy()
            initial_hand_pos = data.xpos[hand_body_id].copy()
            
            # Launch viewer with physics simulation
            print("Running physics simulation with lift test... Close viewer to continue.")
            
            with mujoco.viewer.launch_passive(model, data) as viewer:
                # Set viewer options
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
                
                sim_time = 0.0
                start_time = time.time()
                step_count = 0
                control_ratio = 4  # 100Hz sim / 25Hz control
                
                # Lift parameters
                lift_distance = 1.0  # meters
                lift_time = 2.0  # seconds
                stabilize_time = 1.0  # seconds
                velocity = lift_distance / lift_time
                
                stabilize_steps = int(stabilize_time / model.opt.timestep)
                lift_start_step = stabilize_steps
                lift_end_step = stabilize_steps + int(lift_time / model.opt.timestep)
                
                # Get mocap body ID to control hand position
                mocap_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_mocap")
                mocap_idx = model.body_mocapid[mocap_body_id]
                
                while viewer.is_running():
                    # Control at 25Hz (every 4 sim steps)
                    if step_count % control_ratio == 0:
                        set_actuator_targets(model, data, actuator_targets)
                    
                    # Lift phase: move mocap body upward - hand will follow via weld constraint
                    if lift_start_step <= step_count < lift_end_step:
                        data.mocap_pos[mocap_idx][2] += velocity * model.opt.timestep
                    
                    # Step simulation at 100Hz
                    mujoco.mj_step(model, data)
                    sim_time += model.opt.timestep
                    step_count += 1
                    
                    viewer.sync()
                    
                    # Real-time sync
                    elapsed = time.time() - start_time
                    if sim_time > elapsed:
                        time.sleep(sim_time - elapsed)
            
            # Evaluate grasp after simulation with lift test
            final_obj_pos = data.xpos[obj_body_id].copy()
            final_hand_pos = data.xpos[hand_body_id].copy()
            
            obj_lift = final_obj_pos[2] - initial_obj_pos[2]
            hand_lift = final_hand_pos[2] - initial_hand_pos[2]
            lift_ratio = obj_lift / hand_lift if hand_lift > 0.01 else 0
            
            if lift_ratio > 0.7:
                print(f"✓ Grasp SUCCESS - Object lifted {obj_lift:.3f}m (hand: {hand_lift:.3f}m, ratio: {lift_ratio:.1%})")
            else:
                print(f"✗ Grasp FAILED - Object lifted {obj_lift:.3f}m (hand: {hand_lift:.3f}m, ratio: {lift_ratio:.1%})")
            
            # Get user input
            try:
                cmd = input("\nCommand: ").strip()
            except EOFError:
                break
            
            if not cmd:
                continue
            elif cmd == 'q':
                print("Exiting...")
                break
            elif cmd == 'n':
                self.current_grasp_idx = (self.current_grasp_idx + 1) % len(grasps)
            elif cmd == 'p':
                self.current_grasp_idx = (self.current_grasp_idx - 1) % len(grasps)
            elif cmd == 'N':
                self.current_obj_idx = (self.current_obj_idx + 1) % len(self.objects)
                self.current_grasp_idx = 0
                self.grasps_cache = None
            elif cmd == 'P':
                self.current_obj_idx = (self.current_obj_idx - 1) % len(self.objects)
                self.current_grasp_idx = 0
                self.grasps_cache = None
            elif cmd.startswith('g '):
                try:
                    idx = int(cmd[2:]) - 1
                    if 0 <= idx < len(grasps):
                        self.current_grasp_idx = idx
                    else:
                        print(f"Invalid grasp number. Range: 1-{len(grasps)}")
                except ValueError:
                    print("Invalid number")
            elif cmd.startswith('o '):
                try:
                    idx = int(cmd[2:]) - 1
                    if 0 <= idx < len(self.objects):
                        self.current_obj_idx = idx
                        self.current_grasp_idx = 0
                        self.grasps_cache = None
                    else:
                        print(f"Invalid object number. Range: 1-{len(self.objects)}")
                except ValueError:
                    print("Invalid number")
            else:
                print("Unknown command. Use n/p/N/P/g <num>/o <num>/q")


def headless_test(num_grasps: int = 0, save_validated: bool = True, lift_test: bool = True, lift_distance: float = 1.0):
    """
    Run headless physics test on grasps with lift test.
    
    Args:
        num_grasps: Number of grasps to test. If 0, test ALL grasps across ALL objects.
        save_validated: If True, save passing grasps to VALIDATED_ROOT after each object.
        lift_test: If True, move hand upward and check if object follows
        lift_distance: Distance to lift (meters)
    """
    objects = find_matched_objects()
    if not objects:
        print("No objects found!")
        return
    
    # Create validated output directory
    if save_validated:
        os.makedirs(VALIDATED_ROOT, exist_ok=True)
    
    test_all = (num_grasps == 0)
    
    print(f"\n{'='*70}")
    print("HEADLESS GRASP PHYSICS TEST")
    print(f"{'='*70}")
    if test_all:
        print(f"Testing ALL grasps across {len(objects)} objects...")
    else:
        print(f"Testing up to {num_grasps} grasps across {len(objects)} objects...")
    if save_validated:
        print(f"Saving validated grasps to: {VALIDATED_ROOT}")
    print(f"{'='*70}")
    
    total_success = 0
    total_tested = 0
    total_saved = 0
    remaining_grasps = num_grasps if not test_all else float('inf')
    obj_idx = 0
    start_time = time.time()
    
    while (test_all or remaining_grasps > 0) and obj_idx < len(objects):
        obj = objects[obj_idx]
        grasps = np.load(obj['npy'], allow_pickle=True)
        
        if test_all:
            grasps_to_test = len(grasps)
        else:
            grasps_to_test = min(int(remaining_grasps), len(grasps))
        
        obj_success = 0
        obj_start = time.time()
        passed_grasps = []  # Collect passing grasps for this object
        
        for i in range(grasps_to_test):
            grasp = grasps[i]
            hand_pos, hand_quat = extract_hand_pose(grasp)
            scale = grasp['scale']
            
            try:
                xml = build_scene_xml(obj, scale, hand_pos, hand_quat)
                model = mujoco.MjModel.from_xml_string(xml)
                data = mujoco.MjData(model)
            except Exception as e:
                print(f"ERROR building scene for grasp {i+1}: {e}")
                total_tested += 1
                continue
            
            # Apply grasp
            actuator_targets = apply_grasp_pose(model, data, grasp)
            set_actuator_targets(model, data, actuator_targets)
            mujoco.mj_forward(model, data)
            
            obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
            hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0:palm")
            
            initial_obj_pos = data.xpos[obj_body_id].copy()
            initial_hand_pos = data.xpos[hand_body_id].copy()
            
            if lift_test:
                try:
                    # Phase 1: Stabilize grasp (0.5 seconds)
                    for _ in range(50):
                        set_actuator_targets(model, data, actuator_targets)
                        mujoco.mj_step(model, data)
                    
                    # Phase 2: Lift hand upward at constant velocity
                    # Velocity = lift_distance / time
                    lift_time = 2.0  # seconds
                    velocity = lift_distance / lift_time
                    steps = int(lift_time / model.opt.timestep)
                    
                    # Get mocap body ID to control hand position
                    mocap_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_mocap")
                    
                    if mocap_body_id == -1:
                        raise ValueError("hand_mocap body not found in model")
                    
                    # Find mocap index (mocap_pos is indexed by mocap index, not body ID)
                    # model.body_mocapid[body_id] gives the mocap index (-1 if not mocap)
                    mocap_idx = model.body_mocapid[mocap_body_id]
                    if mocap_idx == -1:
                        raise ValueError(f"Body {mocap_body_id} is not a mocap body")
                    
                    for step in range(steps):
                        # Move mocap body upward - hand will follow via weld constraint
                        data.mocap_pos[mocap_idx][2] += velocity * model.opt.timestep
                        
                        set_actuator_targets(model, data, actuator_targets)
                        mujoco.mj_step(model, data)
                    
                    final_obj_pos = data.xpos[obj_body_id].copy()
                    final_hand_pos = data.xpos[hand_body_id].copy()
                    
                    # Check if object lifted with hand
                    obj_lift = final_obj_pos[2] - initial_obj_pos[2]
                    hand_lift = final_hand_pos[2] - initial_hand_pos[2]
                    
                    # Debug output for all grasps
                    print(f"\nDEBUG - Grasp {i+1} (obj {obj_idx+1}):")
                    print(f"  Initial: hand_z={initial_hand_pos[2]:.4f}, obj_z={initial_obj_pos[2]:.4f}")
                    print(f"  Final:   hand_z={final_hand_pos[2]:.4f}, obj_z={final_obj_pos[2]:.4f}")
                    print(f"  Lift:    hand={hand_lift:.4f}m, obj={obj_lift:.4f}m, ratio={obj_lift/hand_lift if abs(hand_lift)>0.001 else 0:.2%}")
                    
                    # Success if object lifted at least 70% of what hand lifted
                    success = obj_lift > 0.7 * hand_lift
                    drop = hand_lift - obj_lift
                except Exception as e:
                    print(f"ERROR in lift test for grasp {i+1}: {e}")
                    total_tested += 1
                    continue
            else:
                # Original test: just simulate and check if object stays in place
                for _ in range(200):
                    set_actuator_targets(model, data, actuator_targets)
                    mujoco.mj_step(model, data)
                
                final_obj_pos = data.xpos[obj_body_id].copy()
                drop = np.linalg.norm(final_obj_pos - initial_obj_pos)
                success = drop < 0.05
            
            if success:
                total_success += 1
                obj_success += 1
                passed_grasps.append(grasp)
            total_tested += 1
        
        # Save validated grasps for this object
        if save_validated and len(passed_grasps) > 0:
            out_path = os.path.join(VALIDATED_ROOT, f"{obj['code']}.npy")
            np.save(out_path, np.array(passed_grasps, dtype=object))
            total_saved += len(passed_grasps)
        
        obj_time = time.time() - obj_start
        rate = grasps_to_test / obj_time if obj_time > 0 else 0
        pct = 100 * obj_success / grasps_to_test if grasps_to_test > 0 else 0
        saved_str = f" | saved {len(passed_grasps)}" if save_validated else ""
        print(f"[{obj_idx+1:3d}/{len(objects)}] {obj['code'][:40]:40s} | {obj_success:3d}/{grasps_to_test:3d} ({pct:5.1f}%) | {rate:.1f} g/s{saved_str}")
        
        if not test_all:
            remaining_grasps -= grasps_to_test
        obj_idx += 1
    
    total_time = time.time() - start_time
    success_rate = 100 * total_success / total_tested if total_tested > 0 else 0
    print(f"\n{'='*70}")
    print(f"OVERALL: {total_success}/{total_tested} ({success_rate:.1f}%) in {total_time:.1f}s ({total_tested/total_time:.1f} grasps/sec)")
    if save_validated:
        print(f"SAVED: {total_saved} validated grasps to {VALIDATED_ROOT}")
    print(f"{'='*70}")


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Test grasps in MuJoCo with physics')
    parser.add_argument('--headless', action='store_true', help='Run headless test')
    parser.add_argument('--num', type=int, default=0, help='Number of grasps to test (0 = all)')
    parser.add_argument('--no-lift', action='store_true', help='Disable lift test (original static test)')
    parser.add_argument('--lift-distance', type=float, default=1.0, help='Distance to lift in meters')
    parser.add_argument('--no-save', action='store_true', help='Do not save validated grasps')
    
    args = parser.parse_args()
    
    if args.headless or '--headless' in sys.argv:
        headless_test(
            num_grasps=args.num,
            save_validated=not args.no_save,
            lift_test=not args.no_lift,
            lift_distance=args.lift_distance
        )
    else:
        browser = GraspBrowser()
        browser.run()
