import pybullet as p
import pybullet_data
import time
import os
import glob
import numpy as np
import tempfile
import re
from multiprocessing import Pool, cpu_count

class StabilityFilter:
    def __init__(self, gui=False):
        """
        Initialize the PyBullet simulation environment.
        Args:
            gui (bool): If True, shows the simulation window (good for debugging).
                        If False, runs faster in 'headless' mode.
        """
        self.gui = gui
        connection_mode = p.GUI if gui else p.DIRECT
        self.physics_client = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Simulation parameters
        self.gravity = -9.81
        self.sim_steps = 240 * 2  # Run for 2 seconds (at 240Hz)
        self.tipping_threshold_deg = 7.0  # If it tilts more than 15 deg, it's unstable
        self.movement_threshold_m = 0.05   # If it slides/rolls > 5cm, it's unstable

    def setup_scene(self):
        """Resets the simulation and adds a floor."""
        p.resetSimulation()
        p.setGravity(0, 0, self.gravity)
        # Load a standard plane (table surface)
        self.plane_id = p.loadURDF("plane.urdf")
        # Set ground friction high so objects don't slide purely due to ice-physics
        p.changeDynamics(self.plane_id, -1, lateralFriction=1.0)

    def load_object(self, urdf_path, scale=1.0):
        """
        Loads a URDF object into the simulation.
        DexGraspNet provides coacd.urdf files with convex decomposition.
        Fixes missing <mass> elements required by PyBullet.
        """
        # Read and fix the URDF (add missing <mass> element)
        with open(urdf_path, 'r') as f:
            urdf_content = f.read()
        
        # Add <mass value="0.1"/> after each <origin .../> inside <inertial> if missing
        # Pattern: <inertial>...<origin .../>...<inertia .../> (no <mass>)
        def add_mass_if_missing(match):
            inertial_block = match.group(0)
            if '<mass' not in inertial_block:
                # Insert mass after origin
                inertial_block = re.sub(
                    r'(<origin[^/]*/>\s*)',
                    r'\1<mass value="0.1"/>\n      ',
                    inertial_block
                )
            return inertial_block
        
        fixed_urdf = re.sub(
            r'<inertial>.*?</inertial>',
            add_mass_if_missing,
            urdf_content,
            flags=re.DOTALL
        )
        
        # Write fixed URDF to temp file in same directory (so mesh paths resolve)
        urdf_dir = os.path.dirname(urdf_path)
        temp_urdf_path = os.path.join(urdf_dir, '_temp_fixed.urdf')
        with open(temp_urdf_path, 'w') as f:
            f.write(fixed_urdf)
        
        # Spawn just above the table (small drop, not launched)
        start_pos = [0, 0, 1]  # 2cm above ground
        start_orn = p.getQuaternionFromEuler([0.1, 0.1, 0.1])

        try:
            # Load URDF with global scaling
            body_id = p.loadURDF(
                temp_urdf_path,
                basePosition=start_pos,
                baseOrientation=start_orn,
                globalScaling=scale,
                useFixedBase=False
            )
        finally:
            # Clean up temp file
            if os.path.exists(temp_urdf_path):
                os.remove(temp_urdf_path)
        
        # Increase friction of all links so it doesn't slide
        num_joints = p.getNumJoints(body_id)
        # Base link
        p.changeDynamics(body_id, -1, lateralFriction=1.0, rollingFriction=0.01, spinningFriction=0.01)
        # All other links
        for link_idx in range(num_joints):
            p.changeDynamics(body_id, link_idx, lateralFriction=1.0, rollingFriction=0.01, spinningFriction=0.01)
        
        return body_id

    def check_object_stability(self, obj_path):
        """
        Runs the stability test for a single object.
        Returns: (is_stable (bool), reason (str))
        """
        self.setup_scene()
        
        try:
            body_id = self.load_object(obj_path)
        except Exception as e:
            return False, f"Load Error: {str(e)}"

        # 1. Let it drop and settle (Initial Stabilization)
        # We run 100 steps to let it hit the ground and bounce
        for _ in range(100):
            p.stepSimulation()
            if self.gui: time.sleep(1./240.)

        # 2. Record "Resting" Pose
        start_pos, start_orn = p.getBasePositionAndOrientation(body_id)
        start_euler = p.getEulerFromQuaternion(start_orn)

        # 3. Check if it already fell off the world (e.g. simulation blew up)
        if start_pos[2] < -0.1:
             return False, "Fell through floor immediately"

        # 4. Run Stability Test (Wait for 2 seconds)
        for _ in range(self.sim_steps):
            p.stepSimulation()
            if self.gui: time.sleep(1./240.)

        # 5. Measure Displacement
        end_pos, end_orn = p.getBasePositionAndOrientation(body_id)
        end_euler = p.getEulerFromQuaternion(end_orn)

        # Distance moved (Euclidean distance on XY plane)
        dist_moved = np.linalg.norm(np.array(start_pos[:2]) - np.array(end_pos[:2]))
        
        # Rotation changed (Simplest check: angular difference)
        # We assume small changes, so direct Euler diff is okay for rough check
        # For rigorous check, use quaternion difference angle.
        
        # Calculate angle between start and end quaternions
        # 2 * acos(|<q1, q2>|)
        dot_prod = np.abs(np.dot(start_orn, end_orn))
        # Clamp for numerical safety
        dot_prod = min(1.0, max(-1.0, dot_prod))
        angle_diff_rad = 2 * np.arccos(dot_prod)
        angle_diff_deg = np.degrees(angle_diff_rad)

        is_stable = True
        reason = "Stable"

        if dist_moved > self.movement_threshold_m:
            is_stable = False
            reason = f"Rolled away ({dist_moved:.3f}m)"
        elif angle_diff_deg > self.tipping_threshold_deg:
            is_stable = False
            reason = f"Tipped over ({angle_diff_deg:.1f}deg)"
        elif end_pos[2] < -0.1:
            is_stable = False
            reason = "Fell off table"

        return is_stable, reason

    def cleanup(self):
        p.disconnect()


def test_single_object(obj_file):
    """
    Standalone function to test a single object.
    Creates its own PyBullet instance (for multiprocessing).
    Returns: (obj_file, is_stable, reason)
    """
    sim = StabilityFilter(gui=False)
    is_stable, reason = sim.check_object_stability(obj_file)
    sim.cleanup()
    return (obj_file, is_stable, reason)

# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    # Path to your DexGraspNet object directory
    # Example structure: data/meshdata/core-bottle-1234/coacd/decomposed.obj
    # You might need to adjust this glob pattern to match your folder structure
    OBJECT_DIR = "/home/arjun/datasets/dexgraspnet/meshdata"  # Change this to your DexGraspNet meshdata path
    object_files = glob.glob(os.path.join(OBJECT_DIR, "**/coacd.urdf"), recursive=True)

    # If list is empty, create a dummy file for testing
    if not object_files:
        print("No objects found. Please set OBJECT_DIR correctly.")
        print("Example: Checking a built-in cube/duck.")
        object_files = ["cube.urdf"] # PyBullet will fail to find this, but logic holds.

    print(f"Found {len(object_files)} objects to test.")

    # Use parallel processing for speed
    NUM_WORKERS = cpu_count()  # Use all CPU cores
    print(f"Using {NUM_WORKERS} parallel workers...")
    
    stable_objects = []
    unstable_objects = []

    with Pool(NUM_WORKERS) as pool:
        results = pool.imap_unordered(test_single_object, object_files)
        for i in range(len(object_files)):
            try:
                obj_file, is_stable, reason = next(results)
                obj_name = os.path.basename(os.path.dirname(os.path.dirname(obj_file)))  # Get object code
                if is_stable:
                    print(f"[{i+1}/{len(object_files)}] {obj_name}: PASS")
                    stable_objects.append(obj_file)
                else:
                    print(f"[{i+1}/{len(object_files)}] {obj_name}: FAIL -> {reason}")
                    unstable_objects.append(obj_file)
            except Exception as e:
                print(f"[{i+1}/{len(object_files)}] FAIL -> Exception: {str(e)}")
                unstable_objects.append(None)  # Unknown object, but count as failure

    print("\n--- RESULTS ---")
    print(f"Total Objects: {len(object_files)}")
    print(f"Stable: {len(stable_objects)}")
    print(f"Unstable: {len(unstable_objects)}")
    
    # Save results to file
    with open("stable_objects_list.txt", "w") as f:
        for obj in stable_objects:
            f.write(f"{obj}\n")
    
    # Copy stable objects to filtered_meshes directory
    import shutil
    FILTERED_DIR = "/home/arjun/datasets/dexgraspnet/filtered_meshes"
    os.makedirs(FILTERED_DIR, exist_ok=True)
    
    print(f"\nCopying {len(stable_objects)} stable objects to {FILTERED_DIR}...")
    for obj_file in stable_objects:
        # obj_file is path to coacd.urdf, go up two levels to get object folder
        obj_dir = os.path.dirname(os.path.dirname(obj_file))
        obj_name = os.path.basename(obj_dir)
        dest_dir = os.path.join(FILTERED_DIR, obj_name)
        if not os.path.exists(dest_dir):
            shutil.copytree(obj_dir, dest_dir)
    
    print(f"Done! Copied {len(os.listdir(FILTERED_DIR))} objects to {FILTERED_DIR}")