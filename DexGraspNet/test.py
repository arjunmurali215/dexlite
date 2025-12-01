import numpy as np

def parse_grasp(grasp_dict):
    qpos = grasp_dict['qpos']
    scale = grasp_dict['scale']
    
    # 1. Extract Translation (Position)
    pos = np.array([qpos['WRJTx'], qpos['WRJTy'], qpos['WRJTz']])
    
    # 2. Extract Rotation (Euler Angles)
    orn_euler = np.array([qpos['WRJRx'], qpos['WRJRy'], qpos['WRJRz']])
    
    # 3. Extract Joint Angles (The order depends on your Robot's URDF/XML)
    # This is a common ordering for Shadow Hand:
    joint_names = [
        'robot0:FFJ4', 'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 
        'robot0:MFJ4', 'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 
        'robot0:RFJ4', 'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 
        'robot0:LFJ5', 'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 
        'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
    ]
    
    # Note: Some dicts might miss J4 for FF/MF if they are coupled or fixed in the model
    # We use .get() to handle potential missing keys gracefully
    joint_angles = np.array([qpos.get(name, 0.0) for name in joint_names])
    
    return pos, orn_euler, joint_angles, scale

# Example usage with the first item from your list
data = np.load('data/dataset/core-bottle-1a7ba1f4c892e2da30711cdbdbc73924.npy', allow_pickle=True)
first_grasp = data[1]
pos, orn, joints, scale = parse_grasp(first_grasp)

print(f"Grasp Position: {pos}")
print(f"Object Scale: {scale}")