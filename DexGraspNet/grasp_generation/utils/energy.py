"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: energy functions
"""

import torch


def cal_energy(hand_model, object_model, w_dis=100.0, w_pen=100.0, 
               w_spen=10.0, w_joints=1.0, w_table=1.0, verbose=False):
    
    # E_dis
    batch_size, n_contact, _ = hand_model.contact_points.shape
    device = object_model.device
    distance, contact_normal = object_model.cal_distance(hand_model.contact_points)
    E_dis = torch.sum(distance.abs(), dim=-1, dtype=torch.float).to(device)

    # E_fc
    contact_normal = contact_normal.reshape(batch_size, 1, 3 * n_contact)
    transformation_matrix = torch.tensor([[0, 0, 0, 0, 0, -1, 0, 1, 0],
                                          [0, 0, 1, 0, 0, 0, -1, 0, 0],
                                          [0, -1, 0, 1, 0, 0, 0, 0, 0]],
                                         dtype=torch.float, device=device)
    g = torch.cat([torch.eye(3, dtype=torch.float, device=device).expand(batch_size, n_contact, 3, 3).reshape(batch_size, 3 * n_contact, 3),
                   (hand_model.contact_points @ transformation_matrix).view(batch_size, 3 * n_contact, 3)], 
                  dim=2).float().to(device)
    norm = torch.norm(contact_normal @ g, dim=[1, 2])
    E_fc = norm * norm

    # E_joints
    E_joints = torch.sum((hand_model.hand_pose[:, 9:] > hand_model.joints_upper) * (hand_model.hand_pose[:, 9:] - hand_model.joints_upper), dim=-1) + \
        torch.sum((hand_model.hand_pose[:, 9:] < hand_model.joints_lower) * (hand_model.joints_lower - hand_model.hand_pose[:, 9:]), dim=-1)

    # E_pen
    object_scale = object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
    object_surface_points = object_model.surface_points_tensor * object_scale  # (n_objects * batch_size_each, num_samples, 3)
    distances = hand_model.cal_distance(object_surface_points)
    distances[distances <= 0] = 0
    E_pen = distances.sum(-1)

    # E_spen
    E_spen = hand_model.self_penetration()

    # E_table: penalize ANY part of hand mesh below table (VECTORIZED)
    # Pre-compute table_z for all batch items
    batch_size_each = object_model.batch_size_each
    n_objects = len(object_model.object_mesh_list)
    
    # Get mesh min z values for each object (on CPU, these are trimesh objects)
    mesh_min_z = torch.tensor(
        [object_model.object_mesh_list[i].bounds[0, 2] for i in range(n_objects)],
        device=device, dtype=torch.float
    )  # (n_objects,)
    
    # Expand to match batch structure: obj 0 repeated batch_size_each times, then obj 1, etc.
    mesh_min_z_expanded = mesh_min_z.repeat_interleave(batch_size_each)  # (batch_size,)
    
    # Get scales flattened in same order
    scales_flat = object_model.object_scale_tensor.flatten()  # (n_objects * batch_size_each,) = (batch_size,)
    
    # Compute table_z
    table_z = mesh_min_z_expanded * scales_flat  # (batch_size,)
    
    # Vectorized FK transform for all vertices across all links
    # Collect all link vertices and their transforms
    all_link_verts = []
    
    for link_name in hand_model.mesh:
        link_verts = hand_model.mesh[link_name]['vertices']  # (n_verts, 3)
        n_verts = link_verts.shape[0]
        
        # Get transform matrix: (B, 4, 4)
        link_transform = hand_model.current_status[link_name].get_matrix()
        if link_transform.shape[0] == 1 and batch_size > 1:
            link_transform = link_transform.expand(batch_size, -1, -1)
        
        # Expand vertices to batch: (B, n_verts, 3)
        link_verts_batch = link_verts.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Convert to homogeneous coords: (B, n_verts, 4)
        ones = torch.ones(batch_size, n_verts, 1, device=device)
        link_verts_homo = torch.cat([link_verts_batch, ones], dim=2)
        
        # Apply link transform: (B, n_verts, 4) @ (B, 4, 4)^T -> (B, n_verts, 4)
        transformed = torch.bmm(link_verts_homo, link_transform.transpose(1, 2))[:, :, :3]  # (B, n_verts, 3)
        
        all_link_verts.append(transformed)
    
    # Concatenate all vertices: (B, total_verts, 3)
    all_verts = torch.cat(all_link_verts, dim=1)  # (B, ~31000, 3)
 
    all_verts = torch.bmm(all_verts, hand_model.global_rotation.transpose(1, 2)) + hand_model.global_translation.unsqueeze(1)

    penetration_depth = table_z.unsqueeze(1) - all_verts[:, :, 2]  # (B, ~31000), positive = below table
    penetration_depth = torch.clamp(penetration_depth, min=0)  # only penalize if below
    
    # Smooth penalty: sum of squared penetration depths (differentiable!)
    # Scale factor chosen to make E_table comparable to other energy terms (~0-100 range)
    E_table = (penetration_depth ** 2).sum(dim=1)
    
    total = E_fc + w_dis*E_dis + w_pen*E_pen + w_spen*E_spen + w_joints*E_joints + w_table*E_table  
    
    if verbose:
        return total, E_fc, E_dis, E_pen, E_spen, E_joints, E_table
    else:
        return total
