import torch
from torch_scatter import scatter, scatter_min
from torch_sparse import SparseTensor
import math


def calculate_vectors(pos_j, pos_i):
        
    vecs = pos_i - pos_j
    dists = (vecs**2).sum(1).sqrt()
    #assert torch.all(dists!=0), 'data bug'
    
    vec_units = vecs / dists[..., None]
    return vecs, dists, vec_units
    
    
def vector_rejection(a, b):
    
    a_x_b = torch.sum(a * b, dim=-1)
    b_x_b = torch.sum(b * b, dim=-1)
    return a - (a_x_b / b_x_b)[:, None] * b
    

def calculate_angles(a, b):
    
    x = torch.sum(a * b, dim=-1)  
    y = torch.cross(a, b, dim=-1).norm(dim=-1)  
    y = torch.max(y, torch.tensor(1e-9))  
    angle = torch.atan2(y, x)
    return angle  


def dihedral_angle(a, b, c):
    
    plane_a_x_b = torch.cross(a, b, dim=-1)
    plane_b_x_c = torch.cross(b, c, dim=-1)
    x = torch.sum(plane_a_x_b * plane_b_x_c, dim=-1)
    y = b.norm(dim=-1) * torch.sum(a * plane_b_x_c, dim=-1)
    angle = torch.atan2(y, x)
    angle[angle < 0] += 2 * math.pi
    return angle
    
    
def calculate_geo_intra(edge_index, pos):
    
    # Modify the code from: 
    # https://github.com/divelab/DIG/blob/dig-stable/dig/threedgraph/method/comenet/comenet.py
    
    num_nodes = pos.size(0)
    
    j, i = edge_index

    vecs, dist, _ = calculate_vectors(pos[j], pos[i])
    max_dist = dist.max()
    
    _, argmin0 = scatter_min(dist, i, dim_size=num_nodes) 
    argmin0[argmin0 >= len(i)] = 0 
    n0 = j[argmin0]   
    
    add = torch.zeros_like(dist).to(dist.device)
    add[argmin0] = max_dist
    dist1 = dist + add
    _, argmin1 = scatter_min(dist1, i, dim_size=num_nodes)
    argmin1[argmin1 >= len(i)] = 0
    n1 = j[argmin1] 
    
    add = torch.zeros_like(dist).to(dist.device)
    add[argmin1] = max_dist
    dist2 = dist1 + add
    _, argmin2 = scatter_min(dist2, i, dim_size=num_nodes)
    argmin2[argmin2 >= len(i)] = 0
    n2 = j[argmin2]
    
    n0_j = n0[j]
    n1_j = n1[j]
    n2_j = n2[j]
    # ref0, ref1 for i
    n0_i = n0[i]
    n1_i = n1[i]

    # tau: (iref, i, j, jref)
    # when compute tau, do not use n0, n0_j as ref for i and j,
    # because if n0 = j, or n0_j = i, the computed tau is zero
    # so if n0 = j, we choose iref = n1
    # if n0_j = i, we choose jref = n1_j
    mask_iref = n0_i == j
    iref = torch.clone(n0_i)
    iref[mask_iref] = n1_i[mask_iref]
    idx_iref = argmin0[i] 
    idx_iref[mask_iref] = argmin1[i][mask_iref]

    mask_jref = n0_j == i
    jref = torch.clone(n0_j)
    jref[mask_jref] = n1_j[mask_jref]
    idx_jref = argmin0[j]
    idx_jref1 = argmin1[j]
    idx_jref[mask_jref] = idx_jref1[mask_jref]
    idx_jref2 = argmin2[j]
    mask_jref2 = torch.logical_and(mask_jref, n2_j != i)
    idx_jref1[mask_jref2] = idx_jref2[mask_jref2]
    mask_jref1 = n1_j == i
    idx_jref1[mask_jref1] = idx_jref2[mask_jref1]
    
    
    pos_ji, pos_jref, pos_jref1, pos_iref = (
        vecs,
        vecs[idx_jref],
        vecs[idx_jref1],
        vecs[idx_iref]
    )

    # Calculate angles.
    theta = calculate_angles(pos_ji, pos_jref)
    
    # Calculate improper angles
    phi = dihedral_angle(pos_jref, pos_ji, pos_jref1)
    
    # Calculate torsion angles
    tau = dihedral_angle(pos_jref, pos_ji, pos_iref)
    
    # revision
    angles_mask = j[idx_jref] == i 
    phi_mask = j[idx_jref] == j[idx_jref1]
    tau_mask = j[idx_iref] == j
    #theta[angles_mask] = 0.0
    phi[angles_mask] = 0.0
    phi[phi_mask] = 0.0
    tau[angles_mask] = 0.0
    tau[tau_mask] = 0.0
       
    return vecs, dist, theta, phi, tau, argmin0, argmin1

    
def calculate_geo_inter(vecs_l, argmin_l, argmin1_l, 
                        vecs_p, argmin_p, argmin1_p,
                        pos_l, pos_p, edge_index_l2p,
                        edge_index_p2l):
    
    
    j_l2p, i_l2p = edge_index_l2p
    j_p2l, i_p2l = edge_index_p2l
    
    vecs_l2p, dist_l2p, _ = calculate_vectors(pos_l[j_l2p], pos_p[i_l2p])
    vecs_p2l, dist_p2l, _ = calculate_vectors(pos_p[j_p2l], pos_l[i_p2l])
          
    # calculate l2p
    jref_l2p = argmin_l[j_l2p]
    jref1_l2p = argmin1_l[j_l2p]
    iref_l2p = argmin_p[i_l2p]
    
    phi_l2p_mask = jref_l2p == jref1_l2p
    
    vec_jref_l2p, vec_jref1_l2p, vec_iref_l2p = (vecs_l[jref_l2p], vecs_l[jref1_l2p], vecs_p[iref_l2p]) 
    
     # Calculate angles.
    theta_l2p = calculate_angles(vec_jref_l2p, vecs_l2p)
    
    phi_l2p = dihedral_angle(vec_jref_l2p, vecs_l2p, vec_jref1_l2p)
    phi_l2p[phi_l2p_mask] = 0.0
    
    # Calculate torsion angles
    tau_l2p = dihedral_angle(vec_jref_l2p, vecs_l2p, vec_iref_l2p)
     
    # calculate p2l
    jref_p2l = argmin_p[j_p2l]
    jref1_p2l = argmin1_p[j_p2l]
    iref_p2l = argmin_l[i_p2l]
    
    phi_p2l_mask = jref_p2l == jref1_p2l
    
    vec_jref_p2l, vec_jref1_p2l, vec_iref_p2l = (vecs_p[jref_p2l], vecs_p[jref1_p2l], vecs_l[iref_p2l]) 
    
     # Calculate angles.
    theta_p2l = calculate_angles(vec_jref_p2l, vecs_p2l)
    
    # Calculate improper angles
    phi_p2l = dihedral_angle(vec_jref_p2l, vecs_p2l, vec_jref1_p2l)
    phi_p2l[phi_p2l_mask] = 0.0
    
    # Calculate torsion angles
    tau_p2l = dihedral_angle(vec_jref_p2l, vecs_p2l, vec_iref_p2l)
    
    
    return dist_l2p, theta_l2p, phi_l2p, tau_l2p, dist_p2l, theta_p2l, phi_p2l, tau_p2l




