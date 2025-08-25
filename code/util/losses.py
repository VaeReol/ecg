# Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple
import torch
import torch.nn as nn


def build_loss_fn(config: dict, device) -> Tuple[nn.Module, nn.Module]:
    loss_name = config['name']
    if loss_name == "cross_entropy":
        # weight = torch.tensor([6.0, 1.0], dtype=torch.float32, device=device)  # 假设只有两个类别，1号类别权重提高
        loss_fn = nn.CrossEntropyLoss()
        output_act = nn.Softmax(dim=-1)
    elif loss_name == "bce":
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.3], dtype=torch.float32, device=device))
        output_act = nn.Sigmoid()
    else:
        raise ValueError(f"Invalid loss name: {loss_name}")
    return loss_fn, output_act

def simclr_id_loss(z1, z2, id):
    '''
    Computes a contrastive loss for embeddings z1 and z2 based on the SimCLR framework and subject ID pairing.

    Args:
        z1 (torch.Tensor): Embeddings from view 1, shape [B, H].
        z2 (torch.Tensor): Embeddings from view 2, shape [B, H].
        id (torch.Tensor): Subject IDs corresponding to embeddings, shape [B].

    Returns:
        torch.Tensor: The computed contrastive loss.
    '''
    # Ensure all tensors are on the same device
    device = z1.device

    # Normalize embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Compute similarity matrix
    sim_matrix = torch.mm(z1, z2.T)
    sim_matrix_exp = torch.exp(sim_matrix / 0.1)

    # Convert IDs to a boolean matrix for positive pairs
    id_matrix = id.unsqueeze(1) == id.unsqueeze(0)  # Boolean matrix for matching IDs

    # Get upper and lower triangle indices
    rows1, cols1 = torch.triu_indices(id.size(0), id.size(0), offset=1, device=device)
    rows2, cols2 = torch.tril_indices(id.size(0), id.size(0), offset=-1, device=device)

    # Diagonal elements (positive pairs)
    diag_elements = torch.diag(sim_matrix_exp)
    triu_sum = sim_matrix_exp.sum(dim=1)
    tril_sum = sim_matrix_exp.sum(dim=0)

    # Loss terms for diagonal
    loss_diag1 = -torch.mean(torch.log(diag_elements / triu_sum))
    loss_diag2 = -torch.mean(torch.log(diag_elements / tril_sum))
    loss = loss_diag1 + loss_diag2
    loss_terms = 2

    # Upper triangle positive pairs
    upper_mask = id_matrix[rows1, cols1].to(device)  # Ensure mask is on the correct device
    if upper_mask.any():
        selected_rows = rows1[upper_mask]
        selected_cols = cols1[upper_mask]
        triu_elements = sim_matrix_exp[selected_rows.to(device), selected_cols.to(device)]  # Move indices to correct device
        loss_triu = -torch.mean(torch.log(triu_elements / triu_sum[selected_rows]))
        loss += loss_triu
        loss_terms += 1

    # Lower triangle positive pairs
    lower_mask = id_matrix[rows2, cols2].to(device)  # Ensure mask is on the correct device
    if lower_mask.any():
        selected_rows = rows2[lower_mask]
        selected_cols = cols2[lower_mask]
        tril_elements = sim_matrix_exp[selected_rows.to(device), selected_cols.to(device)]  # Move indices to correct device
        loss_tril = -torch.mean(torch.log(tril_elements / tril_sum[selected_cols]))
        loss += loss_tril
        loss_terms += 1

    # Final loss normalization
    return loss / loss_terms if loss_terms > 0 else 0