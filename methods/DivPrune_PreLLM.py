
import os
import torch

from .prune_registry import register_prune

def pairwise_cosine_similarity(matrix):
    norm_matrix = matrix / matrix.norm(dim=1, keepdim=True)
    cosine_similarity = torch.mm(norm_matrix, norm_matrix.t())
    return cosine_similarity


def divprune(
    visual_feature_vectors,
    image_feature_length,
    cosine_matrix=None,
    keep_ratio=0.5,
):
    keep_num = max(1, round(keep_ratio * image_feature_length))
    if cosine_matrix is None:
        cosine_matrix = 1.0 - (pairwise_cosine_similarity(visual_feature_vectors))

    s = torch.empty(
        keep_num, dtype=torch.long, device=visual_feature_vectors.device
    )
    for i in range(keep_num):
        if i == 0:
            m2 = cosine_matrix
        else:
            m2 = torch.index_select(
                cosine_matrix,
                0,
                torch.index_select(
                    s, 0, torch.arange(0, i, device=cosine_matrix.device)
                ),
            )

        if i == 0:
            scores = torch.topk(m2, 2, dim=0, largest=False).values[
                1, :
            ]  # for distance
        else:
            scores = torch.min(m2, dim=0).values  # for distance

        phrase_to_add_idx = torch.argmax(scores)
        s[i] = phrase_to_add_idx
    return s, cosine_matrix


@register_prune('divprune_pre_llm')
def divprune_prune_pre_llm(selected, vit_embeds):
    hidden_dim = vit_embeds.shape[-1]
    patch_num, patch_size = vit_embeds.shape[:2]
    
    keep_ratio = float(os.environ.get('KEEP_RATIO', 0.5))
    
    if patch_num == 1:
        image_feature = vit_embeds.reshape(-1, hidden_dim)
        keep_indices, _ = divprune(image_feature, image_feature.shape[0], keep_ratio=keep_ratio)
        sorted_keep_indices = torch.sort(keep_indices)[0]
        vit_embeds = image_feature[sorted_keep_indices, :]
        keep_num = sorted_keep_indices.shape[0]
    else:
        base = vit_embeds[0]  # [patch_size, hidden_dim]
        patch = vit_embeds[1:].reshape(-1, hidden_dim)  # [(patch_num-1)*patch_size, hidden_dim]
        
        # Prune base 
        keep_indices1, _ = divprune(base, base.shape[0], keep_ratio=keep_ratio)
        sorted_keep1 = torch.sort(keep_indices1)[0]
        pruned_base = base[sorted_keep1, :]
        keep_num_base = sorted_keep1.shape[0]
        
        # Prune patch 
        keep_indices2, _ = divprune(patch, patch.shape[0], keep_ratio=keep_ratio)
        sorted_keep2 = torch.sort(keep_indices2)[0]
        pruned_patch = patch[sorted_keep2, :]
        keep_num_patch = sorted_keep2.shape[0]
        
        # Merge
        vit_embeds = torch.cat([pruned_base, pruned_patch], dim=0)
        keep_num = keep_num_base + keep_num_patch
    
    image_start = torch.where(selected == True)[0][0]
    selected[image_start + keep_num:] = False
    
    return selected, vit_embeds
