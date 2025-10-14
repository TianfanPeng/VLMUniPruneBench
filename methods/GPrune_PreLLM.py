import os
import torch

from .prune_registry import register_prune

def select_important_features(img_fea, keep_ratio, similarity, temperature, iterate_num):
    normalized = torch.nn.functional.normalize(img_fea, p=2, dim=1)
    A = normalized @ normalized.t()
    A[A < similarity] = 0
    A = A / temperature
    A[torch.abs(A) < 1e-6] = -5000
    A = torch.nn.functional.softmax(A, dim=1)
    degree = torch.sum(A > 0, dim=1)
    A = torch.matrix_power(A, iterate_num)
    W = torch.norm(img_fea, dim=-1).view(1, -1).to(dtype=A.dtype)
    result = W @ A
    result = result / degree
    keep_num = max(1, int(keep_ratio * img_fea.shape[0]))
    keep_idx = result.topk(keep_num).indices.flatten()
    return keep_idx

@register_prune('gprune_pre_llm')
def gprune_prune_pre_llm(selected, vit_embeds):
    hidden_dim = vit_embeds.shape[-1]
    patch_num, patch_size = vit_embeds.shape[:2]

    similarity = float(os.environ.get('SIMILARITY', 0.5))
    keep_ratio = float(os.environ.get('KEEP_RATIO', 0.5))
    temperature = float(os.environ.get('TEMPERATURE', 0.03))
    iterate_num = int(os.environ.get('ITERATE_NUM', 5))

    if patch_num == 1:
        image_feature = vit_embeds.reshape(-1, hidden_dim)
        keep_indices = select_important_features(image_feature, keep_ratio, similarity, temperature, iterate_num)
        sorted_keep_indices = torch.sort(keep_indices)[0]
        vit_embeds = image_feature[sorted_keep_indices, :]
        keep_num = sorted_keep_indices.shape[0]
    else:
        base = vit_embeds[0]  # [patch_size, hidden_dim]
        patch = vit_embeds[1:].reshape(-1, hidden_dim)  # [(patch_num-1)*patch_size, hidden_dim]

        # Prune base 
        keep_indices1 = select_important_features(base, keep_ratio, similarity, temperature, iterate_num)
        sorted_keep1 = torch.sort(keep_indices1)[0]
        pruned_base = base[sorted_keep1, :]
        keep_num_base = sorted_keep1.shape[0]

        # Prune patch 
        keep_indices2 = select_important_features(patch, keep_ratio, similarity, temperature, iterate_num)
        sorted_keep2 = torch.sort(keep_indices2)[0]
        pruned_patch = patch[sorted_keep2, :]
        keep_num_patch = sorted_keep2.shape[0]

        # Merge
        vit_embeds = torch.cat([pruned_base, pruned_patch], dim=0)
        keep_num = keep_num_base + keep_num_patch

    image_start = torch.where(selected == True)[0][0]
    selected[image_start + keep_num:] = False

    return selected, vit_embeds
