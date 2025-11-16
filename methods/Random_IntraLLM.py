import os
import torch
from .prune_registry import register_prune

@register_prune('random_intra_llm')
def random_prune_intra_llm(visual_features):
    keep_radio = float(os.environ.get('KEEP_RATIO', 0.5))
    image_len = visual_features.shape[1]
    
    keep_num = int(image_len * keep_radio)
    keep_idx = torch.randperm(image_len, device=visual_features.device)[:keep_num]
    keep_idx = torch.sort(keep_idx)[0]

    return visual_features[:, keep_idx, :], keep_idx