import os
import torch
from .prune_registry import register_prune

@register_prune('fitprune_intra_llm')
def fit_prune_intra_llm(visual_features):
    keep_radio = float(os.environ.get('KEEP_RATIO', 0.5))
    image_len = visual_features.shape[1]
    keep_num = int(image_len * keep_radio)
    image_start_idx = int(os.environ["IMAGE_START"])
    image_end_idx = int(os.environ["IMAGE_END"])

    attn_weights         = torch.load(os.environ.get('ATTENTION_WEIGHT_PATH')).cuda()
    attn_weights_headmax = attn_weights.max(dim=1).values.squeeze() # mix attention map
    # self-attention weight of visual tokens
    attn_self_attention = attn_weights_headmax[image_start_idx:image_end_idx+1, image_start_idx:image_end_idx+1] / (image_end_idx - image_start_idx + 1) 
    attn_self_attention = attn_self_attention.sum(dim=0)
    # cross-attention weight of visual tokens
    attn_cross_attention = attn_weights_headmax[image_end_idx+1:, image_start_idx:image_end_idx+1].sum(dim=0)
    # select topK to delete
    keep_idx = torch.sort(attn_cross_attention*attn_self_attention, dim=-1).indices[:keep_num]


    return visual_features[:, keep_idx, :], keep_idx