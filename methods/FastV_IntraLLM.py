import os
import torch
from .prune_registry import register_prune

@register_prune('fastv_intra_llm')
def fastv_prune_intra_llm(visual_features):
    keep_radio = float(os.environ.get('KEEP_RATIO', 0.5))
    image_len = visual_features.shape[1]
    keep_num = int(image_len * keep_radio)
    image_start_idx = int(os.environ["IMAGE_START"])
    image_end_idx = int(os.environ["IMAGE_END"])

    last_layer_attention = torch.load(os.environ.get('ATTENTION_WEIGHT_PATH')).cuda()
    last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
    last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
    last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[image_start_idx:image_end_idx+1]
    top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(keep_num).indices
    print("keep_idx: ", top_attention_rank_index.shape)
    return visual_features[:, top_attention_rank_index, :], top_attention_rank_index