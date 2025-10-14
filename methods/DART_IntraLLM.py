import os
import torch
from .prune_registry import register_prune

@register_prune('dart_intra_llm')
def dart_prune_intra_llm(visual_features):
    key_states = torch.load(os.environ.get('KEY_STATES_PATH')).cuda()
    hidden_states_norm = torch.load(os.environ.get('HIDDEN_STATES_NORM_PATH')).cuda()


    image_token_start_index = int(os.environ.get('IMAGE_START'))
    image_token_end_index = int(os.environ.get('IMAGE_END'))  #
    image_token_length = image_token_end_index - image_token_start_index + 1
    pivot_image_token = int(os.environ.get('PIVOT_IMAGE_TOKEN', 10))  
    pivot_text_token = int(os.environ.get('PIVOT_TEXT_TOKEN', 5))    
    keep_radio = float(os.environ.get('KEEP_RATIO', 0.5)) 
    total_pivot = pivot_image_token + pivot_text_token
    if total_pivot == 0:
        total_pivot = 1  
    TOKEN_TOPK = int(image_token_length * keep_radio / total_pivot)
    device = key_states.device
    
    key_states = key_states.permute(0, 2, 1, 3).reshape(key_states.shape[0], key_states.shape[2], -1)
    
    k_states_image_token = key_states[0][image_token_start_index:image_token_start_index + image_token_length, :]
    k_states_query_token = key_states[0][image_token_start_index + image_token_length:, :]
    
    k_states_image_token_L1_norm = torch.norm(k_states_image_token, p=1, dim=-1)
    k_states_query_token_L1_norm = torch.norm(k_states_query_token, p=1, dim=-1)
    
    image_indices = (k_states_image_token_L1_norm.topk(pivot_image_token).indices + image_token_start_index).tolist()
    query_indices = (k_states_query_token_L1_norm.topk(pivot_text_token).indices + image_token_start_index + image_token_length).tolist()
    indices_set = set(image_indices + query_indices)
    
    valid_indices = set(range(image_token_start_index, image_token_start_index + image_token_length)) - set(image_indices)
    valid_indices_list = list(valid_indices)
    
    for item in list(indices_set):
        valid_vectors = hidden_states_norm[0][valid_indices_list, :] 
        cos_sim = -torch.nn.functional.cosine_similarity(hidden_states_norm[0][item, :], valid_vectors, dim=-1)
        top_k_indices = cos_sim.topk(TOKEN_TOPK).indices
        
        top_k_real_indices = [valid_indices_list[i] for i in top_k_indices]
        indices_set.update(top_k_real_indices)
        
        valid_indices.difference_update(top_k_real_indices)
        valid_indices_list = list(valid_indices)
    indices_set.difference_update(query_indices)
    retained_image_tokens_index = torch.tensor(list(indices_set), device=device)
    keep_idx = retained_image_tokens_index - image_token_start_index
    assert torch.all(keep_idx >= 0) and torch.all(keep_idx < image_token_length), "Keep idx out of visual range!"
    keep_idx = torch.sort(keep_idx)[0]
    keep_features = visual_features[:, keep_idx, :]
    return keep_features, keep_idx
