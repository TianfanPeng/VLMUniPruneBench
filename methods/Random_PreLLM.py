import os
import torch

from .prune_registry import register_prune

@register_prune('random_pre_llm')
def random_prune_pre_llm(selected, vit_embeds):
    hidden_dim = vit_embeds.shape[-1]
    patch_num, patch_size = vit_embeds.shape[:2]
    keep_radio = float(os.environ.get('KEEP_RATIO', 0.5))
    
    if patch_num == 1:
        vit_token_num = patch_size
        keep_num = max(1, int(vit_token_num * keep_radio))
        perm = torch.randperm(vit_token_num, device=vit_embeds.device)
        selected_idx = perm[:keep_num]
        vit_embeds = vit_embeds.reshape(-1, hidden_dim)[selected_idx, :]
    else:
        base = vit_embeds[0]  # [patch_size, hidden_dim]
        patch = vit_embeds[1:].reshape(-1, hidden_dim)  # [(patch_num-1)*patch_size, hidden_dim]
        
        # Prune base
        base_token_num = base.shape[0]
        keep_num_base = max(1, int(base_token_num * keep_radio))
        perm_base = torch.randperm(base_token_num, device=base.device)
        selected_idx_base = perm_base[:keep_num_base]
        pruned_base = base[selected_idx_base, :]
        
        # Prune patch
        patch_token_num = patch.shape[0]
        keep_num_patch = max(1, int(patch_token_num * keep_radio))
        perm_patch = torch.randperm(patch_token_num, device=patch.device)
        selected_idx_patch = perm_patch[:keep_num_patch]
        pruned_patch = patch[selected_idx_patch, :]
        
        # Merge
        vit_embeds = torch.cat([pruned_base, pruned_patch], dim=0)
        keep_num = keep_num_base + keep_num_patch  # For selected adjustment
    
    image_start = torch.where(selected == True)[0][0]
    selected[image_start + keep_num:] = False
    
    return selected, vit_embeds