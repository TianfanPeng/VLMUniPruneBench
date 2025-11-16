import math
import os

import torch
import torch.nn.functional as F

from .prune_registry import register_prune

ALLOWED_TARGET_TOKENS = {4, 16, 25}


def _resolve_hw(seq_len, override_h=None, override_w=None):
    if override_h is not None and override_w is not None:
        if override_h * override_w != seq_len:
            raise ValueError(
                f"POOL_INPUT_HEIGHT({override_h}) * POOL_INPUT_WIDTH({override_w}) != token_len({seq_len})"
            )
        return override_h, override_w

    if override_h is not None:
        if seq_len % override_h != 0:
            raise ValueError(
                f"POOL_INPUT_HEIGHT({override_h}) does not divide token_len({seq_len})"
            )
        return override_h, seq_len // override_h

    if override_w is not None:
        if seq_len % override_w != 0:
            raise ValueError(
                f"POOL_INPUT_WIDTH({override_w}) does not divide token_len({seq_len})"
            )
        return seq_len // override_w, override_w

    root = int(math.sqrt(seq_len))
    if root * root == seq_len:
        return root, root

    for h in range(root, 0, -1):
        if seq_len % h == 0:
            return h, seq_len // h
    return seq_len, 1


def _apply_pool(feature_maps, target_side, mode):
    if mode == "avg":
        return F.adaptive_avg_pool2d(feature_maps, (target_side, target_side))
    if mode == "max":
        return F.adaptive_max_pool2d(feature_maps, (target_side, target_side))
    raise ValueError(f"Unsupported POOLING_MODE={mode}")


def _pool_tokens(tokens, target_side, mode, override_h=None, override_w=None):
    if tokens.dim() == 2:
        tokens = tokens.unsqueeze(0)
        squeeze = True
    elif tokens.dim() == 3:
        squeeze = False
    else:
        raise ValueError(f"Unexpected tensor dim={tokens.dim()} for pooling")

    batch, seq_len, hidden_dim = tokens.shape
    height, width = _resolve_hw(seq_len, override_h, override_w)
    feature_maps = tokens.permute(0, 2, 1).reshape(batch, hidden_dim, height, width)
    pooled = _apply_pool(feature_maps, target_side, mode)
    pooled = pooled.reshape(batch, hidden_dim, target_side * target_side).permute(0, 2, 1)

    if squeeze:
        return pooled[0]
    return pooled


@register_prune("pool_pre_llm")
def pool_prune_pre_llm(selected, vit_embeds):
    """
    通过 2D 自适应池化将视觉 token 压缩到固定数量 (4/16/25)。
    可选环境变量：
        POOL_TARGET_TOKENS ∈ {4,16,25}, 默认 16
        POOLING_MODE ∈ {'avg','max'}, 默认 'avg'
        POOL_INPUT_HEIGHT / POOL_INPUT_WIDTH : 当 token 数不是完美平方时指定原始网格
    """
    hidden_dim = vit_embeds.shape[-1]
    patch_num, patch_size = vit_embeds.shape[:2]

    target_tokens = int(os.environ.get("POOL_TARGET_TOKENS", 16))
    if target_tokens not in ALLOWED_TARGET_TOKENS:
        raise ValueError(f"POOL_TARGET_TOKENS must be one of {sorted(ALLOWED_TARGET_TOKENS)}")

    target_side = int(math.sqrt(target_tokens))
    pooling_mode = os.environ.get("POOLING_MODE", "avg").lower()

    override_h = os.environ.get("POOL_INPUT_HEIGHT")
    override_w = os.environ.get("POOL_INPUT_WIDTH")
    override_h = int(override_h) if override_h is not None else None
    override_w = int(override_w) if override_w is not None else None

    if override_h is not None and override_w is None and patch_size % override_h != 0:
        raise ValueError(
            f"POOL_INPUT_HEIGHT({override_h}) does not divide patch_size({patch_size})"
        )
    if override_w is not None and override_h is None and patch_size % override_w != 0:
        raise ValueError(
            f"POOL_INPUT_WIDTH({override_w}) does not divide patch_size({patch_size})"
        )
    if (
        override_h is not None
        and override_w is not None
        and override_h * override_w != patch_size
    ):
        raise ValueError(
            f"Provided POOL_INPUT_HEIGHT/POOL_INPUT_WIDTH do not match patch_size={patch_size}"
        )

    if patch_num == 1:
        flattened = vit_embeds.reshape(-1, hidden_dim)
        pooled = _pool_tokens(flattened, target_side, pooling_mode, override_h, override_w)
        vit_embeds = pooled
        keep_num = pooled.shape[0]
    else:
        base = vit_embeds[0]  # [patch_size, hidden_dim]
        pruned_base = _pool_tokens(base, target_side, pooling_mode, override_h, override_w)
        keep_num_base = pruned_base.shape[0]

        patches = vit_embeds[1:]  # [patch_num-1, patch_size, hidden_dim]
        pruned_patches = _pool_tokens(
            patches, target_side, pooling_mode, override_h, override_w
        )
        pruned_patches = pruned_patches.reshape(-1, hidden_dim)
        keep_num_patch = pruned_patches.shape[0]

        vit_embeds = torch.cat([pruned_base, pruned_patches], dim=0)
        keep_num = keep_num_base + keep_num_patch

    image_start = torch.where(selected == True)[0][0]
    selected[image_start + keep_num :] = False

    return selected, vit_embeds

