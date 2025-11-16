import os
import re
import tempfile
import types
from typing import Callable, Dict, Optional

import torch

from methods.prune_registry import PRUNE_FUNC
from utils import compute_attention_weight

ATTENTION_DRIVEN_METHODS = {'pdrop_intra_llm', 'fastv_intra_llm', 'fitprune_intra_llm'}
DART_METHOD = 'dart_intra_llm'
DEFAULT_GENERIC_PRUNE_LAYER = 2
DEFAULT_PDROP_LAYERS = (6, 12, 18)
_PLAN_CACHE: Optional[Dict] = None


def reset_layer_prune_schedule():
    global _PLAN_CACHE
    _PLAN_CACHE = None


def get_layer_prune_schedule() -> Optional[Dict]:
    global _PLAN_CACHE
    if _PLAN_CACHE is None:
        _PLAN_CACHE = _build_plan()
    return _PLAN_CACHE


def handle_layer_prune_logic(
    model,
    plan: Optional[Dict],
    layer_idx: int,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values,
    use_cache: bool,
    cache_position: torch.Tensor,
    position_embeddings: torch.Tensor,
    decoder_layer,
    *,
    dart_forward_impl: Optional[Callable] = None,
    capture_attention_fn: Optional[Callable] = compute_attention_weight,
):
    if not plan:
        return hidden_states, attention_mask, position_ids, position_embeddings, cache_position, None

    if layer_idx in plan['dart_layers']:
        _patch_dart_attention(decoder_layer, dart_forward_impl)

    if capture_attention_fn and layer_idx in plan['attention_layers']:
        capture_attention_fn(
            decoder_layer=decoder_layer,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            save_to_temp_file=True,
        )

    spec = plan['specs'].get(layer_idx)
    if spec is None:
        return hidden_states, attention_mask, position_ids, position_embeddings, cache_position, None

    if spec['method'] == DART_METHOD:
        _save_hidden_states_norm(decoder_layer, hidden_states)

    ctx = _extract_context(hidden_states)
    keep_features, keep_idx = _execute_prune(spec, ctx['visual'])
    (
        hidden_states,
        attention_mask,
        position_ids,
        position_embeddings,
        cache_position,
    ) = _merge_context(
        model=model,
        ctx=ctx,
        pruned_visual=keep_features,
        keep_idx=keep_idx,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )
    _cleanup_temp_files(spec['method'])
    return hidden_states, attention_mask, position_ids, position_embeddings, cache_position, keep_idx


def _build_plan():
    specs = _parse_layer_specs()
    if not specs:
        return None
    return {
        'specs': specs,
        'attention_layers': _layers_requiring(specs, ATTENTION_DRIVEN_METHODS),
        'dart_layers': _layers_requiring(specs, {DART_METHOD}),
    }


def _parse_layer_specs():
    specs: Dict[int, Dict] = {}
    plan = os.environ.get('INTRA_LLM_LAYER_SPECS', '').strip()
    default_ratio = float(os.environ.get('KEEP_RATIO', 0.5))
    if plan:
        for chunk in re.split(r"[;,]", plan):
            entry = chunk.strip()
            if not entry:
                continue
            parts = [p.strip() for p in entry.split(':')]
            if len(parts) < 2:
                raise ValueError("INTRA_LLM_LAYER_SPECS 需使用 'layer:method[:ratio]' 格式")
            try:
                layer_idx = int(parts[0])
            except ValueError as exc:
                raise ValueError(f"无法解析层编号: '{parts[0]}'") from exc
            ratio = default_ratio
            if len(parts) >= 3 and parts[2]:
                try:
                    ratio = float(parts[2])
                except ValueError as exc:
                    raise ValueError(f"无法解析 KEEP_RATIO: '{parts[2]}'") from exc
            specs[layer_idx] = {'method': parts[1], 'ratio': ratio}
    else:
        fallback = os.environ.get('PRUNE_METHOD_INTRA_LLM')
        if fallback:
            for idx in _default_layers(fallback):
                specs[idx] = {'method': fallback, 'ratio': default_ratio}
    return specs


def _default_layers(method: str):
    if method == 'pdrop_intra_llm':
        return DEFAULT_PDROP_LAYERS
    return (DEFAULT_GENERIC_PRUNE_LAYER,)


def _layers_requiring(specs: Dict[int, Dict], targets):
    layers = set()
    for idx, spec in specs.items():
        if spec['method'] in targets:
            prev = idx - 1
            if prev >= 0:
                layers.add(prev)
    return layers


def _extract_context(hidden_states: torch.Tensor):
    start = int(os.environ["IMAGE_START"])
    end = int(os.environ["IMAGE_END"])
    return {
        'start': start,
        'end': end,
        'sys': hidden_states[:, :start, :],
        'visual': hidden_states[:, start:end + 1, :],
        'inst': hidden_states[:, end + 1:, :],
    }


def _merge_context(model, ctx, pruned_visual, keep_idx, position_ids, attention_mask):
    merged = torch.cat([ctx['sys'], pruned_visual, ctx['inst']], dim=1)
    merged_len = merged.size(1)
    keep_idx = keep_idx.to(device=merged.device, dtype=torch.long)
    new_position_ids = _merge_position_ids(position_ids, ctx, keep_idx, merged_len)
    new_position_embeddings = None
    if new_position_ids is not None:
        new_position_embeddings = model.rotary_emb(merged, new_position_ids)
    new_cache_position = torch.arange(merged_len, device=merged.device, dtype=torch.int)
    new_attention_mask = _shrink_attention_mask(attention_mask, merged_len)
    os.environ["IMAGE_END"] = str(ctx['start'] + pruned_visual.shape[1] - 1)
    return merged, new_attention_mask, new_position_ids, new_position_embeddings, new_cache_position


def _merge_position_ids(position_ids, ctx, keep_idx, merged_len):
    if position_ids is None:
        return None
    start, end = ctx['start'], ctx['end']
    if position_ids.ndim == 2:
        sys_ids = position_ids[:, :start]
        inst_ids = position_ids[:, end + 1:]
        vis_ids = position_ids[:, start:end + 1][:, keep_idx]
        device = position_ids.device
        return torch.arange(merged_len, device=device, dtype=position_ids.dtype).unsqueeze(0)
    if position_ids.ndim == 3:
        sys_ids = position_ids[:, :, :start]
        inst_ids = position_ids[:, :, end + 1:]
        vis_ids = position_ids[:, :, start:end + 1][:, :, keep_idx]
        return torch.cat([sys_ids, vis_ids, inst_ids], dim=-1)
    raise ValueError("Unsupported position_ids ndim for pruning merge")


def _shrink_attention_mask(attention_mask: torch.Tensor, merged_len: int):
    if attention_mask.ndim == 2:
        return attention_mask[:, :merged_len]
    if attention_mask.ndim == 4:
        return attention_mask[:, :, :merged_len, :merged_len]
    raise ValueError("Unsupported attention_mask ndim for pruning merge")


def _save_hidden_states_norm(decoder_layer, hidden_states: torch.Tensor):
    hidden_states_norm = decoder_layer.input_layernorm(hidden_states)
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        torch.save(hidden_states_norm.detach().cpu(), tmp_file.name)
        os.environ["HIDDEN_STATES_NORM_PATH"] = tmp_file.name


def _patch_dart_attention(decoder_layer, dart_forward_impl: Optional[Callable]):
    if dart_forward_impl is None:
        return
    attn = decoder_layer.self_attn
    if getattr(attn, "_dart_forward_patched", False):
        return
    attn.forward = types.MethodType(dart_forward_impl, attn)
    attn._dart_forward_patched = True
    attn._capture_dart_key_states = True


def _cleanup_temp_files(method: str):
    keys = ["ATTENTION_WEIGHT_PATH"]
    if method != 'pdrop_intra_llm':
        keys.extend(["KEY_STATES_PATH", "HIDDEN_STATES_NORM_PATH"])
    for key in keys:
        tmp_path = os.environ.get(key)
        if tmp_path:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
            finally:
                os.environ.pop(key, None)


def _execute_prune(spec: Dict, visual_features: torch.Tensor):
    prune_func = PRUNE_FUNC[spec['method']]
    timing_enabled = os.environ.get('METHOD_TIME') == 'True'
    old_keep_ratio = None
    if spec['ratio'] is not None:
        old_keep_ratio = os.environ.get('KEEP_RATIO')
        os.environ['KEEP_RATIO'] = str(spec['ratio'])
    start_event = end_event = None
    if timing_enabled:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    try:
        keep_features, keep_idx = prune_func(visual_features)
    finally:
        if spec['ratio'] is not None:
            if old_keep_ratio is None:
                os.environ.pop('KEEP_RATIO', None)
            else:
                os.environ['KEEP_RATIO'] = old_keep_ratio
    if timing_enabled and start_event is not None and end_event is not None:
        end_event.record()
        torch.cuda.synchronize()
        ms = start_event.elapsed_time(end_event)
        with open(f"method_times_{spec['method']}.txt", "a") as f:
            f.write(f"{ms:.3f} {visual_features.shape[0]}\n")
        with open("method_times.txt", "a") as f:
            f.write(f"{ms:.3f}\n")
    if keep_idx.dim() > 1:
        keep_idx = keep_idx.reshape(-1)
    return keep_features, keep_idx

