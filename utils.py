import os
import torch
import tempfile
from transformers.masking_utils import eager_mask, causal_mask_function

def compute_attention_weight(
    decoder_layer,
    hidden_states,
    attention_mask,
    position_ids,
    past_key_values,
    use_cache,
    cache_position,
    position_embeddings,
    save_to_temp_file=True
):
    """Compute attention weight and optionally save to temporary file."""
    original_impl = decoder_layer.self_attn.config._attn_implementation
    decoder_layer.self_attn.config._attn_implementation = "eager"
    
    states = decoder_layer.input_layernorm(hidden_states)
    attention_mask_ = eager_mask(
        batch_size=1,
        cache_position=cache_position,
        kv_length=past_key_values[0][0].shape[-2],
        kv_offset=0,
        mask_function=causal_mask_function,
        attention_mask=attention_mask.to(torch.bool),
    )
    
    _, attention_weight = decoder_layer.self_attn.forward(
        hidden_states=states,
        attention_mask=attention_mask_,
        position_ids=position_ids,
        past_key_value=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings
    )
    
    decoder_layer.self_attn.config._attn_implementation = original_impl
    
    if save_to_temp_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            torch.save(attention_weight.detach().cpu(), tmp_file.name)
            os.environ["ATTENTION_WEIGHT_PATH"] = tmp_file.name
    
    return attention_weight
