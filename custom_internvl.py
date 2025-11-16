import os
import torch
import tempfile
from typing import Optional
from transformers import GenerationConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.models.qwen2.modeling_qwen2 import BaseModelOutputWithPast, apply_rotary_pos_emb, eager_attention_forward
from transformers.utils import TransformersKwargs
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.masking_utils import create_causal_mask
from methods.prune_registry import PRUNE_FUNC
from layer_prune import get_layer_prune_schedule, handle_layer_prune_logic

@torch.no_grad()
def InternVLChatModel_generate(
    self,
    pixel_values: Optional[torch.FloatTensor] = None,
    input_ids: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    visual_features: Optional[torch.FloatTensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    output_hidden_states: Optional[bool] = None,
    **generate_kwargs,
) -> torch.LongTensor:
    assert self.img_context_token_id is not None
    if pixel_values is not None:
        if visual_features is not None:
            vit_embeds = visual_features
        else:
            vit_embeds = self.extract_feature(pixel_values)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        assert selected.sum() != 0
        image_start = torch.where(selected)[0][0]
        os.environ["IMAGE_START"] = str(image_start.cpu().item())
        image_end = torch.where(selected)[0][-1]
        os.environ["IMAGE_END"] = str(image_end.cpu().item())
        sys_embeds = input_embeds[:image_start, :]
        inst_embeds = input_embeds[image_end + 1:, :]
        if os.environ.get('PRUNE_METHOD_PRE_LLM'):
            
            prune_method = os.environ.get('PRUNE_METHOD_PRE_LLM', 'random_pre_llm')
            
            if os.environ.get('METHOD_TIME') == 'True':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event   = torch.cuda.Event(enable_timing=True)
                start_event.record()

                prune_func = PRUNE_FUNC[prune_method]
                end_event.record()
                torch.cuda.synchronize()
                ms = start_event.elapsed_time(end_event)
                method_name = os.environ.get('PRUNE_METHOD_PRE_LLM')
                with open(f"method_times_{method_name}.txt", "a") as f:
                    f.write(f"{ms:.3f} {vit_embeds.shape[0]}\n")
            else:
                prune_func = PRUNE_FUNC[prune_method]
            print(f"prune_func: {prune_func}, before prune: {vit_embeds.shape}")
            selected, vit_embeds = prune_func(selected, vit_embeds)
            print(f"prune_func: {prune_func}, after prune: {vit_embeds.shape}")
        image_end = torch.where(selected)[0][-1]
        os.environ["IMAGE_END"] = str(image_end.cpu().item())
        print(input_embeds.device)
        vit_embeds = vit_embeds.reshape(-1, C).to(input_embeds.device)
        input_embeds = torch.cat([sys_embeds, vit_embeds, inst_embeds], dim=0)
        input_embeds = input_embeds.reshape(-1, C).to(input_embeds.device)
        input_embeds = input_embeds.reshape(B, -1, C)
    else:
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
    outputs = self.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        generation_config=generation_config,
        output_hidden_states=output_hidden_states,
        use_cache=True,
        **generate_kwargs,
    )
    return outputs




def Qwen2Model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()
    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)
    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    is_prefilling = past_key_values is None or past_key_values.get_seq_length() == 0
    if is_prefilling:
        attention_mask = attention_mask[:, :hidden_states.shape[-2]]

    if is_prefilling and os.environ.get('PREFILL_TIME') == 'True':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event   = torch.cuda.Event(enable_timing=True)
        start_event.record()

    print("before prune: ", hidden_states.shape)
    schedule = get_layer_prune_schedule()
    for decoder_layer in self.layers[:self.config.num_hidden_layers]:
        if schedule and is_prefilling:
            (
                hidden_states,
                attention_mask,
                position_ids,
                position_embeddings,
                cache_position,
                _,
            ) = handle_layer_prune_logic(
                self,
                schedule,
                decoder_layer.self_attn.layer_idx,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
                cache_position,
                position_embeddings,
                decoder_layer,
                dart_forward_impl=Qwen2Attention_forward,
            )
        if hidden_states.shape[1] == 1:
            layer_idx = decoder_layer.self_attn.layer_idx
            kv_len = past_key_values[layer_idx][0].shape[-2] + 1
            attention_mask = attention_mask[:, :kv_len]
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    if is_prefilling and os.environ.get('PREFILL_TIME') == 'True':
        end_event.record()
        torch.cuda.synchronize()
        ms = start_event.elapsed_time(end_event)
        with open("prefill_times.txt", "a") as f:
            f.write(f"{ms:.3f}\n")
    print("after prune: ", hidden_states.shape)
    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values if use_cache else None)

def Qwen2Attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    if getattr(self, "_capture_dart_key_states", False):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            torch.save(key_states.detach().cpu(), tmp_file.name)
            os.environ["KEY_STATES_PATH"] = tmp_file.name
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
