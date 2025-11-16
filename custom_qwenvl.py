import os
import torch
import tempfile
from typing import Optional
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import is_torchdynamo_compiling, logging
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModelOutputWithPast
from transformers.modeling_outputs import BaseModelOutputWithPast
from layer_prune import get_layer_prune_schedule, handle_layer_prune_logic

logger = logging.get_logger(__name__)


def Qwen2_5_VLModel_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs,
):

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    if pixel_values is not None:
        image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id

        n_image_tokens = (image_mask).sum()
        image_start = torch.where(image_mask == 1)[1][0].item()
        image_end = torch.where(image_mask == 1)[1][-1].item()
        os.environ["IMAGE_START"] = str(image_start)
        os.environ["IMAGE_END"] = str(image_end)

        if os.environ.get('PRUNE_METHOD_PRE_LLM'):
            prune_method = os.environ.get('PRUNE_METHOD_PRE_LLM', 'random_pre_llm')
            keep_radio = float(os.getenv("KEEP_RADIO", "0.5"))
            keep_num = int(n_image_tokens * keep_radio)
            if prune_method == 'random_pre_llm':
                img_idx = torch.randperm(n_image_tokens)[:keep_num]
            elif prune_method == 'gprune_pre_llm':
                from methods.GPrune_PreLLM import select_important_features
                similarity = float(os.environ.get('SIMILARITY', 0.5))
                temperature = float(os.environ.get('TEMPERATURE', 0.03))
                iterate_num = int(os.environ.get('ITERATE_NUM', 5))
                img_idx = select_important_features(image_embeds, keep_radio, similarity, temperature, iterate_num)
            elif prune_method == 'divprune_pre_llm':
                from methods.DivPrune_PreLLM import divprune
                img_idx, _ = divprune(image_embeds, image_embeds.shape[0], keep_ratio=keep_radio)
        else:
            img_idx = torch.arange(n_image_tokens)
            keep_num = n_image_tokens.item()
        
        img_idx = (img_idx + image_start).to(inputs_embeds.device)

        sys_idx  = torch.arange(image_start).to(inputs_embeds.device)
        inst_idx = torch.arange(image_end + 1, inputs_embeds.shape[1]).to(inputs_embeds.device)
        keep_idx = torch.cat([sys_idx, img_idx, inst_idx]).to(inputs_embeds.device)
        keep_idx = keep_idx.sort().values

        image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        input_ids = input_ids.to(inputs_embeds.device)
        attention_mask = attention_mask.to(inputs_embeds.device)
        position_ids = position_ids.to(inputs_embeds.device)
        cache_position = cache_position.to(inputs_embeds.device)
        keep_idx = keep_idx.to(inputs_embeds.device)
        inputs_embeds = inputs_embeds[:, keep_idx, :]
        input_ids = input_ids[:, keep_idx]
        attention_mask = attention_mask[:, keep_idx]
        position_ids = position_ids[:, :, keep_idx]
        cache_position = cache_position[keep_idx]
        os.environ["IMAGE_END"] = str(image_start + keep_num - 1)         

    if pixel_values_videos is not None:
        video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0)

        if input_ids is None:
            video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            video_mask = video_mask.all(-1)
        else:
            video_mask = input_ids == self.config.video_token_id

        n_video_tokens = (video_mask).sum()
        n_video_features = video_embeds.shape[0]
        video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if not is_torchdynamo_compiling() and n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )

        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if position_ids is None:
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)


    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    output = Qwen2_5_VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
    return output if return_dict else output.to_tuple()


def Qwen2_5_VLTextModel_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
    ):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")


    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    position_ids, text_position_ids = _split_text_position_ids(position_ids)

    hidden_states = inputs_embeds

    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    is_prefilling = past_key_values is None or past_key_values.get_seq_length() == 0
    schedule = get_layer_prune_schedule()

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if schedule and is_prefilling:
            (
                hidden_states,
                attention_mask,
                position_ids,
                position_embeddings,
                cache_position,
                keep_idx,
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
                dart_forward_impl=Qwen2_5_VLAttention_forword,
            )
            if keep_idx is not None:
                text_position_ids = _reset_text_position_ids(text_position_ids, hidden_states)

        
        if attention_mask.ndim == 2:
            from transformers.modeling_attn_mask_utils import AttentionMaskConverter
            converter = AttentionMaskConverter(True)
            attention_mask = converter.to_4d(attention_mask, hidden_states.shape[1], key_value_length=hidden_states.shape[1], dtype=torch.float32)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(
            v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def _split_text_position_ids(position_ids: torch.Tensor):
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        rotary_position_ids = position_ids[1:]
    else:
        text_position_ids = position_ids[0]
        rotary_position_ids = position_ids
    return rotary_position_ids, text_position_ids


def _reset_text_position_ids(text_position_ids: torch.Tensor, hidden_states: torch.Tensor):
    batch = text_position_ids.shape[0]
    seq_len = hidden_states.shape[1]
    new_ids = torch.arange(seq_len, device=hidden_states.device, dtype=text_position_ids.dtype)
    return new_ids.unsqueeze(0).expand(batch, -1)


from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb, eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

def Qwen2_5_VLAttention_forword(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface= eager_attention_forward
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
            position_ids=position_ids,  # pass positions for FA2
            **kwargs,
        )
        
        if getattr(self, "_capture_dart_key_states", False):
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                torch.save(key_states.detach().cpu(), tmp_file.name)
                os.environ["KEY_STATES_PATH"] = tmp_file.name

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights