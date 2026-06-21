# Copyright 2026 The RL_proj authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu


def _response_lengths(data: TensorDict) -> tuple[torch.Tensor, torch.Tensor, int]:
    prompt_ids = data["prompts"]
    response_ids = data["responses"]
    max_response_len = tu.get_non_tensor_data(data=data, key="max_response_len", default=-1)

    if prompt_ids.is_nested:
        prompt_lens = prompt_ids.offsets().diff()
        response_lens = response_ids.offsets().diff()
        if max_response_len < 0:
            max_response_len = int(response_lens.max().item())
    else:
        attention_mask = data["attention_mask"]
        prompt_lens = attention_mask[:, : prompt_ids.shape[1]].sum(dim=1)
        response_lens = attention_mask[:, prompt_ids.shape[1] :].sum(dim=1)
        max_response_len = response_ids.shape[1]

    return prompt_lens.to(torch.long), response_lens.to(torch.long), int(max_response_len)


def _slice_response_values(
    tensor: torch.Tensor,
    prompt_lens: torch.Tensor,
    response_lens: torch.Tensor,
) -> tuple[torch.Tensor, list[slice]]:
    values = tensor.values() if tensor.is_nested else tensor
    sequence_lens = (prompt_lens + response_lens).to(device=values.device)
    sequence_offsets = sequence_lens.cumsum(dim=0)
    assert int(sequence_offsets[-1].item()) == values.shape[0]

    chunks = []
    slices = []
    for resp_len, seq_offset in zip(response_lens.to(device=values.device), sequence_offsets, strict=True):
        resp_len = int(resp_len.item())
        seq_offset = int(seq_offset.item())
        start = seq_offset - resp_len - 1
        stop = seq_offset - 1
        slices.append(slice(start, stop))
        if resp_len > 0:
            chunks.append(values[start:stop])

    if chunks:
        return torch.cat(chunks, dim=0), slices

    empty_shape = (0, *values.shape[1:])
    return values.new_empty(empty_shape), slices


def _advantage_scalar(advantages: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    advantages = advantages.float()
    response_mask = response_mask.float()
    denom = response_mask.sum(dim=-1).clamp_min(1.0)
    return (advantages * response_mask).sum(dim=-1) / denom


def _scatter_response_tokens(
    flat: torch.Tensor,
    response_lens: torch.Tensor,
    max_response_len: int,
    *,
    fill_value: float = 0.0,
) -> torch.Tensor:
    batch_size = int(response_lens.shape[0])
    output = flat.new_full((batch_size, max_response_len), fill_value)
    cursor = 0
    for i, resp_len in enumerate(response_lens.tolist()):
        resp_len = int(resp_len)
        if resp_len > 0:
            output[i, :resp_len] = flat[cursor : cursor + resp_len]
            cursor += resp_len
    return output


def build_delta_weighted_advantages(
    *,
    log_probs: torch.Tensor,
    hidden_states: torch.Tensor,
    data: TensorDict,
    num_iters: int = 1,
    lam_min: float = 0.8,
    lam_max: float = 1.2,
    impl: str = "Normal",
    eps: float = 1e-8,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Build DelTA token weights and weighted advantages.

    This follows DelTA's public implementation while operating on this repo's
    no-padding NestedTensor outputs. The final tensors are padded back to the
    response layout consumed by the PPO loss.
    """

    impl_name = impl.lower().replace("-", "_")
    if impl_name not in {"normal", "memory_efficient"}:
        raise ValueError(f"Unsupported DelTA impl: {impl}")
    if lam_max < lam_min:
        raise ValueError(f"delta_lam_max must be >= delta_lam_min, got {lam_max} < {lam_min}")

    prompt_lens, response_lens, max_response_len = _response_lengths(data)
    response_mask = data["response_mask"].float()
    advantages = data["advantages"].float()
    adv_scalar = _advantage_scalar(advantages, response_mask)

    target_device = advantages.device
    log_flat, _ = _slice_response_values(log_probs, prompt_lens, response_lens)
    hidden_flat, _ = _slice_response_values(hidden_states, prompt_lens, response_lens)

    if log_flat.numel() == 0:
        zeros = response_mask.new_zeros(response_mask.shape)
        return {"delta_weights": zeros, "delta_weighted_advantages": zeros, "delta_scores": zeros}, {}

    work_device = hidden_flat.device
    response_lens_work = response_lens.to(device=work_device)
    sample_index = torch.repeat_interleave(
        torch.arange(response_lens.numel(), device=work_device),
        response_lens_work,
    )

    adv_token = adv_scalar.to(device=work_device)[sample_index]
    valid_pos = adv_token > 0
    valid_neg = adv_token < 0
    token_count = float(log_flat.numel())

    if not bool(valid_pos.any()) or not bool(valid_neg.any()):
        weights = log_flat.new_ones(log_flat.shape, dtype=torch.float32)
        weighted_adv = adv_token.float() * weights
        scores = log_flat.new_zeros(log_flat.shape, dtype=torch.float32)
    else:
        log_flat = log_flat.float()
        hidden_flat = hidden_flat.float()
        probs = torch.exp(log_flat)
        v = (1.0 - probs).unsqueeze(-1) * hidden_flat

        base_pos = torch.clamp(adv_token.float(), min=0.0)
        base_neg = torch.clamp(-adv_token.float(), min=0.0)
        s_pos = base_pos.sum().clamp_min(eps)
        s_neg = base_neg.sum().clamp_min(eps)
        mu_pos = (base_pos.unsqueeze(-1) * v).sum(dim=0) / s_pos
        mu_neg = (base_neg.unsqueeze(-1) * v).sum(dim=0) / s_neg
        d = torch.ones_like(mu_pos)

        def margins(center_pos: torch.Tensor, center_neg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            d_pos = (((v - center_pos) ** 2) / d).sum(dim=-1)
            d_neg = (((v - center_neg) ** 2) / d).sum(dim=-1)
            return d_neg - d_pos, d_pos - d_neg

        def gamma(margin: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            selected = margin[mask]
            if selected.numel() <= 1:
                return margin.new_tensor(1.0)
            return selected.float().std(unbiased=False).clamp_min(eps)

        margin_pos, margin_neg = margins(mu_pos, mu_neg)
        gamma_pos = gamma(margin_pos, valid_pos)
        gamma_neg = gamma(margin_neg, valid_neg)

        mu_pos_star = mu_pos
        mu_neg_star = mu_neg
        for _ in range(max(int(num_iters), 0)):
            raw_w_pos = torch.sigmoid(margin_pos / gamma_pos)
            raw_w_neg = torch.sigmoid(margin_neg / gamma_neg)
            w_pos = raw_w_pos * base_pos * valid_pos.float()
            w_neg = raw_w_neg * base_neg * valid_neg.float()

            den_pos = w_pos.sum().clamp_min(eps)
            den_neg = w_neg.sum().clamp_min(eps)
            mu_pos_star = (w_pos.unsqueeze(-1) * v).sum(dim=0) / den_pos
            mu_neg_star = (w_neg.unsqueeze(-1) * v).sum(dim=0) / den_neg

            margin_pos, margin_neg = margins(mu_pos_star, mu_neg_star)
            gamma_pos = gamma(margin_pos, valid_pos)
            gamma_neg = gamma(margin_neg, valid_neg)

        w_pos_final = lam_min + (lam_max - lam_min) * torch.sigmoid(margin_pos / gamma_pos)
        w_neg_final = lam_min + (lam_max - lam_min) * torch.sigmoid(margin_neg / gamma_neg)
        zero_weight = log_flat.new_full(log_flat.shape, float(lam_min))
        weights = torch.where(valid_pos, w_pos_final, torch.where(valid_neg, w_neg_final, zero_weight)).float()
        weights = weights * (weights.numel() / weights.sum().clamp_min(eps))
        weighted_adv = adv_token.float() * weights

        w = (mu_pos_star - mu_neg_star) / d
        b = -0.5 * ((mu_pos_star**2 - mu_neg_star**2) / d).sum()
        scores = ((v * w).sum(dim=-1) + b) * torch.sign(adv_token.float())

    weights_padded = _scatter_response_tokens(weights, response_lens, max_response_len).to(target_device)
    weighted_adv_padded = _scatter_response_tokens(weighted_adv, response_lens, max_response_len).to(target_device)
    scores_padded = _scatter_response_tokens(scores, response_lens, max_response_len).to(target_device)
    mask = response_mask.to(device=target_device, dtype=weights_padded.dtype)
    weights_padded = weights_padded * mask
    weighted_adv_padded = weighted_adv_padded * mask
    scores_padded = scores_padded * mask

    denom = mask.sum().clamp_min(1.0)
    metrics = {
        "weight_mean": float((weights_padded * mask).sum().item() / denom.item()),
        "weight_min": float(weights_padded[mask.bool()].min().item()) if bool(mask.bool().any()) else 0.0,
        "weight_max": float(weights_padded[mask.bool()].max().item()) if bool(mask.bool().any()) else 0.0,
        "positive_token_ratio": float(valid_pos.float().mean().item()),
        "negative_token_ratio": float(valid_neg.float().mean().item()),
        "token_count": token_count,
    }

    return (
        {
            "delta_weights": weights_padded,
            "delta_weighted_advantages": weighted_adv_padded,
            "delta_scores": scores_padded,
        },
        metrics,
    )
