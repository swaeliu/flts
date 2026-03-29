import math
import torch


def add_lora_to_linear(linear: torch.nn.Linear, r: int, alpha: int, dropout: float) -> None:
    device = linear.weight.device
    dtype = linear.weight.dtype

    linear.lora_r = r
    linear.lora_alpha = alpha
    linear.lora_scaling = alpha / float(r)
    linear.lora_dropout = torch.nn.Dropout(dropout)

    linear.lora_A = torch.nn.Parameter(torch.zeros(r, linear.in_features, device=device, dtype=dtype))
    linear.lora_B = torch.nn.Parameter(torch.zeros(linear.out_features, r, device=device, dtype=dtype))

    torch.nn.init.kaiming_uniform_(linear.lora_A, a=math.sqrt(5))
    torch.nn.init.zeros_(linear.lora_B)

    linear.weight.requires_grad = False
    if linear.bias is not None:
        linear.bias.requires_grad = False

    old_forward = linear.forward

    def lora_forward(x: torch.Tensor) -> torch.Tensor:
        out = old_forward(x)
        lora = (linear.lora_dropout(x) @ linear.lora_A.t()) @ linear.lora_B.t()
        return out + linear.lora_scaling * lora

    linear.forward = lora_forward


def inject_lora(model: torch.nn.Module, r: int, alpha: int, dropout: float, exclude_keywords=()) -> int:
    replaced = 0
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and not any(k in name.lower() for k in exclude_keywords):
            add_lora_to_linear(mod, r=r, alpha=alpha, dropout=dropout)
            replaced += 1
    return replaced


def mark_only_lora_trainable(model):
    for name, param in model.named_parameters():
        if ("lora_A" in name 
            or "lora_B" in name 
            or "forecast" in name):
            param.requires_grad = True
        else:
            param.requires_grad = False


def get_lora_spec_and_flatdim(model: torch.nn.Module):
    spec = []
    flatdim = 0
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            spec.append((name, tuple(p.shape), p.numel()))
            flatdim += p.numel()
    return spec, flatdim


@torch.no_grad()
def flatten_lora(model: torch.nn.Module, spec, device=None) -> torch.Tensor:
    params = dict(model.named_parameters())
    chunks = []
    for name, shape, numel in spec:
        t = params[name].detach()
        if tuple(t.shape) != tuple(shape):
            raise ValueError(
                f"[flatten_lora] Shape mismatch for {name}: spec={shape}, actual={tuple(t.shape)}"
            )
        chunks.append(t.reshape(-1))
    out = torch.cat(chunks, dim=0) if chunks else torch.empty(0)
    if device is not None:
        out = out.to(device)
    return out


@torch.no_grad()
def load_flat_lora_into_model(model: torch.nn.Module, spec, flat: torch.Tensor) -> None:
    params = dict(model.named_parameters())
    off = 0
    for name, shape, numel in spec:
        piece = flat[off: off + numel].reshape(shape).to(params[name].device)
        params[name].copy_(piece)
        off += numel
    if off != flat.numel():
        raise ValueError(f"Flat LoRA length mismatch: used {off}, provided {flat.numel()}")
