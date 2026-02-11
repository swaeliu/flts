import torch
import torch.nn.functional as F

from lora_utils import inject_lora, mark_only_lora_trainable, load_flat_lora_into_model, flatten_lora

@torch.no_grad()
def evaluate_forecast(model, loader, device):
    model.eval()
    mse_sum = mae_sum = 0.0
    n = 0
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        B, C, L = x.shape
        x_in = x.reshape(B * C, 1, L)
        m_in = m.repeat_interleave(C, 0)
        out = model(x_enc=x_in, input_mask=m_in)
        yhat = out.forecast
        #TODO: clean this 
        if yhat.ndim == 3:
            yhat = yhat[:, 0, :]
        y_true = y.reshape(B * C, -1)
        mse_sum += F.mse_loss(yhat, y_true, reduction="sum").item()
        mae_sum += F.l1_loss(yhat, y_true, reduction="sum").item()
        n += y_true.numel()
    return {"mse": mse_sum / n, "mae": mae_sum / n}

def local_train_lora_steps(
    base_model_ctor,
    base_state_dict,
    spec,
    init_lora_flat,
    train_loader,
    device,
    local_steps,
    lr,
    lora_cfg
):

    model = base_model_ctor().to(device)
    model.load_state_dict(base_state_dict, strict=True)

    #TODO: clean this up a bit. 
    inject_lora(
        model,
        r=lora_cfg["rank"],
        alpha=lora_cfg["alpha"],
        dropout=lora_cfg["dropout"],
        exclude_keywords=lora_cfg["exclude_keywords"],
    )
    mark_only_lora_trainable(model)
    load_flat_lora_into_model(model, spec, init_lora_flat)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()


    it = iter(train_loader)

    
    for step in range(local_steps):
        try:
            x, y, m = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y, m = next(it)

        x, y, m = x.to(device), y.to(device), m.to(device)
        B, C, L = x.shape
        x_in = x.reshape(B * C, 1, L)
        m_in = m.repeat_interleave(C, 0)

        out = model(x_enc=x_in, input_mask=m_in)
        yhat = out.forecast
        if yhat.ndim == 3:
            yhat = yhat[:, 0, :]
        y_true = y.reshape(B * C, -1)

        loss = F.mse_loss(yhat, y_true)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


    updated_flat = flatten_lora(model, spec, device="cpu")
    return updated_flat
