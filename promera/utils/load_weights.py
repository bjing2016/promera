import torch


def load_weights(path, model, load_ema=True, assign=False):
    if type(path) is not str:
        for p in path:
            load_weights(p, model, load_ema=load_ema, assign=assign)
        return
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "ema" in ckpt and load_ema:
        ckpt = ckpt["ema"]["params"]
    else:
        ckpt = ckpt["state_dict"]

    model_state_dict = model.state_dict()
    all_keys = set(list(ckpt.keys()) + list(model_state_dict.keys()))
    filtered_ckpt = {}
    for key in all_keys:
        if key not in model_state_dict:
            print(f"{key} in checkpoint but not in model state dict")
        elif key not in ckpt:
            print(f"{key} in model state dict but not in checkpoint")
        elif model_state_dict[key].shape != ckpt[key].shape:
            print(
                f"Shape mismatch for {key}: {model_state_dict[key].shape} in model state dict, {ckpt[key].shape} in checkpoint"
            )
        else:
            filtered_ckpt[key] = ckpt[key]
    # assign=True replaces the model's tensors with the checkpoint tensors in
    # place of copying into them — required when the model was built on the meta
    # device (its tensors have no storage to copy into).
    model.load_state_dict(filtered_ckpt, strict=False, assign=assign)
