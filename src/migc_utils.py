import safetensors.torch


def load_migc_from_safetensors(unet, adapter_path: str):
    state_dict = safetensors.torch.load_file(adapter_path)
    for name, module in unet.named_modules():
        if hasattr(module, "migc"):
            print(f"Found MIGC in {name}")

            # Get the state dict with the incorrect keys
            state_dict_to_load = {k: v for k, v in state_dict.items() if k.startswith(name)}

            # Create a new state dict, removing the "attn2." prefix from each key
            new_state_dict = {k.replace(f"{name}.migc.", "", 1): v for k, v in state_dict_to_load.items()}

            # Load the corrected state dict
            module.migc.load_state_dict(new_state_dict)
            module.to(device=unet.device, dtype=unet.dtype)
