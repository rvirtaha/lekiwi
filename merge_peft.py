"""Merge PEFT LoRA adapter into base SmolVLA model and save as full model."""

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

adapter_path = Path(sys.argv[1] if len(sys.argv) > 1 else "models/cigarette_pickup_lora")
output_path = Path(sys.argv[2] if len(sys.argv) > 2 else f"{adapter_path}_merged")

# Load adapter config to get base model info
config_path = adapter_path / "config.json"
with open(config_path) as f:
    config = json.load(f)

# Strip PEFT flag so base model loads normally
config_no_peft = {**config, "use_peft": False}
tmp_config = adapter_path / "config_tmp.json"
with open(tmp_config, "w") as f:
    json.dump(config_no_peft, f, indent=4)

# Rename to trick from_pretrained into loading base weights
orig_config = adapter_path / "config_orig.json"
(adapter_path / "config.json").rename(orig_config)
tmp_config.rename(adapter_path / "config.json")

try:
    print("Loading base model (lerobot/smolvla_base)...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.eval()

    # Load LoRA adapter weights — target modules are relative to policy, not policy.model
    print(f"Loading adapter from {adapter_path}...")
    from peft import PeftModel
    peft_policy = PeftModel.from_pretrained(policy, str(adapter_path))
    peft_policy = peft_policy.merge_and_unload()
    policy = peft_policy

    print(f"Saving merged model to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(output_path))

    # Copy preprocessor/postprocessor files from adapter dir
    for f in adapter_path.glob("policy_*.json"):
        shutil.copy(f, output_path / f.name)
    for f in adapter_path.glob("policy_*.safetensors"):
        shutil.copy(f, output_path / f.name)

    # Use the adapter's config (has correct action dim) with fixes applied
    out_config = config.copy()
    out_config["use_peft"] = False
    out_config.pop("compile_model", None)
    out_config.pop("compile_mode", None)
    with open(output_path / "config.json", "w") as f:
        json.dump(out_config, f, indent=4)

    print("Done.")

finally:
    # Restore original config
    orig_config.rename(adapter_path / "config.json")
