import re
import os
from ..meta import MetaField
from ..formatters import calc_model_hash, calc_vae_hash, calc_lora_hash

# --- Existing Helpers (Restored) ---

try:
    from ..formatters import calc_clip_hash
except ImportError:
    def calc_clip_hash(name):
        return f"hash_for_{name}"

def get_model_name(node_id, obj, prompt, extra_data, outputs, input_data):
    mode = input_data[0].get("load_mode", ["full_checkpoint"])[0]
    key = "ckpt_name" if mode == "full_checkpoint" else "base_model"
    return input_data[0].get(key, [None])[0]

def get_model_hash(node_id, obj, prompt, extra_data, outputs, input_data):
    model_name = get_model_name(node_id, obj, prompt, extra_data, outputs, input_data)
    if model_name:
        return calc_model_hash(model_name)
    return None

def get_vae_name(node_id, obj, prompt, extra_data, outputs, input_data):
    if input_data[0].get("load_mode", ["full_checkpoint"])[0] == "separate_components":
        return input_data[0].get("vae_model", [None])[0]
    return None

def get_vae_hash(node_id, obj, prompt, extra_data, outputs, input_data):
    vae_name = get_vae_name(node_id, obj, prompt, extra_data, outputs, input_data)
    if vae_name:
        return calc_vae_hash(vae_name)
    return None

def get_clip_names(node_id, obj, prompt, extra_data, outputs, input_data):
    if input_data[0].get("load_mode", ["full_checkpoint"])[0] == "separate_components":
        clip_names = []
        for key in ["clip_model_1", "clip_model_2", "clip_model_3"]:
            name = input_data[0].get(key, [None])[0]
            if name and name != "None":
                clip_names.append(name)
        return clip_names if clip_names else None
    return None

def get_clip_hashes(node_id, obj, prompt, extra_data, outputs, input_data):
    names = get_clip_names(node_id, obj, prompt, extra_data, outputs, input_data)
    if names:
        return [calc_clip_hash(name) for name in names]
    return None

def get_clip_type(node_id, obj, prompt, extra_data, outputs, input_data):
    if input_data[0].get("load_mode", ["full_checkpoint"])[0] == "separate_components":
        return input_data[0].get("clip_type", [None])[0]
    return None

def get_unet_dtype(node_id, obj, prompt, extra_data, outputs, input_data):
    if input_data[0].get("load_mode", ["full_checkpoint"])[0] == "separate_components":
        return input_data[0].get("weight_dtype", [None])[0]
    return None

def get_metadata_field(field_name, node_id, obj, prompt, extra_data, outputs, input_data):
    metadata_dict = input_data[0].get("metadata", [None])[0]
    if metadata_dict and isinstance(metadata_dict, dict):
        return metadata_dict.get(field_name)
    return None

# --- Hub Node Logic ---

def _get_hub_combined_string(node_id, obj, prompt, extra_data, outputs, input_data):
    """
    Get the combined LoRA string from LoraMetadataHub.

    Primary source: the node's runtime output slot 1 (combined_loras),
    available in _resolved_node_texts after the bulk cache scan.
    Fallback: parse loras_X inputs from input_data (works when inputs are
    hardcoded strings, not links).
    """
    # Primary: retrieve the exact runtime value recorded by the hub itself.
    # This is independent of ComfyUI's execution-cache implementation.
    try:
        from nodes import NODE_CLASS_MAPPINGS
        hub_class = NODE_CLASS_MAPPINGS.get("LoraMetadataHub")
        combined = getattr(hub_class, "runtime_loras", {}).get(str(node_id))
        if combined and isinstance(combined, str):
            return combined
    except Exception:
        pass

    # Fallback: use the resolved output string from the cache
    try:
        from ...capture import _resolved_node_texts
        nid = str(node_id)
        combined = (
            _resolved_node_texts.get(f"{nid}:1")   # slot 1 = combined_loras
            or _resolved_node_texts.get(nid)
        )
        if combined and isinstance(combined, str):
            # Strip leading "None, " entries produced when some loras_X inputs are empty
            parts = [p.strip() for p in combined.split(",") if p.strip() and p.strip().lower() != "none"]
            combined = ", ".join(parts)
            if combined:
                return combined
    except Exception:
        pass

    def clean_text(value):
        if isinstance(value, str):
            value = value.strip()
            return value if value and value.lower() != "none" else None
        if isinstance(value, (list, tuple)):
            for item in value:
                cleaned = clean_text(item)
                if cleaned:
                    return cleaned
        return None

    # Fallback 1: resolved values returned by get_input_data.
    inputs = input_data[0] if input_data else {}
    parts = []
    for i in range(1, 4):
        val = clean_text(inputs.get(f"loras_{i}"))
        if val:
            parts.append(val)
    if parts:
        return ", ".join(parts)

    # Fallback 2: follow the hub's raw graph links to the runtime output slots
    # of RandomLoRA nodes.  Current ComfyUI may not expose linked optional
    # STRING inputs through get_input_data, while the source outputs are still
    # present in the metadata extension's runtime text cache.
    try:
        from ...capture import _resolved_node_texts
        hub_inputs = obj.get("inputs", {}) if isinstance(obj, dict) else {}
        for i in range(1, 4):
            raw = hub_inputs.get(f"loras_{i}")
            val = None
            if (
                isinstance(raw, (list, tuple))
                and len(raw) >= 2
                and isinstance(raw[0], (str, int))
                and isinstance(raw[1], int)
            ):
                source_id, source_slot = str(raw[0]), raw[1]
                val = clean_text(
                    _resolved_node_texts.get(f"{source_id}:{source_slot}")
                    or _resolved_node_texts.get(source_id)
                )
            else:
                val = clean_text(raw)
            if val:
                parts.append(val)
    except Exception:
        pass

    return ", ".join(parts) or None


def parse_lora_hub_data(node_id, obj, prompt, extra_data, outputs, input_data):
    """
    Parse LoRA name/strength pairs from the combined LoRA string.
    Format produced by RandomLoRAFolderModel: "path/lora.safetensors (0.85)"
    """
    combined = _get_hub_combined_string(node_id, obj, prompt, extra_data, outputs, input_data)
    if not combined:
        return []
    matches = re.findall(r"([^,]+?)\s\(([-+]?\d*\.?\d+)\)", combined)
    return [{"name": m[0].strip(), "strength": float(m[1])} for m in matches]


def get_hub_lora_names(node_id, obj, prompt, extra_data, outputs, input_data):
    data = parse_lora_hub_data(node_id, obj, prompt, extra_data, outputs, input_data)
    return [d["name"] for d in data] if data else None

def get_hub_lora_strengths(node_id, obj, prompt, extra_data, outputs, input_data):
    data = parse_lora_hub_data(node_id, obj, prompt, extra_data, outputs, input_data)
    return [d["strength"] for d in data] if data else None

def get_hub_lora_hashes(node_id, obj, prompt, extra_data, outputs, input_data):
    data = parse_lora_hub_data(node_id, obj, prompt, extra_data, outputs, input_data)
    if not data:
        return None
    hashes = []
    for d in data:
        try:
            h = calc_lora_hash(d["name"], input_data)
            hashes.append(h if h else None)
        except Exception:
            hashes.append(None)
    return hashes


# --- Mapping ---

CAPTURE_FIELD_LIST = {
    "ModelAssembler": {
        MetaField.MODEL_NAME: {"selector": get_model_name},
        MetaField.MODEL_HASH: {"selector": get_model_hash},
        MetaField.VAE_NAME: {"selector": get_vae_name},
        MetaField.VAE_HASH: {"selector": get_vae_hash},
        "Clip Model Name(s)": {"selector": get_clip_names},
        "Clip Model Hash(es)": {"selector": get_clip_hashes},
        "Clip Type": {"selector": get_clip_type},
        "UNet Weight Type": {"selector": get_unet_dtype},
    },
    "ModelAssemblerMetadata": {
        MetaField.MODEL_NAME:     {"selector": lambda *args: get_metadata_field("model_name", *args)},
        MetaField.MODEL_HASH:     {"selector": lambda *args: get_metadata_field("model_hash", *args)},
        MetaField.VAE_NAME:       {"selector": lambda *args: get_metadata_field("vae_name", *args)},
        MetaField.VAE_HASH:       {"selector": lambda *args: get_metadata_field("vae_hash", *args)},
        "Clip Model Name(s)": {"selector": lambda *args: get_metadata_field("clip_names", *args)},
        "Clip Model Hash(es)": {"selector": lambda *args: get_metadata_field("clip_hashes", *args)},
        "Clip Type":          {"selector": lambda *args: get_metadata_field("clip_type", *args)},
        "UNet Weight Type":   {"selector": lambda *args: get_metadata_field("unet_dtype", *args)},
    },
    "LoraMetadataHub": {
        MetaField.LORA_MODEL_NAME:     {"selector": get_hub_lora_names},
        MetaField.LORA_MODEL_HASH:     {"selector": get_hub_lora_hashes},
        MetaField.LORA_STRENGTH_MODEL: {"selector": get_hub_lora_strengths},
        MetaField.LORA_STRENGTH_CLIP:  {"selector": get_hub_lora_strengths},
    },
}
