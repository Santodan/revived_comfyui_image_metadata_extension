import json
import os
import re
from collections import defaultdict
from . import hook
from .defs.captures import CAPTURE_FIELD_LIST
from .defs.meta import MetaField
from .defs.formatters import calc_lora_hash, calc_model_hash, extract_embedding_names, extract_embedding_hashes
from .utils.log import print_warning

from nodes import NODE_CLASS_MAPPINGS
from .trace import Trace
from execution import get_input_data
from comfy_execution.graph import DynamicPrompt


class OutputCacheCompat:
    """Handles cache access across ComfyUI versions.
    Uses get_output_cache() in version 0.3.67 and newer, get() in 0.3.66 and lower.
    """
    def __init__(self, cache):
        self._cache = cache

    def get_output_cache(self, input_unique_id, unique_id=None):
        if hasattr(self._cache, "get"):
            return self._cache.get(input_unique_id)
        return getattr(self._cache, "outputs", {}).get(input_unique_id, None)

    def get(self, input_unique_id):
        if hasattr(self._cache, "get"):
            return self._cache.get(input_unique_id)
        return getattr(self._cache, "outputs", {}).get(input_unique_id, None)

    def get_cache(self, input_unique_id, unique_id=None):
        if hasattr(self._cache, "get_cache"):
            return self._cache.get_cache(input_unique_id, unique_id)
        return self.get_output_cache(input_unique_id, unique_id)


# ---------------------------------------------------------------------------
# Helpers to walk the prompt graph and extract raw text regardless of how
# many indirection levels (wired text nodes, concat nodes, etc.) there are.
# ---------------------------------------------------------------------------

# Node class names that are text-concatenation / joining nodes.
_CONCAT_CLASS_HINTS = [
    "concat", "join", "combine", "mixer",
    "string", "text",        # many "TextConcatenate", "StringJoin" nodes
]

# Input key names that carry text payloads inside concat-style nodes.
_TEXT_KEY_HINTS = [
    "text", "string", "input", "value", "prompt",
    "text1", "text2", "text_a", "text_b",
    "string1", "string2",
    "positive_prompt", "negative_prompt",
]


def _is_link(value):
    """Return True when *value* looks like a ComfyUI node-output link [node_id, index]."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    )


def _resolve_text_from_graph(value, prompt, outputs, _visited=None, batch_index=0):
    """
    Recursively resolve *value* to a plain string by walking the prompt graph.

    *value* can be:
      - A plain string  → returned as-is.
      - A link          → follow to the source node and recurse.
      - None            → returns None.

    *batch_index* selects which entry to use when a cache slot holds a list
    of strings (i.e. when a list was fed into the node, generating one image
    per entry).  Pass the current image's position in the batch so each image
    gets its own prompt text rather than always the first one.

    The function tries (in order):
      1. The execution cache (already-evaluated output).
      2. A ``text`` / ``string`` / similar field on the source node's inputs
         (handles CLIPTextEncode with a wired-in text node).
      3. Concatenation / joining nodes whose text inputs are all resolved
         recursively and joined with the node's separator.

    *_visited* prevents infinite loops on cyclic graphs.
    """
    if _visited is None:
        _visited = set()

    if value is None:
        return None

    # Already a plain string – nothing to resolve.
    if isinstance(value, str):
        return value if value.strip() else None

    # Unwrap single-element lists that ComfyUI sometimes produces.
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _resolve_text_from_graph(value[0], prompt, outputs, _visited, batch_index)

    if not _is_link(value):
        return None

    node_id = str(value[0])
    out_idx = value[1]

    if node_id in _visited:
        return None
    _visited = _visited | {node_id}

    # ── 1. Try the execution cache first ────────────────────────────────────
    if outputs is not None:
        cached = outputs.get(node_id)
        if cached is not None:
            # cached is a list of output-slot values
            if isinstance(cached, (list, tuple)) and len(cached) > out_idx:
                slot = cached[out_idx]
                # When the slot itself is a list, each entry corresponds to one
                # image in the batch — pick the right one via batch_index.
                if isinstance(slot, list):
                    idx = min(batch_index, len(slot) - 1)
                    slot = slot[idx]
                if isinstance(slot, str) and slot.strip():
                    return slot
                # slot might itself be a link
                resolved = _resolve_text_from_graph(slot, prompt, outputs, _visited, batch_index)
                if resolved:
                    return resolved

    # ── 2. Walk the graph node ───────────────────────────────────────────────
    node = prompt.get(node_id)
    if node is None:
        return None

    node_inputs = node.get("inputs", {})
    class_type = node.get("class_type", "").lower()

    # Direct text field on this node (e.g. CLIPTextEncode whose "text" is
    # a hard-coded string, or a primitive String node).
    for key in ("text", "string", "value", "val", "prompt",
                "positive_prompt", "negative_prompt"):
        raw = node_inputs.get(key)
        if raw is None:
            continue
        if isinstance(raw, str) and raw.strip():
            return raw
        if _is_link(raw):
            resolved = _resolve_text_from_graph(raw, prompt, outputs, _visited)
            if resolved:
                return resolved

    # ── 3. Concatenation / joining nodes ────────────────────────────────────
    is_concat = any(hint in class_type for hint in _CONCAT_CLASS_HINTS)
    if is_concat:
        # Collect all text-like input keys in stable order.
        candidate_keys = sorted(
            (k for k in node_inputs if any(h in k.lower() for h in _TEXT_KEY_HINTS)),
            key=lambda k: (re.sub(r'\d+', '', k),
                           int(re.search(r'\d+', k).group()) if re.search(r'\d+', k) else 0)
        )
        parts = []
        for k in candidate_keys:
            resolved = _resolve_text_from_graph(node_inputs[k], prompt, outputs, _visited, batch_index)
            if resolved:
                parts.append(resolved)

        if parts:
            sep_raw = node_inputs.get("delimiter", node_inputs.get("separator", " "))
            sep = sep_raw.replace("\\n", "\n") if isinstance(sep_raw, str) else " "
            return sep.join(parts)

    # ── 4. Fallback: scan only text-hinted input keys, never model/clip/vae ──
    _NON_TEXT_KEYS = {"model", "clip", "vae", "control_net", "image", "mask",
                      "latent", "latent_image", "samples", "upscale_model",
                      "positive", "negative", "conditioning"}
    for key, raw in node_inputs.items():
        if key.lower() in _NON_TEXT_KEYS:
            continue
        if _is_link(raw):
            # Only follow if the key name hints at text content
            if any(h in key.lower() for h in _TEXT_KEY_HINTS):
                resolved = _resolve_text_from_graph(raw, prompt, outputs, _visited, batch_index)
                if resolved:
                    return resolved

    return None


def _resolve_clip_text_encode_prompt(node_id, prompt, outputs, batch_index=0):
    """
    Given a CLIPTextEncode node's *node_id*, return its resolved text string.

    The CLIPTextEncode node has a single ``text`` input which may be:
      - A hard-coded string.
      - A link to another node (primitive, text node, concat node, …).
      - A list of strings when a list was wired in (one entry per batch image).
    """
    node = prompt.get(str(node_id))
    if node is None:
        return None
    raw = node.get("inputs", {}).get("text")
    if raw is None:
        return None
    # Hard-coded string directly in the node
    if isinstance(raw, str):
        return raw if raw.strip() else None
    # List of strings wired directly (rare but possible)
    if isinstance(raw, list) and raw and isinstance(raw[0], str):
        idx = min(batch_index, len(raw) - 1)
        return raw[idx] if raw[idx].strip() else None
    return _resolve_text_from_graph(raw, prompt, outputs, batch_index=batch_index)


def _follow_conditioning_to_clip_text(cond_value, prompt, outputs, _depth=0, batch_index=0):
    """
    Follow a conditioning link chain until we reach a CLIPTextEncode and
    resolve its text.

    *batch_index* is forwarded all the way down so that when the text source
    is a list (one string per batch image), the correct entry is selected.
    """
    if _depth > 20:  # safety limit
        return None
    if not _is_link(cond_value):
        return None

    src_id = str(cond_value[0])
    src_node = prompt.get(src_id)
    if src_node is None:
        return None

    src_class = src_node.get("class_type", "")
    src_inputs = src_node.get("inputs", {})

    # ── Direct CLIPTextEncode ─────────────────────────────────────────────
    if src_class == "CLIPTextEncode":
        return _resolve_clip_text_encode_prompt(src_id, prompt, outputs, batch_index)

    # ── Node with its own text field (e.g. some conditioning wrappers) ───
    for k in ("text", "string", "prompt"):
        raw = src_inputs.get(k)
        if raw is not None:
            resolved = _resolve_text_from_graph(raw, prompt, outputs, batch_index=batch_index)
            if resolved:
                return resolved

    # ── Conditioning passthrough: follow the *first* conditioning input ──
    PASSTHROUGH_KEYS = ("conditioning", "cond", "conditioning_1", "conditioning_2")
    for k in PASSTHROUGH_KEYS:
        if k in src_inputs:
            result = _follow_conditioning_to_clip_text(
                src_inputs[k], prompt, outputs, _depth + 1, batch_index
            )
            if result:
                return result

    # ── Last resort: any link-valued input that isn't a model/image slot ─
    _SKIP_KEYS = {"model", "clip", "vae", "image", "mask", "latent",
                  "latent_image", "samples", "positive", "negative"}
    for k, v in src_inputs.items():
        if k in _SKIP_KEYS:
            continue
        if _is_link(v):
            result = _follow_conditioning_to_clip_text(v, prompt, outputs, _depth + 1, batch_index)
            if result:
                return result

    return None


def _find_prompt_texts(prompt, outputs, batch_index=0):
    """
    Walk the prompt graph to find the positive and negative prompt strings.

    Looks for a KSampler-like node (by class name OR by having both
    ``positive`` and ``negative`` conditioning inputs plus a sampler-like
    input such as ``seed``, ``steps``, or ``noise_seed``).  From there it
    follows each conditioning chain independently so the two polarities
    never get mixed up.
    """
    SAMPLER_CLASSES = {
        "KSampler", "KSamplerAdvanced", "SamplerCustom",
        "KSamplerSelect", "KSampler_inspire",
        "KSamplerAdvancedPipe", "KSamplerPipe",
        "FluxKSampler", "FluxSampler",
        "Sampler",
    }
    # Inputs that indicate this node is a sampler even if the class name is unknown
    SAMPLER_HINT_KEYS = {"seed", "steps", "cfg", "sampler_name", "noise_seed", "denoise"}

    for node_id, node in prompt.items():
        class_type = node.get("class_type", "")
        node_inputs = node.get("inputs", {})

        is_sampler = (
            class_type in SAMPLER_CLASSES
            or (
                "positive" in node_inputs
                and "negative" in node_inputs
                and bool(SAMPLER_HINT_KEYS & set(node_inputs.keys()))
            )
        )
        if not is_sampler:
            continue

        # Resolve each polarity independently — critical to keep them separate
        pos_text = _follow_conditioning_to_clip_text(
            node_inputs.get("positive"), prompt, outputs, batch_index=batch_index
        )
        neg_text = _follow_conditioning_to_clip_text(
            node_inputs.get("negative"), prompt, outputs, batch_index=batch_index
        )

        if pos_text or neg_text:
            return pos_text, neg_text

    return None, None


# ---------------------------------------------------------------------------
# Main Capture class (original logic preserved, prompt resolution patched)
# ---------------------------------------------------------------------------

class Capture:
    @classmethod
    def get_inputs(cls):
        inputs = {}
        prompt = hook.current_prompt
        extra_data = hook.current_extra_data

        if hook.prompt_executer and hook.prompt_executer.caches:
            raw_outputs = hook.prompt_executer.caches.outputs
            outputs = (
                raw_outputs
                if hasattr(raw_outputs, "get_output_cache")
                else OutputCacheCompat(raw_outputs)
            )
        else:
            outputs = None

        for node_id, obj in prompt.items():
            class_type = obj["class_type"]
            if class_type not in NODE_CLASS_MAPPINGS:
                continue
            obj_class = NODE_CLASS_MAPPINGS[class_type]
            node_inputs = obj["inputs"]

            input_data = get_input_data(
                node_inputs, obj_class, node_id, outputs, DynamicPrompt(prompt), extra_data
            )

            for node_class, metas in CAPTURE_FIELD_LIST.items():
                if class_type != node_class:
                    continue

                for meta, field_data in metas.items():
                    if field_data.get("validate") and not field_data["validate"](
                        node_id, obj, prompt, extra_data, outputs, input_data
                    ):
                        continue

                    if meta not in inputs:
                        inputs[meta] = []

                    value = field_data.get("value")
                    if value is not None:
                        inputs[meta].append((node_id, value))
                        continue

                    selector = field_data.get("selector")
                    if selector:
                        v = selector(node_id, obj, prompt, extra_data, outputs, input_data)
                        cls._append_value(inputs, meta, node_id, v)
                        continue

                    field_name = field_data["field_name"]
                    value = input_data[0].get(field_name)

                    # ── KEY FIX ──────────────────────────────────────────────
                    # If get_input_data returned a link reference instead of a
                    # resolved string (happens when the text input is wired),
                    # resolve it ourselves by walking the graph.
                    if _is_link(value):
                        value = _resolve_text_from_graph(value, prompt, outputs)
                    # ─────────────────────────────────────────────────────────

                    if value is not None:
                        format_func = field_data.get("format")
                        v = cls._apply_formatting(value, input_data, format_func)
                        cls._append_value(inputs, meta, node_id, v)

        return inputs

    @staticmethod
    def _apply_formatting(value, input_data, format_func):
        if isinstance(value, list) and len(value) > 0:
            value = value[0]
        if format_func:
            value = format_func(value, input_data)
        return value

    @staticmethod
    def _append_value(inputs, meta, node_id, value):
        if isinstance(value, list):
            for x in value:
                inputs[meta].append((node_id, x))
        elif value is not None:
            inputs[meta].append((node_id, value))

    @classmethod
    def get_lora_strings_and_hashes(cls, inputs_before_sampler_node):

        def clean_name(n):
            return os.path.splitext(os.path.basename(n))[0].replace('\\', '_').replace('/', '_').replace(' ', '_').replace(':', '_')

        lora_assertion_re = re.compile(r"<(lora|lyco):([a-zA-Z0-9_\./\\-]+):([0-9.]+)>")

        prompt_texts = [
            val[1]
            for key in [MetaField.POSITIVE_PROMPT, MetaField.NEGATIVE_PROMPT]
            for val in inputs_before_sampler_node.get(key, [])
            if isinstance(val[1], str)
        ]
        prompt_joined = " ".join(prompt_texts).replace("\n", " ").replace("\r", " ").lower()

        lora_names = inputs_before_sampler_node.get(MetaField.LORA_MODEL_NAME, [])
        lora_weights = inputs_before_sampler_node.get(MetaField.LORA_STRENGTH_MODEL, [])
        lora_hashes = inputs_before_sampler_node.get(MetaField.LORA_MODEL_HASH, [])

        lora_names_from_prompt, lora_weights_from_prompt, lora_hashes_from_prompt = [], [], []
        if "<lora:" in prompt_joined:
            for text in prompt_texts:
                for _, name, weight in re.findall(lora_assertion_re, text.replace("\n", " ").replace("\r", " ")):
                    lora_names_from_prompt.append(("prompt_parse", name))
                    lora_weights_from_prompt.append(("prompt_parse", float(weight)))
                    h = calc_lora_hash(name)
                    if h:
                        lora_hashes_from_prompt.append(("prompt_parse", h))

        all_names = lora_names + lora_names_from_prompt
        all_weights = lora_weights + lora_weights_from_prompt
        all_hashes = lora_hashes + lora_hashes_from_prompt

        inputs_before_sampler_node[MetaField.LORA_MODEL_NAME] = all_names
        inputs_before_sampler_node[MetaField.LORA_STRENGTH_MODEL] = all_weights
        inputs_before_sampler_node[MetaField.LORA_MODEL_HASH] = all_hashes

        grouped = defaultdict(list)
        for name, weight, hsh in zip(all_names, all_weights, all_hashes):
            if not (name and weight and hsh):
                continue
            grouped[(hsh[1], weight[1])].append(clean_name(name[1]))

        hashes_in_prompt = {h[1].lower() for h in lora_hashes_from_prompt}

        lora_strings, lora_hashes_list = [], []

        for (hsh, weight), names in grouped.items():
            canonical = min(names, key=len)
            present = hsh.lower() in hashes_in_prompt
            if not present:
                lora_strings.append(f"<lora:{canonical}:{weight}>")
            lora_hashes_list.append(f"{canonical}: {hsh}")

        updated_prompts = []
        if "<lora:" in prompt_joined:
            for text in prompt_texts:
                def replace(m):
                    tag, raw_name, weight = m.group(1), m.group(2), m.group(3)
                    return f"<{tag}:{clean_name(raw_name)}:{weight}>"
                updated_prompts.append(lora_assertion_re.sub(replace, text))
        else:
            updated_prompts = prompt_texts

        lora_hashes_string = ", ".join(lora_hashes_list)
        return lora_strings, lora_hashes_string, updated_prompts

    @classmethod
    def gen_pnginfo_dict(cls, inputs_before_sampler_node, inputs_before_this_node, prompt, save_civitai_sampler=True, batch_index=0):
        pnginfo = {}

        if not inputs_before_sampler_node:
            inputs_before_sampler_node = defaultdict(list)
            cls._collect_all_metadata(prompt, inputs_before_sampler_node)

        # ── PATCH: resolve prompts from graph when capture missed them ───────
        outputs = None
        if hook.prompt_executer and hook.prompt_executer.caches:
            raw_outputs = hook.prompt_executer.caches.outputs
            outputs = (
                raw_outputs
                if hasattr(raw_outputs, "get_output_cache")
                else OutputCacheCompat(raw_outputs)
            )

        current_positive = None
        current_negative = None
        pos_list = inputs_before_sampler_node.get(MetaField.POSITIVE_PROMPT, [])
        neg_list = inputs_before_sampler_node.get(MetaField.NEGATIVE_PROMPT, [])
        if pos_list:
            current_positive = pos_list[0][1] if len(pos_list[0]) > 1 else None
        if neg_list:
            current_negative = neg_list[0][1] if len(neg_list[0]) > 1 else None

        # If either prompt is missing or is just a link reference, re-resolve
        if (not current_positive or _is_link(current_positive) or
                not current_negative or _is_link(current_negative)):
            graph_pos, graph_neg = _find_prompt_texts(prompt, outputs, batch_index=batch_index)
            if graph_pos and (not current_positive or _is_link(current_positive)):
                inputs_before_sampler_node[MetaField.POSITIVE_PROMPT] = [("graph", graph_pos)]
            if graph_neg and (not current_negative or _is_link(current_negative)):
                inputs_before_sampler_node[MetaField.NEGATIVE_PROMPT] = [("graph", graph_neg)]
        # ─────────────────────────────────────────────────────────────────────

        def is_simple(value):
            return isinstance(value, (str, int, float, bool)) or value is None

        def extract(meta_key, label, source=inputs_before_sampler_node):
            val_list = source.get(meta_key, [])
            for link in val_list:
                if len(link) <= 1:
                    continue
                candidate = link[1]
                if candidate is None:
                    continue
                if isinstance(candidate, str):
                    if not candidate.strip():
                        continue
                elif not is_simple(candidate):
                    continue
                value = str(candidate)
                pnginfo[label] = value
                return value
            return None

        positive = extract(MetaField.POSITIVE_PROMPT, "Positive prompt") or ""
        if not positive.strip():
            print_warning("Positive prompt is empty!")

        negative = extract(MetaField.NEGATIVE_PROMPT, "Negative prompt") or ""
        lora_strings, lora_hashes, updated_prompts = cls.get_lora_strings_and_hashes(inputs_before_sampler_node)

        if updated_prompts:
            positive = updated_prompts[0]

        if lora_strings:
            positive += " " + " ".join(lora_strings)

        pnginfo["Positive prompt"] = positive.strip()
        pnginfo["Negative prompt"] = negative.strip()

        if not extract(MetaField.STEPS, "Steps"):
            print_warning("Steps are empty, full metadata won't be added!")
            return {}

        samplers = inputs_before_sampler_node.get(MetaField.SAMPLER_NAME)
        schedulers = inputs_before_sampler_node.get(MetaField.SCHEDULER)

        if save_civitai_sampler:
            pnginfo["Sampler"] = cls.get_sampler_for_civitai(samplers, schedulers)
        elif samplers:
            sampler_name = samplers[0][1]
            if schedulers and schedulers[0][1] != "normal":
                sampler_name += f"_{schedulers[0][1]}"
            pnginfo["Sampler"] = sampler_name

        extract(MetaField.CFG, "CFG scale")
        extract(MetaField.SEED, "Seed")

        clip_skip = extract(MetaField.CLIP_SKIP, "Clip skip")
        if clip_skip is None:
            pnginfo["Clip skip"] = "1"

        image_width_data = inputs_before_sampler_node.get(MetaField.IMAGE_WIDTH, [[None]])
        image_height_data = inputs_before_sampler_node.get(MetaField.IMAGE_HEIGHT, [[None]])

        def extract_dimension(data):
            return data[0][1] if data and len(data[0]) > 1 and isinstance(data[0][1], int) else None

        width = extract_dimension(image_width_data)
        height = extract_dimension(image_height_data)
        if width and height:
            pnginfo["Size"] = f"{width}x{height}"

        extract(MetaField.MODEL_NAME, "Model")
        extract(MetaField.MODEL_HASH, "Model hash")
        extract(MetaField.VAE_NAME, "VAE", inputs_before_this_node)
        extract(MetaField.VAE_HASH, "VAE hash", inputs_before_this_node)

        denoise = inputs_before_sampler_node.get(MetaField.DENOISE)
        dval = denoise[0][1] if denoise else None
        if dval and 0 < float(dval) < 1:
            pnginfo["Denoising strength"] = float(dval)

        if inputs_before_this_node.get(MetaField.UPSCALE_BY) or inputs_before_this_node.get(MetaField.UPSCALE_MODEL_NAME):
            pnginfo["Denoising strength"] = float(dval or 1.0)

        extract(MetaField.UPSCALE_BY, "Hires upscale", inputs_before_this_node)
        extract(MetaField.UPSCALE_MODEL_NAME, "Hires upscaler", inputs_before_this_node)

        if lora_hashes:
            pnginfo["Lora hashes"] = f'"{lora_hashes}"'

        pnginfo.update(cls.gen_loras(inputs_before_sampler_node))
        pnginfo.update(cls.gen_embeddings(inputs_before_sampler_node))

        hashes = cls.get_hashes_for_civitai(inputs_before_sampler_node, inputs_before_this_node)
        if hashes:
            pnginfo["Hashes"] = json.dumps(hashes)

        return pnginfo

    @classmethod
    def _collect_all_metadata(cls, prompt, result_dict):
        # ── PATCH: use the graph-walk resolver for prompt texts ───────────────
        outputs = None
        if hook.prompt_executer and hook.prompt_executer.caches:
            raw_outputs = hook.prompt_executer.caches.outputs
            outputs = (
                raw_outputs
                if hasattr(raw_outputs, "get_output_cache")
                else OutputCacheCompat(raw_outputs)
            )

        def _append_metadata(meta, node_id, value):
            if value is not None:
                result_dict[meta].append((node_id, value, 0))

        resolved = {
            "prompt": Trace.find_node_with_fields(prompt, {"positive", "negative"}),
            "denoise": Trace.find_node_with_fields(prompt, {"denoise"}),
            "sampler": Trace.find_node_with_fields(prompt, {"seed", "steps", "cfg", "sampler_name", "scheduler"}),
            "size": Trace.find_node_with_fields(prompt, {"width", "height"}),
            "model": Trace.find_node_with_fields(prompt, {"ckpt_name"}),
        }

        for node_id, node in Trace.find_all_nodes_with_fields(prompt, {"lora_name", "strength_model"}):
            if node is not None:
                inputs = node.get("inputs", {})
                name = inputs.get("lora_name")
                strength = inputs.get("strength_model")
                _append_metadata(MetaField.LORA_MODEL_NAME, node_id, name)
                _append_metadata(MetaField.LORA_MODEL_HASH, node_id, calc_lora_hash(name) if name else None)
                _append_metadata(MetaField.LORA_STRENGTH_MODEL, node_id, strength)

        model_node = resolved.get("model")
        if model_node and model_node[1] is not None:
            node_id, node = model_node
            inputs = node.get("inputs", {})
            name = inputs.get("ckpt_name")
            _append_metadata(MetaField.MODEL_NAME, node_id, name)
            _append_metadata(MetaField.MODEL_HASH, node_id, calc_model_hash(name) if name else None)

        denoise_node = resolved.get("denoise")
        if denoise_node and denoise_node[1] is not None:
            node_id, node = denoise_node
            val = node.get("inputs", {}).get("denoise")
            _append_metadata(MetaField.DENOISE, node_id, val)

        sampler_node = resolved.get("sampler")
        if sampler_node and sampler_node[1] is not None:
            node_id, node = sampler_node
            inputs = node.get("inputs", {})
            for key, meta in {
                "sampler_name": MetaField.SAMPLER_NAME,
                "scheduler": MetaField.SCHEDULER,
                "seed": MetaField.SEED,
                "steps": MetaField.STEPS,
                "cfg": MetaField.CFG,
            }.items():
                _append_metadata(meta, node_id, inputs.get(key))

        size_node = resolved.get("size")
        if size_node and size_node[1] is not None:
            node_id, node = size_node
            inputs = node.get("inputs", {})
            for key, meta in {
                "width": MetaField.IMAGE_WIDTH,
                "height": MetaField.IMAGE_HEIGHT,
            }.items():
                _append_metadata(meta, node_id, inputs.get(key))

        # ── PATCHED prompt resolution ─────────────────────────────────────────
        # Only match nodes that look like samplers (have seed/steps/cfg etc.)
        # to avoid accidentally matching ConditioningCombine or similar nodes
        # that also have positive+negative keys but are NOT the sampler.
        _SAMPLER_HINT_KEYS = {"seed", "steps", "cfg", "sampler_name", "noise_seed", "denoise"}
        _SAMPLER_CLASSES = {
            "KSampler", "KSamplerAdvanced", "SamplerCustom", "KSamplerSelect",
            "KSampler_inspire", "KSamplerAdvancedPipe", "KSamplerPipe",
            "FluxKSampler", "FluxSampler", "Sampler",
        }

        found_prompts = False
        for node_id, node in prompt.items():
            node_inputs = node.get("inputs", {})
            class_type = node.get("class_type", "")

            # Must have both conditioning keys
            if "positive" not in node_inputs or "negative" not in node_inputs:
                continue

            # Must look like a sampler node — don't match conditioning utility nodes
            is_sampler = (
                class_type in _SAMPLER_CLASSES
                or bool(_SAMPLER_HINT_KEYS & set(node_inputs.keys()))
            )
            if not is_sampler:
                continue

            pos_text = _follow_conditioning_to_clip_text(node_inputs.get("positive"), prompt, outputs, batch_index=0)
            neg_text = _follow_conditioning_to_clip_text(node_inputs.get("negative"), prompt, outputs, batch_index=0)

            if pos_text or neg_text:
                if pos_text:
                    _append_metadata(MetaField.POSITIVE_PROMPT, node_id, pos_text)
                if neg_text:
                    _append_metadata(MetaField.NEGATIVE_PROMPT, node_id, neg_text)

                # Embeddings from resolved text
                for text in (pos_text, neg_text):
                    if not text:
                        continue
                    for emb_name, emb_hash in zip(
                        extract_embedding_names(text), extract_embedding_hashes(text)
                    ):
                        _append_metadata(MetaField.EMBEDDING_NAME, node_id, emb_name)
                        _append_metadata(MetaField.EMBEDDING_HASH, node_id, emb_hash)

                found_prompts = True
                break  # First sampler we find is enough

        # Final fallback – old behaviour preserved for edge-cases
        if not found_prompts:
            for node_id, node in Trace.find_all_nodes_with_fields(prompt, {"positive", "negative"}):
                if node is None:
                    continue
                inputs = node.get("inputs", {})
                pos_ref = inputs.get("positive", [None])[0]
                neg_ref = inputs.get("negative", [None])[0]

                def resolve_text(ref):
                    if isinstance(ref, list):
                        ref = ref[0]
                    if not isinstance(ref, str):
                        return None
                    n = prompt.get(ref)
                    if n is None:
                        return None
                    raw = n.get("inputs", {}).get("text")
                    if isinstance(raw, str):
                        return raw
                    return _resolve_text_from_graph(raw, prompt, outputs)

                pos_text = resolve_text(pos_ref)
                neg_text = resolve_text(neg_ref)
                _append_metadata(MetaField.POSITIVE_PROMPT, pos_ref, pos_text)
                _append_metadata(MetaField.NEGATIVE_PROMPT, neg_ref, neg_text)

                for text in (pos_text, neg_text):
                    if not text:
                        continue
                    for name, h in zip(extract_embedding_names(text), extract_embedding_hashes(text)):
                        _append_metadata(MetaField.EMBEDDING_NAME, node_id, name)
                        _append_metadata(MetaField.EMBEDDING_HASH, node_id, h)

    @classmethod
    def extract_model_info(cls, inputs, meta_field_name, prefix):
        model_info_dict = {}
        model_names = inputs.get(meta_field_name, [])
        model_hashes = inputs.get(f"{meta_field_name}_HASH", [])

        for index, (model_name, model_hash) in enumerate(zip(model_names, model_hashes)):
            field_prefix = f"{prefix}_{index}"
            model_info_dict[f"{field_prefix} name"] = os.path.splitext(os.path.basename(model_name[1]))[0]
            model_info_dict[f"{field_prefix} hash"] = model_hash[1]

        return model_info_dict

    @classmethod
    def gen_loras(cls, inputs):
        return cls.extract_model_info(inputs, MetaField.LORA_MODEL_NAME, "Lora")

    @classmethod
    def gen_embeddings(cls, inputs):
        return cls.extract_model_info(inputs, MetaField.EMBEDDING_NAME, "Embedding")

    @classmethod
    def gen_parameters_str(cls, pnginfo_dict):
        if not pnginfo_dict or not isinstance(pnginfo_dict, dict):
            return ""

        def clean_value(value):
            if value is None:
                return ""
            return str(value).strip().replace("\n", " ")

        def strip_embedding_prefix(text):
            return text.replace("embedding:", "")

        cleaned_dict = {k: clean_value(v) for k, v in pnginfo_dict.items()}

        pos = strip_embedding_prefix(cleaned_dict.get("Positive prompt", ""))
        neg = strip_embedding_prefix(cleaned_dict.get("Negative prompt", ""))

        result = [pos]
        if neg:
            result.append(f"Negative prompt: {neg}")

        s_list = [
            f"{k}: {v}"
            for k, v in cleaned_dict.items()
            if k not in {"Positive prompt", "Negative prompt"} and v not in {None, ""}
        ]

        result.append(", ".join(s_list))
        return "\n".join(result)

    @classmethod
    def get_hashes_for_civitai(cls, inputs_before_sampler_node, inputs_before_this_node):
        def extract_single(inputs, key):
            items = inputs.get(key, [])
            return items[0][1] if items and len(items[0]) > 1 else None

        def extract_named_hashes(names, hashes, prefix):
            result = {}
            for name, h in zip(names, hashes):
                base_name = os.path.splitext(os.path.basename(name[1]))[0]
                result[f"{prefix}:{base_name}"] = h[1]
            return result

        resource_hashes = {}

        model = extract_single(inputs_before_sampler_node, MetaField.MODEL_HASH)
        if model:
            resource_hashes["model"] = model

        vae = extract_single(inputs_before_this_node, MetaField.VAE_HASH)
        if vae:
            resource_hashes["vae"] = vae

        upscaler_hash = extract_single(inputs_before_this_node, MetaField.UPSCALE_MODEL_HASH)
        if upscaler_hash:
            resource_hashes["upscaler"] = upscaler_hash

        resource_hashes.update(extract_named_hashes(
            inputs_before_sampler_node.get(MetaField.LORA_MODEL_NAME, []),
            inputs_before_sampler_node.get(MetaField.LORA_MODEL_HASH, []),
            "lora"
        ))

        resource_hashes.update(extract_named_hashes(
            inputs_before_sampler_node.get(MetaField.EMBEDDING_NAME, []),
            inputs_before_sampler_node.get(MetaField.EMBEDDING_HASH, []),
            "embed"
        ))

        return resource_hashes

    @classmethod
    def get_sampler_for_civitai(cls, sampler_names, schedulers):
        """
        Get the pretty sampler name for Civitai.
        Reference: https://github.com/civitai/civitai/blob/main/src/server/common/constants.ts
        """
        sampler_dict = {
            'euler': 'Euler',
            'euler_ancestral': 'Euler a',
            'heun': 'Heun',
            'dpm_2': 'DPM2',
            'dpm_2_ancestral': 'DPM2 a',
            'lms': 'LMS',
            'dpm_fast': 'DPM fast',
            'dpm_adaptive': 'DPM adaptive',
            'dpmpp_2s_ancestral': 'DPM++ 2S a',
            'dpmpp_sde': 'DPM++ SDE',
            'dpmpp_sde_gpu': 'DPM++ SDE',
            'dpmpp_2m': 'DPM++ 2M',
            'dpmpp_2m_sde': 'DPM++ 2M SDE',
            'dpmpp_2m_sde_gpu': 'DPM++ 2M SDE',
            'ddim': 'DDIM',
            'plms': 'PLMS',
            'uni_pc': 'UniPC',
            'uni_pc_bh2': 'UniPC',
            'lcm': 'LCM'
        }

        sampler = None
        scheduler = None

        if sampler_names and len(sampler_names) > 0:
            sampler = sampler_names[0][1]
        if schedulers and len(schedulers) > 0:
            scheduler = schedulers[0][1]

        def get_scheduler_name(sampler_name, scheduler):
            if scheduler == "karras":
                return f"{sampler_name} Karras"
            elif scheduler == "exponential":
                return f"{sampler_name} Exponential"
            elif scheduler == "normal":
                return sampler_name
            else:
                return f"{sampler_name}_{scheduler}"

        if not sampler:
            return None

        if sampler in sampler_dict:
            return get_scheduler_name(sampler_dict[sampler], scheduler)

        return get_scheduler_name(sampler, scheduler)