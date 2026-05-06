"""
HScore inference API.

Requires the ``[inference]`` optional dependencies:
    pip install hscore-client[inference]

This module is intentionally lazy about importing torch and transformers —
the imports happen at call time so that ``import hscore`` works without
torch installed (e.g. for license checks or probe loading only).

Typical usage
-------------
    import hscore

    # Single score
    score = hscore.score("The Eiffel Tower is in Berlin.")
    print(score)  # e.g. 0.91

    # Batch
    scores = hscore.batch_score(["text 1", "text 2"], device="cuda")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .model import load_probe

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────────────

# Map hscore model_id → default HuggingFace model name.
# Customers can override via the base_model argument.
_DEFAULT_BASE_MODELS: dict[str, str] = {
    "hscore-qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
}

# Module-level LRU: avoids re-loading the (large) base LLM between calls.
_MODEL_CACHE: dict[str, tuple[object, object]] = {}  # hf_name → (model, tokenizer)


# ── Public API ─────────────────────────────────────────────────────────────────

def score(
    text: str,
    model_id: str = "hscore-qwen2.5-7b",
    base_model: str | None = None,
    device: str = "cuda",
    layer: int = 16,
) -> float:
    """
    Score *text* for hallucination risk. Returns a float in [0, 1].

    Higher scores indicate higher internal friction — i.e. the base LLM's
    hidden states look more like the hallucinated distribution at the probe
    layer. A score near 1.0 suggests the model is uncertain or confabulating.

    Args:
        text:       The text to score (model output or prompt+completion).
        model_id:   HScore probe to use. Defaults to ``"hscore-qwen2.5-7b"``.
        base_model: HuggingFace model name for the base LLM. If None, the
                    default for *model_id* is used (see ``_DEFAULT_BASE_MODELS``).
        device:     PyTorch device string, e.g. ``"cuda"``, ``"cpu"``, ``"mps"``.
        layer:      Transformer hidden-state layer index to probe (0 = embedding).

    Returns:
        Float in [0, 1].

    Raises:
        LicenseError:   If the license is invalid or expired.
        ImportError:    If ``torch`` / ``transformers`` are not installed.
        KeyError:       If *model_id* has no known default base model and
                        *base_model* was not supplied.
    """
    probe = load_probe(model_id)
    hf_name = _resolve_base_model(model_id, base_model)
    model, tokenizer = _load_base_model(hf_name, device)
    hidden = _extract_hidden(text, model, tokenizer, device, layer)
    return _probe_score(probe, hidden)


def batch_score(
    texts: list[str],
    model_id: str = "hscore-qwen2.5-7b",
    base_model: str | None = None,
    device: str = "cuda",
    layer: int = 16,
) -> list[float]:
    """
    Score a list of texts for hallucination risk.

    Reuses the cached base model and probe across all texts. More efficient
    than calling ``score()`` in a loop because model loading happens once.

    Args:
        texts:      List of strings to score.
        model_id:   HScore probe to use.
        base_model: Override for the HuggingFace base model name.
        device:     PyTorch device string.
        layer:      Transformer hidden-state layer index.

    Returns:
        List of floats in [0, 1], same order as *texts*.
    """
    if not texts:
        return []

    probe = load_probe(model_id)
    hf_name = _resolve_base_model(model_id, base_model)
    model, tokenizer = _load_base_model(hf_name, device)

    return [
        _probe_score(probe, _extract_hidden(t, model, tokenizer, device, layer))
        for t in texts
    ]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _resolve_base_model(model_id: str, override: str | None) -> str:
    if override:
        return override
    if model_id not in _DEFAULT_BASE_MODELS:
        raise KeyError(
            f"No default base model known for '{model_id}'. "
            "Pass base_model='<HuggingFace repo>' explicitly."
        )
    return _DEFAULT_BASE_MODELS[model_id]


def _load_base_model(hf_name: str, device: str) -> tuple[object, object]:
    """Load (and cache) a HuggingFace CausalLM + tokenizer."""
    if hf_name not in _MODEL_CACHE:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "The HScore inference API requires torch and transformers. "
                "Install them with: pip install hscore-client[inference]"
            ) from exc

        logger.info("Loading base model %s on %s …", hf_name, device)
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            torch_dtype=torch.float16,
            device_map=device,
        ).eval()
        _MODEL_CACHE[hf_name] = (model, tokenizer)
        logger.info("Base model loaded.")

    return _MODEL_CACHE[hf_name]


def _extract_hidden(
    text: str,
    model: object,
    tokenizer: object,
    device: str,
    layer: int,
) -> object:
    """
    Run a single forward pass and return the mean hidden state at *layer*
    over the last 10 tokens, shape (hidden_size,), dtype float32.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for inference. "
            "Install with: pip install hscore-client[inference]"
        ) from exc

    inputs = tokenizer(  # type: ignore[operator]
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)  # type: ignore[operator]

    # hidden_states is a tuple of (n_layers+1) tensors, each [1, seq_len, hidden_size]
    hidden = out.hidden_states[layer][0, -10:, :].mean(0).float()
    return hidden


def _probe_score(probe: object, hidden: object) -> float:
    """Apply the FrictionProbe to a hidden-state vector, returning P(hallucinated)."""
    import numpy as np

    h = hidden.cpu().numpy().reshape(1, -1)  # type: ignore[union-attr]
    # predict_proba returns [[P(class_0), P(class_1)]]
    # Convention: class 1 = hallucinated / high friction
    proba = probe.predict_proba(h)  # type: ignore[union-attr]
    return float(proba[0, 1])
