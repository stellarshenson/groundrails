"""R19-H168 EuroBERT load-compatibility shim - the ONLY deviation from stock loading.

EuroBERT ships pinned remote code (`modeling_eurobert.py`) written against
transformers 4.40.0.dev0. This project runs transformers 5.14.1, which REMOVED
the `"default"` entry from `ROPE_INIT_FUNCTIONS` (surviving keys: dynamic,
linear, llama3, longrope, proportional, yarn). EuroBERT's config carries
`rope_scaling: null`, so its rotary module takes the `"default"` branch and dies
with `KeyError: 'default'` at construction.

Downgrading transformers is NOT an option - it would invalidate every banked
checkpoint read in this campaign. The shim re-registers the 4.40 default under
its old key. It is behaviour-preserving BY CONSTRUCTION: it reproduces the exact
closed form transformers 4.40 supplied, and `verify()` checks that form against
the analytic RoPE definition rather than trusting the reimplementation.

    inv_freq[i] = 1 / theta^(2i/d),  i = 0 .. d/2-1

Recorded as a deviation in the R19-H168 registration. It touches model LOADING
only - no weight, no hyperparameter, no data path.
"""

import torch
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

KEY = "default"


def _head_dim(config):
    return getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads)


def _theta(config):
    """EuroBERT's RoPE base, wherever this transformers version keeps it.

    5.x migrates the checkpoint's top-level `rope_theta: 250000` into
    `rope_parameters = {"rope_theta": 250000, "rope_type": "default"}`. The
    config ALSO exposes `default_theta = 10000.0`, which is NOT this model's
    base - silently falling back to it would give a wrong-but-trainable
    positional encoding, so a miss raises instead.
    """
    rp = getattr(config, "rope_parameters", None)
    if isinstance(rp, dict) and rp.get("rope_theta") is not None:
        return float(rp["rope_theta"])
    t = getattr(config, "rope_theta", None)
    if t is not None:
        return float(t)
    raise RuntimeError(
        "EuroBERT RoPE base not found in config under `rope_parameters['rope_theta']` "
        "or `rope_theta`. Refusing to fall back to `default_theta` - a wrong base "
        "trains silently.")


def _compute_default_rope_parameters(config=None, device=None, seq_len=None, **kw):
    """The transformers 4.40 `"default"` RoPE initialiser, verbatim in behaviour."""
    base = _theta(config)
    dim = int(_head_dim(config) * getattr(config, "partial_rotary_factor", 1.0))
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(
            device=device, dtype=torch.float) / dim)
    )
    return inv_freq, 1.0


def install():
    """Idempotent. Returns True if the shim was needed, False if stock already works."""
    if KEY in ROPE_INIT_FUNCTIONS:
        return False
    ROPE_INIT_FUNCTIONS[KEY] = _compute_default_rope_parameters
    return True


def verify(config, tol=1e-6):
    """Positive control - the shim's output must match the analytic definition.

    Checked independently of the implementation above so a typo in the exponent
    cannot pass. A wrong RoPE loads and produces finite activations, so this and
    the gate's masked-token check are the only things standing between a broken
    positional encoding and a training run that silently learns nothing.
    """
    inv_freq, scaling = ROPE_INIT_FUNCTIONS[KEY](config, device=None)
    dim = int(_head_dim(config) * getattr(config, "partial_rotary_factor", 1.0))
    base = _theta(config)
    want = torch.tensor([1.0 / (base ** ((2 * i) / dim)) for i in range(dim // 2)])
    max_err = float((inv_freq - want).abs().max())
    ok = bool(inv_freq.shape == want.shape and max_err <= tol
              and abs(scaling - 1.0) <= tol and abs(float(inv_freq[0]) - 1.0) <= tol)
    return {"n_freqs": int(inv_freq.numel()), "expected_n": dim // 2,
            "head_dim": _head_dim(config), "rope_theta": base,
            "inv_freq_first": round(float(inv_freq[0]), 8),
            "inv_freq_last": round(float(inv_freq[-1]), 8),
            "max_abs_err_vs_analytic": max_err, "attention_scaling": scaling,
            "pass": ok}
