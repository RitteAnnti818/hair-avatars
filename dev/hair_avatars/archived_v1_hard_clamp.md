# Archived: Inextensible-chain v1 (hard norm-clamp) — 2026-08-21

**Status: superseded by v2 (magnitude/direction decoupled), see `next_steps_plan.md` and the live
docstrings in `scene/flame_gaussian_model.py`. This file exists only as a historical code snapshot —
this version was deleted from the working tree when v2 replaced it, and no separate git commit ever
captured it, so it is reconstructed here from the same session that wrote it, for the record.**

## Result (why it was abandoned)

Tested static-only (`--disable_strand_dynamic`) on subjects 306/302, FREE sequence, against the
free-vector static-only baseline (+0.281dB / +0.014dB precise-hair-crop ΔPSNR):

| cap | 306 | 302 |
|---|---|---|
| 0.3 (= `threshold_strand_static`) | +0.050dB | −0.027dB |
| 0.5 (relaxed) | +0.045dB | −0.051dB |

Both cap values underperformed the free-vector baseline on both subjects; relaxing the cap didn't
recover the gain, which ruled out "cap value too tight" as the explanation. A gradient analysis (see
`next_steps_plan.md`) showed why: this clamp entangles magnitude and direction through a single raw
vector's norm, so the *direction* gradient decays toward zero once the raw vector's norm drifts past
the cap (`d(output)/d(raw) -> 0` as `||raw|| -> infinity`) — verified numerically, and shown to hold
almost identically for a smoother tanh-based version of the same clamp, so smoothing the kink alone
doesn't fix it either. v2 (magnitude and direction as separate parameters) was designed specifically to
remove this entanglement.

## The code (as it existed, in `scene/flame_gaussian_model.py`)

```python
@staticmethod
def _clamp_norm(v, cap, eps=1e-8):
    """Rescale v so ||v|| <= cap exactly, leaving v unchanged when it's already inside the cap.
    Unlike a ReLU-threshold penalty (which only discourages ||v|| > cap via a loss gradient, and can
    still be outweighed by a stronger competing gradient), this makes ||v|| > cap unreachable by
    construction -- no parameter value can produce a v whose clamped output exceeds the cap."""
    norm = v.norm(dim=-1, keepdim=True)
    return v * (cap / norm.clamp(min=cap + eps))

def _update_strand_delta_cumsum(self, timestep):
    """Accumulate the per-link learned offset delta_j = static_j + g(m(t))*dynamic_j(t) along each
    strand's chain: Delta_j = Delta_{j-1} + w_j * delta_j, Delta_0 = 0 (root stays perfectly
    rigid). w_j ramps 0 (root) -> 1 (tip) so freedom grows away from the scalp attachment.
    g(m(t)) is the motion gate (opt-in via enable_motion_gate); when off, dynamic_j(t) is applied
    unconditionally, i.e. g === 1 (previous behavior, unchanged).

    enable_inextensible_chain (opt-in): hard-clamps ||static_j|| and ||dynamic_j(t)|| (each
    separately, before the motion gate) to inextensible_static_cap / inextensible_dynamic_cap.
    Structural replacement for lambda_strand_static/lambda_strand_dynamic's soft ReLU penalty --
    see _clamp_norm. Since Delta_j - Delta_{j-1} = w_j * delta_j, this also hard-bounds the
    adjacent-link separation lambda_strand_coherence only discourages softly, targeting the
    comet-tail failure mode (a single/few near-tip links, where w_j->1, developing an unbounded
    offset under strong photometric gradient) structurally rather than via penalty weight."""
    static_t = self._strand_delta_static
    if self.enable_inextensible_chain:
        static_t = self._clamp_norm(static_t, self.inextensible_static_cap)
    delta_t = static_t
    if self._strand_delta_dynamic is not None:
        dynamic_t = self._strand_delta_dynamic[:, :, timestep, :]
        if self.enable_inextensible_chain:
            dynamic_t = self._clamp_norm(dynamic_t, self.inextensible_dynamic_cap)
        if self.enable_motion_gate and self.motion_gate_ref is not None:
            dynamic_t = dynamic_t * self._compute_motion_gate(timestep)
        delta_t = delta_t + dynamic_t
    # (num_strands, max_chain_len, 3)
    link_idx = torch.arange(self.max_chain_len, device=delta_t.device).float()[None, :]  # (1, L)
    denom = (self.strand_chain_len.float() - 1).clamp(min=1)[:, None]  # (num_strands, 1)
    w = (link_idx / denom).clamp(0, 1)  # (num_strands, L), root->0, tip->1
    self.strand_delta_cumsum = torch.cumsum(delta_t * w[..., None], dim=1)  # (num_strands, max_chain_len, 3)
```

This version reused the existing `_strand_delta_static`/`_strand_delta_dynamic` free-vector tensors
directly (clamped just before the cumsum) — no new state, no changes needed to `training_setup`,
`save_ply`/`load_ply`, or the loss section in `train.py`. That minimal footprint is precisely why it
was the first thing tried; v2's decoupled parameterization needed new tensors and touches all of those
call sites (see the current, live code).

## Gradient analysis that ruled out a smoother variant too

```python
def hard_clamp(v, cap, eps=1e-8):
    norm = v.norm(dim=-1, keepdim=True)
    return v * (cap / norm.clamp(min=cap + eps))

def soft_clamp(v, cap, eps=1e-8):
    norm = v.norm(dim=-1, keepdim=True)
    return v * (cap * torch.tanh(norm / cap) / norm.clamp(min=eps))
```
Tangential (direction-steering) Jacobian `d(out_y)/d(v_y)` at `v=(n,0,0)`, measured by finite
difference, for both variants at increasing `n` (raw norm) relative to `cap`:

| n / cap | hard | soft (tanh) |
|---|---|---|
| 0.5x | 1.000 | 0.924 |
| 1.0x | 1.000 | 0.762 |
| 2x | 0.500 | 0.482 |
| 5x | 0.200 | 0.200 |
| 10x | 0.100 | 0.100 |
| 30x | 0.033 | 0.033 |

Both decay asymptotically as `~cap/n`; the soft version is additionally *worse* than hard clamp in the
below-cap regime (0.924, 0.762 vs. hard clamp's exact 1.0 no-op there). Never trained — ruled out by
this analysis alone before spending GPU time on it.
