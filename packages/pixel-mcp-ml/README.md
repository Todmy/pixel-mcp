# pixel-mcp-ml

ML extras for [pixel-mcp](../pixel-mcp). Currently ships DINOv2 perceptual similarity — the Level 1 escalation gate of the Convergence Loop. Future slices add OmniParser element detection and VLM bridges (Qwen2.5-VL, Claude Sonnet API).

## Install

The package itself is lightweight (Pillow, numpy, typer). The actual DINOv2 model needs the `dinov2` extra, which pulls in `transformers` + `torch` + `safetensors`:

```bash
uv tool install pixel-mcp-ml --extra dinov2
```

You can install the package without the extra (e.g. to use it as a typed Python dependency without paying the torch download cost). Calling DINOv2 functions in that state raises `DINOv2NotInstalledError` with the install hint above.

## CLI

```bash
pixel-mcp-ml dinov2-compare <image_a> <image_b> [--model-size small|base] [--json]
```

Human-readable output:

```
Similarity: 0.9714
```

JSON output (`--json`):

```json
{"similarity": 0.9714, "model_size": "small", "device": "cpu"}
```

Exit codes: `0` success, `1` if either image is missing, `12` if the DINOv2 extras are not installed.

## Python API

```python
from pixel_mcp_ml import compute_dinov2_similarity, compute_dinov2_similarity_batch

# Single pair
sim = compute_dinov2_similarity("design.png", "render.png", model_size="small")

# Many pairs — model loaded once
pairs = [(Path("a1.png"), Path("b1.png")), (Path("a2.png"), Path("b2.png"))]
sims = compute_dinov2_similarity_batch(pairs)
```

The first call loads the model (defaults to `facebook/dinov2-small`, ~88MB). Subsequent calls reuse the cached model. Device is auto-detected: CUDA, then MPS (Apple Silicon), then CPU.

## Model sizes

| Size  | HF id                    | Approx. weights |
|-------|--------------------------|-----------------|
| small | `facebook/dinov2-small`  | ~88 MB          |
| base  | `facebook/dinov2-base`   | ~330 MB         |

`small` is the default — it is fast enough on CPU and good enough as a Level 1 gate. `base` is available for cases where small-but-meaningful UI differences keep slipping past.

## Wiring into the Convergence Loop

Slice v0.5-2 ships this package as a standalone tool. The wiring into the `pixel-mcp check` Level 1 escalation gate lands in v0.5-3.
