# evo2-silicon

**Native macOS Apple Silicon (MPS) Inference Loader for Arc Institute's Evo2 (StripedHyena)**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Platform: macOS Apple Silicon](https://img.shields.io/badge/Platform-Apple_Silicon_(MPS)-gray.svg)](https://developer.apple.com/metal/pytorch/)

**Author:** Chun Tang ([ORCID: 0009-0008-4128-2071](https://orcid.org/0009-0008-4128-2071))

## Overview

Upstream releases of Arc Institute's [Evo2](https://github.com/ArcInstitute/evo2) require CUDA-specific `flash_attn` and `triton` kernels, preventing local execution on macOS workstations. This repository provides a lightweight runtime loader (`app.py`) documented in upstream workaround discussion [ArcInstitute/evo2#222](https://github.com/ArcInstitute/evo2/issues/222) to run StripedHyena genomic models locally on Apple Silicon unified memory via PyTorch Metal Performance Shaders (`mps`).

## Technical Approach

1. **Runtime Kernel Bypass:** Intercepts hard-coded `flash_attn` import checks at load time without modifying upstream library source files.
2. **Unified Memory Loading:** Routes `torch.load` checkpoint placement across `meta`/`mps` in `bfloat16` to avoid host-RAM duplication spikes on M-series Macs.
3. **Local Cache Resolution:** Resolves Hugging Face Hub snapshot paths cleanly on macOS filesystems.

## Quickstart

```bash
pip install -r requirements.txt
python3 app.py
```

## Verification Prompt (Cas9 Scaffold Sequence)

```text
GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGCTTT
```

## Reference Throughput

| Hardware | Unified Memory | Model | Precision | Latency |
| :--- | :--- | :--- | :--- | :--- |
| Apple Mac Studio (M3 Ultra) | 256 GB | `evo2_7b` | `bfloat16` | ~0.75 s / sequence |

## License

Released under the [Apache-2.0 License](LICENSE).
