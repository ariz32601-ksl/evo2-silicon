## Sovereign-Evo2
**Native Apple Silicon (MPS) Loader for Arc Institute's Evo2 (StripedHyena)**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hardware](https://img.shields.io/badge/Hardware-Apple_Silicon-gray.svg)](https://www.apple.com/mac/)

**Developed by KSL Research**

## The Problem
The official Evo2 release relies on `flash_attn` and `triton` kernels which are hard-coded for NVIDIA CUDA. This prevents local inference on macOS devices, blocking researchers from running genomic models offline.

## The Sovereign Solution
This repository contains a patched `app.py` loader that:
1.  **Runtime Interception:** Surgically bypasses Flash Attention checks without modifying the source library.
2.  **Memory Optimization:** Redirects `torch.load` to `mps`/`meta` to prevent CPU OOM spikes (Critical for M-Series Unified Memory).
3.  **Cache Resolution:** Implements smart pathing for HuggingFace caching on macOS filesystems.

## Usage

### 1. Install Dependencies
pip install -r requirements.txt

### 2. Run the Loader
python3 app.py

### 3. Verify (Cas9 Scaffold Sequence)
Paste this sequence into the UI to verify structural generation:
GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGCTTT

## Benchmarks
Hardware: Mac Studio (M3 Ultra, 128GB)
Model: evo2_7b
Performance: ~0.75s / sequence
Precision: BFloat16

## Acknowledgements
Principal Investigator: KSL Research
Algorithmic Architect: Sovereign Intelligence Unit (AI Co-Pilot)
Infrastructure: DeepSeek-Tang & The God-Stack Cluster