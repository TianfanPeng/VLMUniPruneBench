# UniPruneBench: A Visual Input Token Compression Benchmark for Large Multimodal Models

[![Project Page](https://img.shields.io/badge/Project_Page-blue)](https://uniprunebench-lmm.github.io/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2511.02650)


UniPruneBench is an open-source benchmark for evaluating visual input token compression in large multimodal models (LMMs). It integrates with VLMEvalKit to enable efficient pruning evaluations on various datasets, focusing on methods like [DART](https://github.com/ZichenWen1/DART), [PyramidDrop](https://github.com/Cooperx521/PyramidDrop), [FastV](https://github.com/pkunlp-icler/FastV), [FitPrune](https://github.com/ywh187/FitPrune), [VTW](https://github.com/lzhxmu/VTW), [G-Prune](https://github.com/jytmelon/G-Prune/), [LLaVA-PruMerge](https://github.com/42Shawn/LLaVA-PruMerge), [divprune](https://github.com/vbdi/divprune), [MustDrop](https://github.com/liuting20/MustDrop), [VisPruner](https://github.com/Theia-4869/VisPruner) and so on.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [🎯 Goal](#-the-goal-of-uniprunebench)
- [🆕 News](#-news)
- [📂 File Structure](#-file-structure)
- [🏗️ QuickStart](#-quickstart)
- [📊 Accuracy Evaluation](#-accuracy-evaluation)
- [📊 Latency Evaluation](#-latency-evaluation)

- [📊 Evaluation Results](#-evaluation-results)
- [🧩 Adding a New Pruning Method](#-adding-a-new-pruning-method)
- [🖊️ Citation](#-citation)

## 🎯 The Goal of UniPruneBench

**The benchmark is designed to:**
1. Provide an **easy-to-use**, **open-source toolkit** for evaluating token pruning strategies in large multimodal models, making results reproducible and comparable.
2. Enable developers to test pruning methods on benchmarks with minimal setup – just specify the method and keep ratio, and the codebase handles integration with VLMEvalKit.

**The benchmark is not designed to:**
1. Reproduce exact numerical results from pruning papers without adaptation, due to differences in hooks, pruning settings, and environments.


## 🆕 News
- **[2025-10-14]** Released codebase for UniPruneBench, supporting Qwen2.5-VL and InternVL models with multiple pruning strategies.

## 🏗️ QuickStart

### Environment Installation
```bash
# Create the Conda environment with Python 3.12.3
conda create -n vlm-prune python=3.12.3 -y

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate vlm-prune

# Install the packages using pip
pip install transformers==4.54.0
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn==2.7.3
pip install timm==1.0.19

cd VLMEvalKit && pip install -e .
```

## 📊 Accuracy Evaluation
Evaluate on a benchmark like MME with 88.9% pruning (keeping 11.1% of tokens).

> **New:** Use `INTRA_LLM_LAYER_SPECS="layer:method[:ratio],..."` to describe per-layer intra-LLM pruning.  
> Example: `INTRA_LLM_LAYER_SPECS="2:fitprune_intra_llm:0.5;6:pdrop_intra_llm:0.3;12:pdrop_intra_llm:0.2"`

#### Qwen2.5-VL Example
```bash
export KEEP_RATIO=0.111       
export PRUNE_METHOD_PRE_LLM=divprune_pre_llm
export INTRA_LLM_LAYER_SPECS="2:fastv_intra_llm:0.5"

python run_qwenvl2_5.py \
    --model Qwen2.5-VL-7B-Instruct \
    --data MME \
    --verbose
```

#### InternVL-3 Example
```bash
export KEEP_RATIO=0.111
export INTRA_LLM_LAYER_SPECS="2:fitprune_intra_llm:0.5;6:pdrop_intra_llm:0.3;12:pdrop_intra_llm:0.2"

python run_internvl3.py \
    --model InternVL3-8B \
    --data MME \
    --verbose
```
## 📊 Latency Evaluation
To profile timings, add these environment variables:
```bash
export METHOD_TIME=True     # prune subroutine GPU ms
export PREFILL_TIME=True    # whole pre-fill stage GPU ms
```
Then run accuracy evaluation again, and the raw timings will be appended to `method_times_<method>.txt` and `prefill_times.txt`.


## 📂 File Structure

### File Structure
| File | Purpose |
|------|---------|
| `custom_qwenvl.py` | Monkey-patch for **Qwen2.5-VL**  |
| `custom_internvl.py` | Monkey-patch for **InternVL-3** |
| `run_qwenvl2_5.py` | Entry point for Qwen2.5-VL models |
| `run_internvl3.py` | Entry point for InternVL-3 models |
| `methods/` | Implementation of different pruning methods |
| `utils.py` | `compute_attention_weight()` helper (eager mode) |

## 📊 Evaluation Results



## 🧩 Adding a New Pruning Method

We provide a **decorator-based registry** so that a new pruner can be plugged in **without touching the runner scripts**.  
Below are minimal examples for the two most common insertion points: **Pre-LLM** (right after the ViT) and **Intra-LLM** (inside the language-model stack).  

---

#### 1. Pre-LLM Pruning (affects visual tokens **before** they reach the LLM)

**File:** `methods/Random_PreLLM.py`
```python
import os, torch
from .prune_registry import register_prune

@register_prune('random_pre_llm')          # ← name that will be used in PRUNE_METHOD=
def random_prune_pre_llm(selected, vit_embeds):
    """
    selected   : 1-D bool tensor, length = total #tokens.
                  True for image tokens, False for text/system tokens.
    vit_embeds : Tensor of shape [N_vis, D] – already flattened visual tokens.
    return     : (selected, vit_embeds)  (modified in-place)
    """
    keep_ratio = float(os.environ.get('KEEP_RATIO', 0.5))
    vis_len = vit_embeds.shape[0]
    keep_num = max(1, int(vis_len * keep_ratio))

    perm = torch.randperm(vis_len, device=vit_embeds.device)
    keep_idx = perm[:keep_num].sort()[0]

    vit_embeds = vit_embeds[keep_idx]          # shrink visual sequence
    # map back to the full mask
    image_start = torch.where(selected)[0][0]
    selected[image_start + keep_num :] = False
    return selected, vit_embeds
```

**Run:**
```bash
export PRUNE_METHOD=random_pre_llm
python run_<model>.py --data MME --verbose
```

---

#### 2. Intra-LLM Pruning (prune **inside** the language model)

**File:** `methods/Random_IntraLLM.py`
```python
import os, torch
from .prune_registry import register_prune

@register_prune('random_intra_llm')
def random_prune_intra_llm(visual_features):
    """
    visual_features : [B, L_vis, D]  (already projected to LM dim)
    return          : (pruned_features, keep_idx)
    """
    keep_ratio = float(os.environ.get('KEEP_RATIO', 0.5))
    B, L, D = visual_features.shape
    keep_num = max(1, int(L * keep_ratio))

    keep_idx = torch.randperm(L, device=visual_features.device)[:keep_num].sort()[0]
    return visual_features[:, keep_idx, :], keep_idx
```

**Run:**
```bash
export PRUNE_METHOD=random_intra_llm
python run_<model>.py --data MMBench --verbose
```

---

#### 3. One-Minute Checklist for **Your Own Method**
1. Create `methods/YourMethod_<Stage>.py`.
2. Implement function with the **exact signature** shown above.
3. Decorate with `@register_prune('your_method_<stage>')`.
4. Add an import line to `methods/__init__.py` (or simply `from . import YourMethod_<Stage>` in `prune_registry.py`).
5. Trigger via `export PRUNE_METHOD=your_method_<stage>`.

That’s it—no other files need to be modified.








## 🖊️ Citation

If you find this work helpful, please consider starring 🌟 this repo. Thanks for your support!

```bibtex
@article{peng2025visual,
title={Can Visual Input Be Compressed? A Visual Token Compression Benchmark for Large Multimodal Models},
author={Peng, Tianfan and Du, Yuntao and Ji, Pengzhou and Dong, Shijie and Jiang, Kailin and Ma, Mingchuan and Tian, Yijun and Bi, Jinhe and Li, Qian and Du, Wei and Xiao, Feng and Cui, Lizhen},
journal={arXiv preprint arXiv:2511.02650},
year={2025}
}
```

## 💡 Acknowledgement

We sincerely appreciate the following github repos for their valuable codebase we build upon:

- https://github.com/open-compass/VLMEvalKit
- https://github.com/pkunlp-icler/FastV
- https://github.com/ZichenWen1/DART
- https://github.com/Cooperx521/PyramidDrop
- https://github.com/ywh187/FitPrune
- https://github.com/lzhxmu/VTW
- https://github.com/jytmelon/G-Prune/
- https://github.com/42Shawn/LLaVA-PruMerge
- https://github.com/vbdi/divprune
- https://github.com/liuting20/MustDrop
- https://github.com/Theia-4869/VisPruner

---
