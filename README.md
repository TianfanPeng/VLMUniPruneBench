# UniPruneBench: A Visual Input Token Compression Benchmark for Large Multimodal Models


UniPruneBench is an open-source benchmark for evaluating visual input token compression in large multimodal models (LMMs). It integrates with VLMEvalKit to enable efficient pruning evaluations on various datasets, focusing on methods like DART, Pdrop, Fastv, FitPrune and so on.

## Recent Codebase Changes
- **[2025-10-14]** Initial release. 

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

### Run a Single-Dataset Evaluation
Evaluate on a benchmark like MME with 88.9% pruning (keeping 11.1% of tokens).

#### Qwen2.5-VL Example
```bash
export KEEP_RATIO=0.111       
export PRUNE_METHOD_PRE_LLM=divprune_pre_llm

python run_qwenvl2_5.py \
    --model Qwen2.5-VL-7B-Instruct \
    --data MME \
    --verbose
```

#### InternVL-3 Example
```bash
export KEEP_RATIO=0.111
export PRUNE_METHOD_INTRA_LLM=fitprune_intra_llm

python run_internvl3.py \
    --model InternVL3-8B \
    --data MME \
    --verbose
```

## 📊 File Structure, Models, and Evaluation Results

### File Structure
| File | Purpose |
|------|---------|
| `custom_qwenvl.py` | Monkey-patch for **Qwen2.5-VL**  |
| `custom_internvl.py` | Monkey-patch for **InternVL-3** |
| `run_qwenvl2_5.py` | Entry point – swaps forward hooks and launches VLMEvalKit |
| `run_internvl3.py` | Same for InternVL models |
| `utils.py` | `compute_attention_weight()` helper (eager mode) |


#### Time Profiling
To profile timings, add these environment variables:
```bash
export METHOD_TIME=True     # prune subroutine GPU ms
export PREFILL_TIME=True    # whole pre-fill stage GPU ms
```
Raw timings are appended to `method_times_<method>.txt` and `prefill_times.txt`.



## 🎯 The Goal of UniPruneBench

**The benchmark is designed to:**
1. Provide an **easy-to-use**, **open-source toolkit** for evaluating token pruning strategies in large multimodal models, making results reproducible and comparable.
2. Enable developers to test pruning methods on benchmarks with minimal setup – just specify the method and keep ratio, and the codebase handles integration with VLMEvalKit.

**The benchmark is not designed to:**
1. Reproduce exact numerical results from pruning papers without adaptation, due to differences in hooks, pruning settings, and environments.

## 🖊️ Citation

If you find this work helpful, please consider starring 🌟 this repo. Thanks for your support!

---