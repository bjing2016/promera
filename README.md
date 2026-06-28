<div align="center">

# Promera

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/weights-HuggingFace-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/bjing-mit/promera/)
[![tinyprot](https://img.shields.io/badge/built%20on-tinyprot-2ea44f)](https://github.com/bjing2016/tinyprot)

Promera is a dual-purpose biomolecular generative model for structure prediction and binder design, with advanced filtering abilities for _de novo_ design pipelines. Please see our preprint for detailed benchmarking and case studies.

Promera is built upon [tinyprot](https://github.com/bjing2016/tinyprot), a lightweight library for biomolecular structure processing.

<img src="promera.png" width="500">

</div>

---

## Installation

1. Install the packages and dependencies.
```bash
pip install git+https://github.com/bjing2016/promera.git
```
For development, you can clone the repo and use [uv](https://docs.astral.sh/uv/) for a pinned and reproducible environment:
```bash
git clone https://github.com/bjing2016/promera.git && cd promera
uv sync
```

2. Follow the initialization instructions in [tinyprot](https://github.com/bjing2016/tinyprot).

3. Download the [weights from HuggingFace](https://huggingface.co/bjing-mit/promera/resolve/main/promera_2606.ckpt) and point `PROMERA_WEIGHTS` to the weights path.

---

## Running structure prediction

1. Prepare a directory of JSON schemas; see [`examples/`](examples/) or the [tinyprot](https://github.com/bjing2016/tinyprot) docs.

2. Fetch MSAs to the global cache
```bash
python -m tinyprot.mmseqs2 --url https://api.colabfold.com --input [schema_dir]
# or with local MSA server or database
```

3. Run inference
```bash
python -m promera input=path/to/schemas/ output=out/
```

Command line arguments (`arg=value`) can be provided to override any of the settings in the [inference config](promera/inference/cofolding.yaml) or (rarely necessary) the [model config](promera/model/config.yaml).

### MSA checks

> [!NOTE]
> By default, the script will raise errors for protein chains without MSAs pre-fetched by `tinyprot`. Set `assert_msa=False` to disable this behavior and (optionally) include per-chain `use_msa: true` and `use_msa: false` in the schema JSON for fine-grained control.

### Parallelization

> [!TIP]
> If launched with `srun`, jobs will automatically parallelize over nodes and GPUs.

---

## Custom inference workflows

Promera is architected to easily support custom inference workflows without needing to clone and modify the repository. The main entrypoint accepts `--task` and `--task_config` command line variables, which point to `promera.inference.Cofolding` by default.

To run custom inference, pass any `--task <module.path.TaskClass>` and associated `--task_config <path.to.yaml>`. The class must implement `__init__(cfg)`, `__getitem__`, `__len__`, and `run_batch(model, batch)`. See [`promera/inference/cofolding.py`](promera/inference/cofolding.py) for a reference.

---

## Binder design

De novo binder design is built into Promera via the `promera.inference.Design` task. It supports both free minibinder design and VHH nanobody design with framework conditioning.

### Setup: LigandMPNN

Sequence redesign after backbone diffusion requires LigandMPNN. Use the promera-compatible fork [`bjing2016/LigandMPNN`](https://github.com/bjing2016/LigandMPNN), which adds an in-memory, GPU-batched redesign entrypoint (`design_batched`) and caches the loaded model across calls — the design loop scores every backbone in a batch in one pass, with no per-backbone PDB round-trip or checkpoint reload.

```bash
git clone https://github.com/bjing2016/LigandMPNN ../LigandMPNN
cd ../LigandMPNN && bash get_model_params.sh ./model_params
export LIGANDMPNN_DIR=$(pwd)
```

### Optional: AbMPNN for VHH design

For antibody/nanobody inverse folding, Promera can use AbMPNN weights. Download the AbMPNN checkpoint:

```bash
wget "https://zenodo.org/records/8164693/files/abmpnn.pt?download=1" \
    -O "$LIGANDMPNN_DIR/model_params/abmpnn.pt"
```

Then set:

```bash
export ABMPNN_CHECKPOINT="$LIGANDMPNN_DIR/model_params/abmpnn.pt"
```

To use AbMPNN in a design workflow, set the inverse folding type in your design YAML:

```yaml
inverse_folder:
  type: abmpnn
  num_seqs: 1
```

### Running design

1. Prepare target schemas; see [`examples/targets/`](examples/targets/) for examples.

2. Fetch MSAs as in structure prediction.

3. Run backbone diffusion + sequence redesign:

```bash
# Minibinder design
python -m promera \
    --task promera.inference.Design \
    --task_config examples/design_minibinder.yaml \
    input=examples/targets/ output=out/
```

```bash
# VHH nanobody design
python -m promera \
    --task promera.inference.Design \
    --task_config examples/design_vhh.yaml \
    input=examples/targets/ output=out/
```

Copy and edit [`examples/design_minibinder.yaml`](examples/design_minibinder.yaml) or [`examples/design_vhh.yaml`](examples/design_vhh.yaml) for your design setting.

---

## Optimizing performance

The defaults are conservative (fp32, one target at a time). The knobs below speed up inference substantially on a Hopper GPU (H100) and **preserve the prediction** (bf16 differs from fp32 by no more than the run-to-run spread across diffusion samples). The diffusion roll-out dominates runtime, so most of the speedup targets it.

Recommended settings, in priority order — **(1) raise `batch_size` until the GPU saturates, (2) `amp=bf16`, (3) `compile_score` + `pad_tokens_to_multiple`**; skip `cudagraph_score`:

```bash
python -m promera input=schemas/ output=out/ \
    batch_size=16 amp=bf16 \
    model.structure_module_args.compile_score=true \
    pad_tokens_to_multiple=32 \
    meta_init=true          # skip init, load straight from the checkpoint
```

- **`batch_size=N`** — fold/design N items per forward pass; the main lever for GPU utilization (`batch_size=1` leaves an H100 idle). Items are padded to the batch's largest size, so `sort_by_size=true` (default) groups similar sizes; pick N to fit memory (an OOM batch is skipped with a warning, not a crash). Results are independent of `batch_size` up to normal nondeterminism.
- **`amp=bf16`** — runs the GPU forward passes in bfloat16 (the single biggest lever). bf16 is stable here; **fp16 overflows — avoid it.** `amp_diffusion` overrides the diffusion precision only. Use this flag, **not** Lightning's `precision=bf16-mixed`: the diffusion sampler opts out of autocast unless `amp`/`amp_diffusion` is set, so a blanket Lightning autocast collides with it and crashes.
- **`compile_score=true`** — `torch.compile`s the diffusion token transformer. It is keyed on tensor shape, so each new padded token count triggers a ~60 s recompile; pair it with **`pad_tokens_to_multiple=32`** to round padded sizes to a shared shape (compile once instead of per size). Worth it for a narrow size spread (e.g. binder design); for a broad screen, bucket sizes and set `TORCHINDUCTOR_CACHE_DIR` on a shared path.
- **`cudagraph_score`** *(not recommended)* — only helps the launch-bound `batch_size=1` case and does nothing at saturation; prefer raising `batch_size`.
- **`meta_init=true`** — builds the model on the `meta` device and loads weights directly from the checkpoint (fastest startup). Requires the checkpoint to cover 100% of parameters/buffers; keep `false` for training.

For **design**, `batch_size` + `amp=bf16` is the recommendation — backbone generation, inverse folding (one in-memory LigandMPNN pass with a cached model, no PDB round-trip), and refold all batch across the backbones in a batch. `compile_score`/`cudagraph_score` aren't worth it; a large fraction of each design is CPU work (the scRMSD/DockQ/CIF metrics), so end-to-end gains are smaller than the per-phase GPU speedups.

---

## License

MIT. Other licenses may apply to third-party source code noted in comments / file headers.

## Acknowledgements

Parts of this package were started from https://github.com/jwohlwend/boltz/, https://github.com/lucidrains/alphafold3-pytorch/, and https://github.com/aqlaboratory/openfold/. A big thanks to the developers of these open-source libraries.
