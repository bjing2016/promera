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

Defaults are conservative (fp32, one target at a time). These knobs are opt-in and preserve the prediction (bf16 stays within the run-to-run diffusion-sample spread). Recommended, in priority order:

```bash
python -m promera input=schemas/ output=out/ \
    batch_size=16 amp=bf16 \
    model.structure_module_args.compile_score=true \
    pad_tokens_to_multiple=32 \
    meta_init=true
```

- **`batch_size=N`** — fold/design N items per forward pass; the main utilization lever. Items pad to the batch's largest size, so `sort_by_size=true` (default) groups similar sizes; an out-of-memory batch is skipped with a warning. Results are independent of `batch_size` up to normal nondeterminism.
- **`amp=bf16`** — bf16 forward passes, the single biggest lever (`amp_diffusion` overrides diffusion only). **fp16 overflows — avoid it.** Use this flag, not Lightning's `precision=bf16-mixed`, which collides with the diffusion sampler and crashes.
- **`compile_score=true`** — compiles the diffusion token transformer. Keyed on shape, so pair with **`pad_tokens_to_multiple=32`** to share one compiled shape across a size spread; set `TORCHINDUCTOR_CACHE_DIR` to reuse across runs.
- **`cudagraph_score`** *(not recommended)* — only helps `batch_size=1`; prefer raising `batch_size`.
- **`meta_init=true`** — loads weights straight from the checkpoint (fastest startup); requires a complete checkpoint, so keep `false` for training.

For **design**, `batch_size` + `amp=bf16` is the recommendation: backbone generation, inverse folding (one in-memory LigandMPNN pass with a cached model, no PDB round-trip), and refold all batch across the backbones. `compile_score`/`cudagraph_score` aren't worth it, and a large fraction of each design is CPU-bound metrics regardless.

---

## License

MIT. Other licenses may apply to third-party source code noted in comments / file headers.

## Acknowledgements

Parts of this package were started from https://github.com/jwohlwend/boltz/, https://github.com/lucidrains/alphafold3-pytorch/, and https://github.com/aqlaboratory/openfold/. A big thanks to the developers of these open-source libraries.
