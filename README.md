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

Sequence redesign after backbone diffusion requires [LigandMPNN](https://github.com/dauparas/LigandMPNN).

```bash
git clone https://github.com/dauparas/LigandMPNN ../LigandMPNN
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

## License

MIT. Other licenses may apply to third-party source code noted in comments / file headers.

## Acknowledgements

Parts of this package were started from https://github.com/jwohlwend/boltz/, https://github.com/lucidrains/alphafold3-pytorch/, and https://github.com/aqlaboratory/openfold/. A big thanks to the developers of these open-source libraries.
