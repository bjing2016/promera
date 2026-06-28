import argparse
import os

_DEFAULT_MODEL_CONFIG = os.path.join(os.path.dirname(__file__), "model/config.yaml")
_DEFAULT_TASK_CONFIG = os.path.join(
    os.path.dirname(__file__), "inference/cofolding.yaml"
)

parser = argparse.ArgumentParser()
parser.add_argument("--model_config", type=str, default=_DEFAULT_MODEL_CONFIG)
parser.add_argument("--task_config", type=str, default=_DEFAULT_TASK_CONFIG)
parser.add_argument("--weights", type=str, default=os.environ.get("PROMERA_WEIGHTS"))
parser.add_argument(
    "--task",
    type=str,
    default="promera.inference.Cofolding",
    help="Dotted path to an inference task class",
)
args, extra = parser.parse_known_args()

from omegaconf import OmegaConf

model_cfg = OmegaConf.merge(
    OmegaConf.load(args.model_config), OmegaConf.from_cli(extra)
)
task_cfg = OmegaConf.merge(OmegaConf.load(args.task_config), OmegaConf.from_cli(extra))

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import functools
from promera.data.utils import collate
from promera.utils.load_weights import load_weights


def _get_attr_from_path(path):
    import importlib

    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


model_cfg.trainer.devices = int(
    os.environ.get("SLURM_NTASKS_PER_NODE", model_cfg.trainer.devices)
)
torch.set_float32_matmul_precision(model_cfg.set_float32_matmul_precision)

# Build the task up front so we can decide the scheduling strategy before the
# Trainer is constructed. A task opts into the dynamic work-queue scheduler (big
# jobs first, ranks pull more work as they finish) by implementing
# make_batch_sampler; when it does, we hand Lightning our own batch sampler and
# disable its static DistributedSampler. Everything still runs under Lightning's
# DDP across GPUs/nodes — only the index scheduling changes.
task_cls = _get_attr_from_path(args.task)
task = task_cls(task_cfg)

dynamic_schedule = bool(task_cfg.get("dynamic_schedule", False)) and hasattr(
    task, "make_batch_sampler"
)

num_nodes = int(os.environ.get("SLURM_NNODES", 1))
trainer_kwargs = OmegaConf.to_container(model_cfg.trainer, resolve=True)
world_size = num_nodes * int(trainer_kwargs.get("devices", 1) or 1)
if world_size > 1:
    # Explicit DDP with a generous collective timeout. Inference here is
    # embarrassingly parallel (each rank writes to disk, predict_step returns
    # None), so ranks finish at very different times. The end-of-run barrier below
    # waits out that legitimate imbalance, but the timeout still bounds a genuinely
    # stuck rank so it aborts (and SLURM reaps the step) instead of hanging the
    # node allocation forever.
    from datetime import timedelta
    from pytorch_lightning.strategies import DDPStrategy

    trainer_kwargs["strategy"] = DDPStrategy(timeout=timedelta(minutes=60))

trainer = pl.Trainer(
    **trainer_kwargs,
    num_nodes=num_nodes,
    use_distributed_sampler=not dynamic_schedule,
)

meta_init = bool(model_cfg.get("meta_init", False))
if meta_init:
    # Build on meta (no allocation, no weight init) then fill every tensor from
    # the checkpoint via an assign-load — the fastest path to inference.
    with torch.device("meta"):
        model = _get_attr_from_path(model_cfg.model._target_)(model_cfg)
else:
    model = _get_attr_from_path(model_cfg.model._target_)(model_cfg)

load_weights(args.weights, model, assign=meta_init)

if meta_init:
    stranded = [
        n
        for n, t in (*model.named_parameters(), *model.named_buffers())
        if t.is_meta
    ]
    if stranded:
        raise RuntimeError(
            f"meta_init=true but {len(stranded)} tensor(s) were not provided by "
            f"the checkpoint and remain on the meta device, e.g. {stranded[:5]}. "
            f"Set meta_init=false for this checkpoint/config."
        )

model.inference_task = task

batch_size = int(task_cfg.get("batch_size", 1))
# Round the padded token count up to this multiple so a compiled, token-
# dimensioned denoiser (compile_score) sees a stable shape across batches and
# stops recompiling on every new size (1 = off; see collate / README).
_token_multiple = int(task_cfg.get("pad_tokens_to_multiple", 1) or 1)
_collate_fn = functools.partial(collate, token_multiple=_token_multiple)
if dynamic_schedule:
    # world_size/rank are only used by the sampler to decide whether it is safe
    # to wipe a stale queue (single process) and for logging — the claim queue
    # itself is global and self-coordinating, so accuracy here is not critical.
    world_size = int(
        os.environ.get("SLURM_NTASKS", trainer.num_devices * trainer.num_nodes)
    )
    rank = int(os.environ.get("SLURM_PROCID", 0))
    loader = DataLoader(
        task,
        batch_sampler=task.make_batch_sampler(world_size, rank, batch_size),
        collate_fn=_collate_fn,
        num_workers=model_cfg.data.workers,
    )
else:
    loader = DataLoader(
        task,
        batch_size=batch_size,
        collate_fn=_collate_fn,
        num_workers=model_cfg.data.workers,
    )

# return_predictions=False: tasks write outputs to disk and predict_step returns
# None, so the default cross-rank gather of predictions at the end is wasted work
# and an extra collective that can hang.
trainer.predict(model, loader, return_predictions=False)

# Coordinated DDP teardown. Without this, the first rank to finish returns from
# predict and its process exits, killing the TCPStore it hosts; still-working ranks
# then hit "Broken pipe" / NCCL heartbeat errors and wedge (GPUs idle, never
# exiting). Barrier so every rank rendezvous before ANY teardown, then destroy the
# process group collectively.
import torch.distributed as dist

if dist.is_available() and dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()
