from tinyprot.structure import Structure
from tinyprot.feature import AF3Featurizer
from tinyprot.msa import load_msa_from_dir, construct_paired_msa, make_dummy_msa
import numpy as np
import json
import os
import time
import torch
from .utils import (
    finalize_feats,
    compute_agg_confidence,
    compute_contact_stats,
    msa_summary,
)


def _log(msg):
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    print(f"[{time.strftime('%H:%M:%S')} rank{rank}] {msg}", flush=True)


def _to_fp32(x):
    """Recursively cast floating tensors in a dict/list tree back to fp32.

    The trunk runs under autocast (fp16/bf16) when amp is set, so its outputs
    must be promoted before the fp32 diffusion path consumes them (s/z/s_inputs)."""
    if torch.is_tensor(x):
        return x.float() if x.is_floating_point() else x
    if isinstance(x, dict):
        return {k: _to_fp32(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_fp32(v) for v in x]
    return x


class Cofolding:
    def __init__(self, cfg):
        self.cfg = cfg
        split = [f[:-5] for f in os.listdir(cfg.input) if f.endswith(".json")]
        mul = cfg.diffusion_samples
        savedir = cfg.output
        self.items = []
        for seed_idx in range(cfg.num_seeds):
            for name in split:
                if cfg.skip_existing and os.path.exists(
                    f"{savedir}/{name}/{name}_seed{seed_idx}_samp{mul-1}.cif"
                ):
                    _log(f"Skipping {name} seed{seed_idx}")
                    continue
                self.items.append((name, seed_idx))

        # With batch_size > 1, items in a batch are padded to the batch's largest
        # token/atom count, so mixing very different sizes wastes compute. Sorting
        # items by size groups similar-sized targets into the same batch, which
        # keeps the GPU efficiently saturated. We sort *descending* (largest
        # first): under the dynamic scheduler this dispatches the longest jobs
        # before the short ones (longest-processing-time greedy), so big targets
        # don't become end-of-run stragglers while still being size-grouped.
        # Order-only change (results are unaffected); opt out with
        # sort_by_size=false.
        if cfg.get("sort_by_size", True):
            self.items.sort(key=lambda it: self._schema_size(it[0]), reverse=True)

        self._last_batch_end = None

    def make_batch_sampler(self, world_size, rank, batch_size):
        """Return a shared work-queue batch sampler (see schedule.py).

        Opting into this (the default, via dynamic_schedule in the task config)
        replaces Lightning's static DistributedSampler with dynamic claim-as-you-
        finish scheduling across all ranks/nodes. The claim queue lives under the
        output dir, keyed by the SLURM job id so concurrent jobs don't collide and
        a re-run (new job id) starts a fresh queue. skip_existing still handles
        already-finished targets, so the queue only ever covers this run's work."""
        from .schedule import DynamicClaimBatchSampler

        token = os.environ.get("SLURM_JOB_ID", "local")
        claim_dir = os.path.join(self.cfg.output, f".promera_schedule_{token}")
        return DynamicClaimBatchSampler(
            len(self.items), batch_size, claim_dir, world_size, rank
        )

    def _schema_size(self, name):
        """Cheap token-count estimate for a target (sum of polymer/ligand
        sequence lengths from the schema JSON) used to group similar-sized
        targets into the same batch."""
        try:
            with open(f"{self.cfg.input}/{name}.json") as f:
                schema = json.loads(f.read())
        except OSError:
            return 0
        total = 0
        for key, val in schema.items():
            if key == "connections" or not isinstance(val, dict):
                continue
            total += len(val.get("sequence", ""))
        return total

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cfg = self.cfg
        name, seed_idx = self.items[idx]

        t0 = time.time()
        with open(f"{cfg.input}/{name}.json") as f:
            schema = json.loads(f.read())
        struct = Structure.from_schema(schema)

        msas = load_msa_from_dir(cfg.msa_dir, struct.chains)

        for chain_id, chain in struct.chains.items():
            use_msa = schema.get(chain_id, {}).get("use_msa", None)
            if use_msa is False:
                msas[chain_id] = make_dummy_msa(chain.seq)
            elif use_msa is True:
                if "polypeptide" in chain.type and msas[chain_id].path is None:
                    raise ValueError(f"{name}: chain {chain_id} has no MSA")

        if getattr(cfg, "assert_msa", False):
            for chain_id, chain in struct.chains.items():
                if "polypeptide" in chain.type and msas[chain_id].path is None:
                    raise ValueError(f"{name}: chain {chain_id} has no MSA")

        pairing = construct_paired_msa(msas)
        feats = AF3Featurizer(struct, msas, pairing).featurize(compute_frames=True)

        # Carry per-chain MSA depth/path on the struct (batch size is always 1,
        # so this rides through to run_batch without a collated feature key).
        struct.msa_summary = msa_summary(msas)

        _log(f"__getitem__ {name} seed{seed_idx}: {time.time()-t0:.2f}s")

        return finalize_feats(feats, struct, name, seed_idx)

    def run_batch(self, model, batch):
        import contextlib

        cfg = self.cfg
        savedir = cfg.output
        mul = self.cfg.diffusion_samples

        # Optional mixed precision (cfg.amp = "fp16"/"bf16"). The trunk
        # (pairformer) dominates wall-clock on larger targets, so autocasting it
        # — plus confidence/contact — is the main throughput lever on Hopper.
        # Diffusion picks up the same setting via cfg.amp inside sample_diffusion.
        amp = cfg.get("amp", None)
        amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
        amp_ctx = (
            torch.autocast("cuda", dtype=amp_dtype)
            if amp_dtype is not None
            else contextlib.nullcontext()
        )

        # batch_size may be > 1: every batch element is an independent target.
        # The model forward (pairformer / diffusion / confidence / contact) is
        # fully batched over B; the per-element bookkeeping (struct, name, seed,
        # output paths) is resolved in the write loop via b = n // mul.
        B = len(batch["name"])

        for b in range(B):
            n_chains = len(batch["struct"][b].chains)
            n_tokens = int(batch["token_pad_mask"][b].sum())
            n_atoms = int(batch["atom_pad_mask"][b].sum())
            _log(
                f"Running {batch['name'][b]} seed{batch['seed_idx'][b]}: "
                f"{n_chains} chains, {n_tokens} tokens, {n_atoms} atoms"
            )

        t0 = time.time()
        try:
            with amp_ctx:
                out = model.pairformer_forward(
                    batch, recycling_steps=cfg.recycling_steps
                )
            # Promote trunk outputs to fp32 for the fp32 diffusion path; the
            # confidence/contact passes re-autocast from these fp32 inputs.
            if amp_dtype is not None:
                out = _to_fp32(out)
            t1 = time.time()

            if cfg.save_distogram:
                pdist = out["pdistogram"][-1].softmax(-1).cpu().numpy()
                for b in range(B):
                    name = batch["name"][b]
                    seed_idx = batch["seed_idx"][b]
                    ntoks_b = int(batch["token_pad_mask"][b].sum())
                    os.makedirs(f"{savedir}/{name}", exist_ok=True)
                    np.save(
                        f"{savedir}/{name}/{name}_seed{seed_idx}_distogram.npy",
                        pdist[b, :ntoks_b, :ntoks_b],
                    )

            diffusion_out = model.sample_diffusion(batch, out, cfg)
            t2 = time.time()

            all_samples = diffusion_out["sample_atom_coords"].float().cpu().numpy()
            all_traj = diffusion_out["sample_noisy"].float().cpu().numpy()

            coords = diffusion_out["sample_atom_coords"]
            if model.cfg.model.has_confidence:
                confidences = []
                for i in range(mul):
                    with amp_ctx:
                        conf = model.sm_confidence_module(
                            batch,
                            out | {"sample_atom_coords": coords[i::mul]},
                            multiplicity=1,
                        )
                    # Cast metric tensors back to fp32 so the aggregate
                    # confidence math is precision-independent under amp.
                    conf = {
                        k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                        for k, v in conf.items()
                    }
                    conf["pae_logits"] = conf["pae_logits"].cpu()
                    conf["pde_logits"] = conf["pde_logits"].cpu()
                    torch.cuda.empty_cache()
                    confidences.append(conf)
                t3 = time.time()
            else:
                t3 = t2

            if getattr(model.cfg.model, "has_contact_module", False):
                contact_outs = []
                for i in range(mul):
                    with amp_ctx:
                        contact_out = model.contact_module(
                            batch,
                            out | {"sample_atom_coords": coords[i::mul]},
                            multiplicity=1,
                        )
                    contact_out["contact_logits"] = (
                        contact_out["contact_logits"].float().cpu()
                    )
                    contact_out["pred_dist"] = contact_out["pred_dist"].float().cpu()
                    torch.cuda.empty_cache()
                    contact_outs.append(contact_out)
                t4 = time.time()
            else:
                t4 = t3

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            names = ", ".join(str(batch["name"][b]) for b in range(B))
            _log(f"OOM for batch [{names}], skipping")
            return

        def copy_sample_to_struct(struct, samp):
            i = 0
            for key, chain in struct.chains.items():
                for j, aname in enumerate(struct.chains[key].aname):
                    for k, c in enumerate(aname):
                        if c != "":
                            chain.coords[j, k] = samp[i]
                            chain.mask[j, k] = True
                            i += 1

        t_last_gpu = t4
        t_io_start = time.time()
        # all_samples is laid out [B * mul] as b * mul + s (repeat_interleave),
        # so b = n // mul selects the target and s = n % mul the sample index.
        for n, samp in enumerate(all_samples):
            b = n // mul
            s = n % mul
            struct = batch["struct"][b]
            name = batch["name"][b]
            seed_idx = batch["seed_idx"][b]
            ntoks = int(batch["token_pad_mask"][b].sum())

            os.makedirs(f"{savedir}/{name}", exist_ok=True)
            copy_sample_to_struct(struct, samp)
            struct.to_mmcif(
                f"{savedir}/{name}/{name}_seed{seed_idx}_samp{s}.cif", metadata=True
            )

            conf_data = {}

            if model.cfg.model.has_confidence:
                confidence = confidences[s]
                pde = confidence["pde"][b, :ntoks, :ntoks]
                pae = confidence["pae"][b, :ntoks, :ntoks]
                plddt = confidence["plddt"][b, :ntoks]
                pae_logits = confidence["pae_logits"][b, :ntoks, :ntoks].to(pae.device)
                frame_mask = batch["frames_mask"][b, :ntoks]
                asym_id = batch["asym_id_"][b]

                agg_conf = compute_agg_confidence(
                    pde=pde,
                    pae=pae,
                    plddt=plddt,
                    pae_logits=pae_logits,
                    asym_id=asym_id,
                    frame_mask=frame_mask,
                    use_torch=True,
                )
                conf_data.update(agg_conf)

                if cfg.save_full_confidence:
                    np.savez(
                        f"{savedir}/{name}/{name}_seed{seed_idx}_samp{s}_conf.npz",
                        pde=pde.cpu().numpy(),
                        pae=pae.cpu().numpy(),
                        plddt=plddt.cpu().numpy(),
                        frame_mask=frame_mask.cpu().numpy(),
                        asym_id=asym_id,
                    )

            if getattr(model.cfg.model, "has_contact_module", False):
                contact_out = contact_outs[s]
                conf_data["iCS"] = compute_contact_stats(
                    contact_out["contact_logits"][b, :ntoks, :ntoks],
                    contact_out["pred_dist"][b, :ntoks, :ntoks],
                    batch["asym_id_"][b][:ntoks],
                )

            if conf_data:
                conf_data.update(struct.msa_summary)
                with open(
                    f"{savedir}/{name}/{name}_seed{seed_idx}_samp{s}_conf.json", "w"
                ) as f:
                    f.write(json.dumps(conf_data, indent=4))

        t_io_end = time.time()
        _log(
            f"Done batch ({B} target(s)): "
            f"pairformer={t1-t0:.1f}s  diffusion={t2-t1:.1f}s  "
            f"confidence={t3-t2:.1f}s  contact={t4-t3:.1f}s  "
            f"write={t_io_end-t_io_start:.1f}s  total={t_io_end-t0:.1f}s"
        )
        self._last_batch_end = t_last_gpu

        if cfg.save_traj:
            steps = np.arange(0, all_traj.shape[1], 10)
            for n, traj in enumerate(all_traj):
                b = n // mul
                s = n % mul
                struct = batch["struct"][b]
                name = batch["name"][b]
                seed_idx = batch["seed_idx"][b]
                i = 0
                for key, chain in struct.chains.items():
                    chain._models = np.zeros((len(steps), *chain.coords.shape))
                    for j, aname in enumerate(struct.chains[key].aname):
                        for k, c in enumerate(aname):
                            if c != "":
                                chain._models[:, j, k] = traj[steps, i]
                                i += 1
                struct.to_mmcif(
                    f"{savedir}/{name}/{name}_seed{seed_idx}_samp{s}_traj.cif",
                    models=[
                        {k: struct.chains[k]._models[i] for k in struct.chains}
                        for i, _ in enumerate(steps)
                    ],
                )
