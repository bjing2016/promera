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
        self._last_batch_end = None

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

        cfg = self.cfg
        savedir = cfg.output
        mul = self.cfg.diffusion_samples

        name = batch["name"][0]
        seed_idx = batch["seed_idx"][0]

        n_chains = len(batch["struct"][0].chains)
        n_tokens = int(batch["token_pad_mask"][0].sum())
        n_atoms = int(batch["atom_pad_mask"][0].sum())
        _log(
            f"Running {name} seed{seed_idx}: "
            f"{n_chains} chains, {n_tokens} tokens, {n_atoms} atoms"
        )

        t0 = time.time()
        try:
            out = model.pairformer_forward(batch, recycling_steps=cfg.recycling_steps)
            t1 = time.time()

            os.makedirs(f"{savedir}/{name}", exist_ok=True)
            if cfg.save_distogram:
                np.save(
                    f"{savedir}/{name}/{name}_seed{seed_idx}_distogram.npy",
                    out["pdistogram"][-1].softmax(-1).cpu().numpy(),
                )

            diffusion_out = model.sample_diffusion(batch, out, cfg)
            t2 = time.time()

            all_samples = diffusion_out["sample_atom_coords"].cpu().numpy()
            all_traj = diffusion_out["sample_noisy"].cpu().numpy()
            struct = batch["struct"][0]

            coords = diffusion_out["sample_atom_coords"]
            if model.cfg.model.has_confidence:
                confidences = []
                for i in range(mul):
                    conf = model.sm_confidence_module(
                        batch,
                        out | {"sample_atom_coords": coords[i::mul]},
                        multiplicity=1,
                    )
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
                    contact_out = model.contact_module(
                        batch,
                        out | {"sample_atom_coords": coords[i::mul]},
                        multiplicity=1,
                    )
                    contact_out["contact_logits"] = contact_out["contact_logits"].cpu()
                    contact_out["pred_dist"] = contact_out["pred_dist"].cpu()
                    torch.cuda.empty_cache()
                    contact_outs.append(contact_out)
                t4 = time.time()
            else:
                t4 = t3

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            _log(f"OOM for {name} seed{seed_idx}, skipping")
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
        for n, samp in enumerate(all_samples):
            copy_sample_to_struct(struct, samp)
            struct.to_mmcif(
                f"{savedir}/{name}/{name}_seed{seed_idx}_samp{n}.cif", metadata=True
            )

            conf_data = {}

            if model.cfg.model.has_confidence:
                confidence = confidences[n % mul]
                ntoks = int(batch["token_pad_mask"][n // mul].sum())
                pde = confidence["pde"][n // mul, :ntoks, :ntoks]
                pae = confidence["pae"][n // mul, :ntoks, :ntoks]
                plddt = confidence["plddt"][n // mul, :ntoks]
                pae_logits = confidence["pae_logits"][n // mul, :ntoks, :ntoks].to(
                    pae.device
                )
                frame_mask = batch["frames_mask"][n // mul, :ntoks]
                asym_id = batch["asym_id_"][n // mul]

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
                        f"{savedir}/{name}/{name}_seed{seed_idx}_samp{n}_conf.npz",
                        pde=pde.cpu().numpy(),
                        pae=pae.cpu().numpy(),
                        plddt=plddt.cpu().numpy(),
                        frame_mask=frame_mask.cpu().numpy(),
                        asym_id=asym_id,
                    )

            if getattr(model.cfg.model, "has_contact_module", False):
                contact_out = contact_outs[n % mul]
                ntoks_n = int(batch["token_pad_mask"][n // mul].sum())
                conf_data["iCS"] = compute_contact_stats(
                    contact_out["contact_logits"][n // mul, :ntoks_n, :ntoks_n],
                    contact_out["pred_dist"][n // mul, :ntoks_n, :ntoks_n],
                    batch["asym_id_"][n // mul][:ntoks_n],
                )

            if conf_data:
                conf_data.update(struct.msa_summary)
                with open(
                    f"{savedir}/{name}/{name}_seed{seed_idx}_samp{n}_conf.json", "w"
                ) as f:
                    f.write(json.dumps(conf_data, indent=4))

        t_io_end = time.time()
        _log(
            f"Done {name} seed{seed_idx}: "
            f"pairformer={t1-t0:.1f}s  diffusion={t2-t1:.1f}s  "
            f"confidence={t3-t2:.1f}s  contact={t4-t3:.1f}s  "
            f"write={t_io_end-t_io_start:.1f}s  total={t_io_end-t0:.1f}s"
        )
        self._last_batch_end = t_last_gpu

        if cfg.save_traj:
            steps = np.arange(0, all_traj.shape[1], 10)
            for n, traj in enumerate(all_traj):
                i = 0
                for key, chain in struct.chains.items():
                    chain._models = np.zeros((len(steps), *chain.coords.shape))
                    for j, aname in enumerate(struct.chains[key].aname):
                        for k, c in enumerate(aname):
                            if c != "":
                                chain._models[:, j, k] = traj[steps, i]
                                i += 1
                struct.to_mmcif(
                    f"{savedir}/{name}/{name}_seed{seed_idx}_samp{n}_traj.cif",
                    models=[
                        {k: struct.chains[k]._models[i] for k in struct.chains}
                        for i, _ in enumerate(steps)
                    ],
                )
