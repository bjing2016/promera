"""De novo binder generation via the Promera cofolding model.

Given a target schema (.json), generate a binder by:
  1. Building a schema with the target chains plus a binder placeholder
     (all-X for `binder.type=protein`, framework + variable-length CDR Xs for `vhh`).
  2. (Optional) Flagging epitope or paratope residues via feats["is_epitope"].
  3. Running the cofolding pass (pairformer → diffusion → confidence).
  4. (Optional) Inverse-folding the binder backbone with LigandMPNN /
     ProteinMPNN / SolubleMPNN / AbMPNN and re-folding each redesigned sequence.

Run:
    python -m promera \\
        --task promera.inference.Design \\
        --task_config examples/diffusion_vhh.yaml \\
        --weights $PROMERA_WEIGHTS \\
        input=targets/ output=out/
"""

import copy
import json
import os
import random
import re
import time

import numpy as np
import torch

from tinyprot.feature import AF3Featurizer
from tinyprot.msa import construct_paired_msa, load_msa_from_dir
from tinyprot.structure import Structure

from promera.data.utils import collate
from .utils import (
    _AA3TO1,
    _copy_sample_to_struct,
    _resolve_residue_idx,
    _struct_to_pdb,
    compute_agg_confidence,
    compute_contact_stats,
    compute_dockq,
    compute_interface_contacts,
    compute_self_consistency_rmsd,
    compute_target_lddt,
    finalize_feats,
    msa_summary,
    run_lmpnn_redesign,
)

_INPUT_EXT = ".json"

_MPNN_VARIANTS = {
    "proteinmpnn": "protein_mpnn",
    "solublempnn": "soluble_mpnn",
    "ligandmpnn": "ligand_mpnn",
    "abmpnn": "ab_mpnn"
}

def _log(msg):
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    print(f"[{time.strftime('%H:%M:%S')} rank{rank}] {msg}", flush=True)


def _build_binder_chain(binder_cfg, rng=None) -> dict:
    """Return the binder chain entry to splice into the schema.

    Protein: all-X of `length`.
    VHH: framework string with `<tag>` placeholders replaced by random-length X
    runs, where each tag's length range comes from `cdr_lengths[tag]`. A range
    can be `[min, max]` or a single int for a fixed length. `rng` is a
    `random.Random` instance for reproducible sampling per backbone.
    """
    rng = rng or random
    btype = binder_cfg.type
    if btype == "protein":
        spec = binder_cfg.length
        n = spec if isinstance(spec, int) else rng.randint(spec[0], spec[1])
        return {"type": "protein", "sequence": "X" * n}
    if btype == "vhh":
        fw = binder_cfg.framework
        cdr_lengths = binder_cfg.get("cdr_lengths", {}) or {}

        def _replace(match):
            tag = match.group(1)
            if tag not in cdr_lengths:
                raise ValueError(
                    f"<{tag}> in framework has no entry in binder.cdr_lengths"
                )
            spec = cdr_lengths[tag]
            n = spec if isinstance(spec, int) else rng.randint(spec[0], spec[1])
            return "X" * n

        seq = re.sub(r"<(\w+)>", _replace, fw)
        return {"type": "protein", "sequence": seq}
    raise ValueError(f"unknown binder.type: {btype}")


def _inverse_fold(
    pdb_path: str, binder_chain: str, ifold_cfg, binder_seq: str, lmpnn_dir: str
) -> list:
    """Run sequence redesign on the binder chain.

    binder_seq is the resolved binder sequence (X marks positions to design).
    Non-X positions are passed to LigandMPNN as fixed.
    """
    ftype = ifold_cfg.type
    if ftype == "none":
        return []
    fixed = " ".join(
        f"{binder_chain}{i+1}" for i, aa in enumerate(binder_seq) if aa != "X"
    )
    if ftype in _MPNN_VARIANTS:
        return run_lmpnn_redesign(
            pdb_path,
            binder_chain,
            lmpnn_dir,
            num_seqs=ifold_cfg.num_seqs,
            model_type=_MPNN_VARIANTS[ftype],
            fixed_residues=fixed,
        )
    if ftype == "abmpnn":
        raise NotImplementedError("abmpnn inverse folder — TODO")
    raise ValueError(f"unknown inverse_folder.type: {ftype}")


def _build_schema_for_input(
    input_path: str, binder_cfg, target_chains=None, rng=None
) -> dict:
    """Load a tinyprot schema JSON and append the binder chain."""
    binder_chain = binder_cfg.chain
    binder_entry = _build_binder_chain(binder_cfg, rng=rng)

    with open(input_path) as f:
        schema = json.load(f)
    if target_chains:
        schema = {
            k: v for k, v in schema.items() if k == "connections" or k in target_chains
        }
    if binder_chain in schema:
        raise ValueError(
            f"binder.chain '{binder_chain}' conflicts with a chain in {input_path}. "
            f"Pick a different binder.chain in the task config."
        )
    schema[binder_chain] = binder_entry
    return schema


def _resolve_epitope_idx(
    schema: dict, epitope_chain: str, epitope_resnums: list
) -> list:
    """Resolve 1-based residue numbers to 0-based token indices in the schema."""
    if not epitope_resnums or not epitope_chain:
        return []
    return _resolve_residue_idx(schema, epitope_chain, epitope_resnums, resnum_map=None)


def _compute_framework_distogram(
    template_path,
    template_chain_id,
    framework_str,
    schema,
    binder_chain,
    n_tokens,
):
    """Build (distogram_emb, distogram_mask) numpy arrays for VHH framework conditioning.

    Maps each binder framework position to the corresponding Cα coord of a reference
    VHH structure by exact-substring matching the framework parts (text between
    <cdrh*> tags) against the template chain sequence. Only framework-framework
    pairs in the binder chain receive distogram entries; everything else stays masked.
    Bins match the model's training: torch.linspace(1, 50, 50), bin index 0..50.
    """
    parts = [p for p in re.split(r"<cdrh\d+>", framework_str, flags=re.I) if p]

    template = Structure.from_mmcif(template_path)
    if template_chain_id not in template.chains:
        raise ValueError(
            f"chain {template_chain_id!r} not in {template_path}; "
            f"available: {list(template.chains)}"
        )
    chain = template.chains[template_chain_id]

    template_seq = ""
    template_ca = []
    for i in range(len(chain.rname)):
        rname = str(chain.rname[i])
        anames = [str(a) for a in chain.aname[i] if str(a)]
        if "CA" in anames:
            ca = chain.coords[i, anames.index("CA")]
            template_seq += _AA3TO1.get(rname, "X")
            template_ca.append(np.asarray(ca, dtype=np.float32))
        else:
            template_seq += "X"
            template_ca.append(None)

    cursor = 0
    matches = []
    for part in parts:
        idx = template_seq.find(part, cursor)
        if idx == -1:
            raise ValueError(
                f"Framework part not found in template "
                f"{template_path}:{template_chain_id} starting from index {cursor}\n"
                f"  looking for: {part!r}\n"
                f"  template seq: {template_seq}"
            )
        matches.append((idx, len(part)))
        cursor = idx + len(part)

    binder_seq = schema[binder_chain]["sequence"]
    binder_len = len(binder_seq)
    coords_per_binder_pos = [None] * binder_len
    binder_pos = 0
    for part_idx, (template_start, length) in enumerate(matches):
        for k in range(length):
            if binder_pos >= binder_len or binder_seq[binder_pos] == "X":
                raise ValueError(
                    f"Binder framework / schema mismatch at part {part_idx}, "
                    f"binder_pos {binder_pos}, char "
                    f"{binder_seq[binder_pos] if binder_pos < binder_len else 'EOF'!r}"
                )
            ca = template_ca[template_start + k]
            if ca is not None:
                coords_per_binder_pos[binder_pos] = ca
            binder_pos += 1
        while binder_pos < binder_len and binder_seq[binder_pos] == "X":
            binder_pos += 1

    chain_offset = 0
    for key in schema:
        if key == "connections":
            continue
        if key == binder_chain:
            break
        chain_offset += len(schema[key]["sequence"])

    template_coords = np.zeros((n_tokens, 3), dtype=np.float32)
    has_template = np.zeros(n_tokens, dtype=bool)
    for binder_pos, ca in enumerate(coords_per_binder_pos):
        if ca is not None:
            has_template[chain_offset + binder_pos] = True
            template_coords[chain_offset + binder_pos] = ca

    diff = template_coords[:, None, :] - template_coords[None, :, :]
    dists = np.sqrt((diff**2).sum(-1))
    boundaries = np.linspace(1, 50, 50)
    bin_idx = (dists[..., None] > boundaries).sum(-1).astype(np.int64)

    pair_mask = has_template[:, None] & has_template[None, :]
    distogram_emb = np.where(pair_mask, bin_idx, 0).astype(np.int64)
    return distogram_emb, pair_mask


def _compute_target_distogram(
    template_path,
    template_chain_id,
    schema,
    schema_chain_id,
    n_tokens,
    subsample_frac=1.0,
    seed=0,
    pinned_indices=None,
):
    """Build (distogram_emb, distogram_mask) for target chain conditioning.

    subsample_frac: fraction of resolved positions to keep (1.0 = all).
    pinned_indices: token indices always included regardless of subsampling.
    """
    template = Structure.from_mmcif(template_path)
    if template_chain_id not in template.chains:
        raise ValueError(
            f"chain {template_chain_id!r} not in {template_path}; "
            f"available: {list(template.chains)}"
        )
    chain = template.chains[template_chain_id]

    chain_offset = 0
    for key in schema:
        if key == "connections":
            continue
        if key == schema_chain_id:
            break
        chain_offset += len(schema[key]["sequence"])
    chain_len = len(schema[schema_chain_id]["sequence"])

    template_coords = np.zeros((n_tokens, 3), dtype=np.float32)
    has_template = np.zeros(n_tokens, dtype=bool)
    for i in range(min(len(chain.rname), chain_len)):
        anames = [str(a) for a in chain.aname[i]]
        if "CA" in anames:
            ca_j = anames.index("CA")
            if chain.mask[i, ca_j]:
                template_coords[chain_offset + i] = chain.coords[i, ca_j]
                has_template[chain_offset + i] = True

    if subsample_frac < 1.0:
        pinned_set = set(pinned_indices or [])
        positions = np.where(has_template)[0]
        pinned = np.array([p for p in positions if p in pinned_set], dtype=int)
        pool = np.array([p for p in positions if p not in pinned_set], dtype=int)
        n_keep = max(0, round(len(positions) * subsample_frac) - len(pinned))
        sampled = np.random.default_rng(seed).choice(
            pool, size=min(n_keep, len(pool)), replace=False
        )
        has_template = np.zeros(n_tokens, dtype=bool)
        has_template[sampled] = True
        has_template[pinned] = True

    diff = template_coords[:, None, :] - template_coords[None, :, :]
    dists = np.sqrt((diff**2).sum(-1))
    boundaries = np.linspace(1, 50, 50)
    bin_idx = (dists[..., None] > boundaries).sum(-1).astype(np.int64)

    pair_mask = has_template[:, None] & has_template[None, :]
    distogram_emb = np.where(pair_mask, bin_idx, 0).astype(np.int64)
    return distogram_emb, pair_mask


def _sample_dir(savedir: str, name: str, b_idx: int) -> str:
    return f"{savedir}/{name}/sample{b_idx}"


class Design:
    def __init__(self, cfg):
        self.cfg = cfg

        if os.path.isfile(cfg.input):
            self._input_dir = os.path.dirname(os.path.abspath(cfg.input))
            files = [os.path.basename(cfg.input)]
        else:
            self._input_dir = cfg.input
            files = sorted(f for f in os.listdir(cfg.input) if f.endswith(_INPUT_EXT))

            target_filter = getattr(cfg, "targets", None) or getattr(cfg, "target_list", None)
            if target_filter:
                target_set = set(target_filter)
                files = [f for f in files if os.path.splitext(f)[0] in target_set]

        savedir = cfg.output
        n_backbones = cfg.num_backbones

        self.items = []
        for fname in files:
            name = os.path.splitext(fname)[0]
            for b_idx in range(n_backbones):
                last_refold = f"{_sample_dir(savedir, name, b_idx)}/refolds/sample{b_idx}_design0_refold4.cif"
                if cfg.skip_existing and os.path.exists(last_refold):
                    continue
                self.items.append((fname, b_idx))
        self._last_batch_end = None

        self._target_template_chain = None
        self._target_template_chain_id = None
        target_tmpl = getattr(cfg, "target_template", None)
        if target_tmpl:
            chain_id = getattr(target_tmpl, "chain", "A")
            tmpl_struct = Structure.from_mmcif(target_tmpl.path)
            self._target_template_chain = tmpl_struct.chains[chain_id]
            self._target_template_chain_id = chain_id

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cfg = self.cfg
        fname, b_idx = self.items[idx]
        name = os.path.splitext(fname)[0]
        input_path = os.path.join(self._input_dir, fname)

        rng = random.Random(b_idx)

        t0 = time.time()
        schema = _build_schema_for_input(
            input_path,
            binder_cfg=cfg.binder,
            target_chains=cfg.target_chains or None,
            rng=rng,
        )
        struct = Structure.from_schema(schema)        
        msas = load_msa_from_dir(cfg.msa_dir, struct.chains)
        
        pairing = construct_paired_msa(msas)
        feats = AF3Featurizer(struct, msas, pairing).featurize(compute_frames=True)
        feats = finalize_feats(feats, struct, name, seed_idx=b_idx)
        feats["backbone_idx"] = b_idx
        struct.msa_summary = msa_summary(msas)

        # finalize_feats zeros is_epitope; override after the call.
        epitope_idx = _resolve_epitope_idx(
            schema,
            cfg.epitope_chain,
            list(cfg.epitope_residues or []),
        )
        if epitope_idx:
            feats["is_epitope"][epitope_idx] = 1

        paratope_idx = []
        if cfg.binder.type == "vhh" and cfg.binder.get("paratope_from_cdrs", False):
            binder_seq = schema[cfg.binder.chain]["sequence"]
            cdr_positions = [i + 1 for i, aa in enumerate(binder_seq) if aa == "X"]
            paratope_idx = _resolve_residue_idx(
                schema,
                cfg.binder.chain,
                cdr_positions,
                resnum_map=None,
            )
            if paratope_idx:
                feats["is_epitope"][paratope_idx] = 1

        framework_tokens = 0
        if cfg.binder.type == "vhh" and cfg.binder.get("framework_template", None):
            tmpl = cfg.binder.framework_template
            n_tokens = len(feats["restype"])
            disto_emb, disto_mask = _compute_framework_distogram(
                tmpl["path"],
                tmpl.get("chain", "B"),
                cfg.binder.framework,
                schema,
                cfg.binder.chain,
                n_tokens,
            )
            feats["distogram_emb"] = disto_emb
            feats["distogram_mask"] = disto_mask
            framework_tokens = int(disto_mask.any(-1).sum())

        target_template_tokens = 0
        target_tmpl = getattr(cfg, "target_template", None)
        if target_tmpl:
            chain_id = getattr(target_tmpl, "chain", "A")
            subsample_frac = getattr(target_tmpl, "subsample_frac", 1.0)
            n_tokens = len(feats["restype"])
            disto_emb_t, disto_mask_t = _compute_target_distogram(
                target_tmpl.path,
                chain_id,
                schema,
                chain_id,
                n_tokens,
                subsample_frac=subsample_frac,
                seed=b_idx,
                pinned_indices=epitope_idx,
            )
            if "distogram_mask" in feats:
                feats["distogram_emb"] = feats["distogram_emb"] + disto_emb_t
                feats["distogram_mask"] = feats["distogram_mask"] | disto_mask_t
            else:
                feats["distogram_emb"] = disto_emb_t
                feats["distogram_mask"] = disto_mask_t
            target_template_tokens = int(disto_mask_t.any(-1).sum())

        binder_len = len(schema[cfg.binder.chain]["sequence"])
        _log(
            f"__getitem__ {name} b{b_idx}: "
            f"t={time.time()-t0:.2f}s  "
            f"epitope_tokens={len(epitope_idx)}  paratope_tokens={len(paratope_idx)}  "
            f"framework_tokens={framework_tokens}  target_template_tokens={target_template_tokens}  "
            f"binder_len={binder_len}"
        )
        return feats

    # ------------------------------------------------------------------ #
    # GPU pass                                                            #
    # ------------------------------------------------------------------ #

    def _save_samples(self, struct_template, sample_coords, path_fn):
        """Write samples as CIF + PDB. path_fn(n) returns (cif_path, pdb_path)."""
        pdb_paths = []
        for n, samp in enumerate(sample_coords):
            s = copy.deepcopy(struct_template)
            _copy_sample_to_struct(s, samp)
            cif_path, pdb_path = path_fn(n)
            os.makedirs(os.path.dirname(cif_path), exist_ok=True)
            s.to_mmcif(cif_path, metadata=True)
            _struct_to_pdb(s, pdb_path)
            pdb_paths.append(pdb_path)
        return pdb_paths

    def _refold_with_seq(
        self, model, struct, design_chain, design_seq, recycling_steps, diff_cfg, device
    ):
        """Re-featurize and run pairformer + diffusion + confidence with a redesigned sequence."""
        schema = struct.to_schema()
        schema[design_chain] = {"type": "protein", "sequence": design_seq}
        struct_new = Structure.from_schema(schema)
        
        msas = load_msa_from_dir(self.cfg.msa_dir, struct_new.chains)

        # Refold non-binder chains with their real MSAs (use_msa: true).
        for chain_id, chain in struct_new.chains.items():
            if chain_id == design_chain:
                continue
            if "polypeptide" in chain.type and msas[chain_id].path is None:
                raise ValueError(
                    f"refold: non-binder chain {chain_id} has no MSA "
                    "(use_msa: true required for non-binder chains)"
                )

        pairing = construct_paired_msa(msas)
        feats = AF3Featurizer(struct_new, msas, pairing).featurize(compute_frames=True)
        feats = finalize_feats(feats, struct_new, "refold", seed_idx=0)
        msa_info = msa_summary(msas)
        batch = collate([feats])
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

        with torch.no_grad():
            out = model.pairformer_forward(batch, recycling_steps=recycling_steps)
            diff = model.sample_diffusion(batch, out, diff_cfg)

        samples = diff["sample_atom_coords"].cpu().numpy()
        agg_confs = []
        if model.cfg.model.has_confidence:
            coords = diff["sample_atom_coords"]
            mul = diff_cfg.diffusion_samples
            ntoks = int(batch["token_pad_mask"][0].sum())
            for i in range(mul):
                conf = model.sm_confidence_module(
                    batch,
                    out | {"sample_atom_coords": coords[i::mul]},
                    multiplicity=1,
                )
                pde = conf["pde"][0, :ntoks, :ntoks]
                pae = conf["pae"][0, :ntoks, :ntoks]
                plddt = conf["plddt"][0, :ntoks]
                pae_logits = conf["pae_logits"][0, :ntoks, :ntoks]
                frame_mask = batch["frames_mask"][0, :ntoks]
                asym_id = batch["asym_id_"][0]
                agg_conf = compute_agg_confidence(
                    pde=pde,
                    pae=pae,
                    plddt=plddt,
                    pae_logits=pae_logits,
                    asym_id=asym_id,
                    frame_mask=frame_mask,
                    use_torch=True,
                )
                torch.cuda.empty_cache()

                if getattr(model.cfg.model, "has_contact_module", False):
                    contact_out = model.contact_module(
                        batch,
                        out | {"sample_atom_coords": coords[i::mul]},
                        multiplicity=1,
                    )
                    agg_conf["contact_scores"] = compute_contact_stats(
                        contact_out["contact_logits"][0, :ntoks, :ntoks].cpu(),
                        contact_out["pred_dist"][0, :ntoks, :ntoks].cpu(),
                        batch["asym_id_"][0][:ntoks],
                    )
                    torch.cuda.empty_cache()

                agg_conf.update(msa_info)
                agg_confs.append(agg_conf)
        return struct_new, samples, agg_confs

    def run_batch(self, model, batch):
        cfg = self.cfg
        savedir = cfg.output

        name = batch["name"][0]
        b_idx = int(batch["backbone_idx"][0])
        struct = batch["struct"][0]
        device = batch["restype"].device
        binder_chain = cfg.binder.chain
        binder_seq = "".join(
            _AA3TO1.get(str(r), "X") for r in struct.chains[binder_chain].rname
        )

        sample_dir = _sample_dir(savedir, name, b_idx)
        refold_dir = f"{sample_dir}/refolds"
        os.makedirs(sample_dir, exist_ok=True)

        from types import SimpleNamespace
        import tempfile

        backbone_diff_cfg = SimpleNamespace(
            diffusion=cfg.diffusion,
            diffusion_samples=1,
            diffusion_steps=cfg.diffusion_steps,
        )
        refold_diff_cfg = SimpleNamespace(
            diffusion=cfg.diffusion,
            diffusion_samples=5,
            diffusion_steps=cfg.diffusion_steps,
        )

        t0 = time.time()
        if self._last_batch_end is not None:
            _log(f"{name} b{b_idx}: idle={t0-self._last_batch_end:.2f}s")

        out = model.pairformer_forward(batch, recycling_steps=cfg.recycling_steps)
        t1 = time.time()
        diffusion_out = model.sample_diffusion(batch, out, backbone_diff_cfg)
        t2 = time.time()

        all_samples = diffusion_out["sample_atom_coords"].cpu().numpy()
        coords = diffusion_out["sample_atom_coords"]

        agg_conf = None

        ntoks = int(batch["token_pad_mask"][0].sum())
        conf = model.sm_confidence_module(
            batch,
            out | {"sample_atom_coords": coords},
            multiplicity=1,
        )
        pde = conf["pde"][0, :ntoks, :ntoks]
        pae = conf["pae"][0, :ntoks, :ntoks]
        plddt = conf["plddt"][0, :ntoks]
        pae_logits = conf["pae_logits"][0, :ntoks, :ntoks]
        frame_mask = batch["frames_mask"][0, :ntoks]
        asym_id = batch["asym_id_"][0]
        agg_conf = compute_agg_confidence(
            pde=pde,
            pae=pae,
            plddt=plddt,
            pae_logits=pae_logits,
            asym_id=asym_id,
            frame_mask=frame_mask,
            use_torch=True,
        )
        agg_conf.update(struct.msa_summary)
        if cfg.save_full_confidence:
            np.savez(
                f"{sample_dir}/backbone_confidence.npz",
                pde=pde.cpu().numpy(),
                pae=pae.cpu().numpy(),
                plddt=plddt.cpu().numpy(),
                frame_mask=frame_mask.cpu().numpy(),
                asym_id=asym_id,
            )
        torch.cuda.empty_cache()

        if getattr(model.cfg.model, "has_contact_module", False):
            contact_out = model.contact_module(
                batch,
                out | {"sample_atom_coords": coords},
                multiplicity=1,
            )
            agg_conf["contact_score"] = compute_contact_stats(
                contact_out["contact_logits"][0, :ntoks, :ntoks].cpu(),
                contact_out["pred_dist"][0, :ntoks, :ntoks].cpu(),
                batch["asym_id_"][0][:ntoks],
            )
            torch.cuda.empty_cache()

        with open(f"{sample_dir}/backbone_confidence.json", "w") as f:
            f.write(json.dumps(agg_conf, indent=4))

        bb_paths = self._save_samples(
            struct,
            all_samples,
            lambda n: (f"{sample_dir}/backbone.cif", f"{sample_dir}/backbone.pdb"),
        )
        backbone_pdb = bb_paths[0]

        backbone_struct = copy.deepcopy(struct)
        _copy_sample_to_struct(backbone_struct, all_samples[0])
        if cfg.binder.type == "vhh" and cfg.binder.get("paratope_from_cdrs", False):
            paratope_positions = [i for i, aa in enumerate(binder_seq) if aa == "X"]
        else:
            paratope_positions = []
        epitope_chain = cfg.epitope_chain or None
        epitope_positions = None
        if epitope_chain and cfg.epitope_residues:
            ridx = backbone_struct.chains[epitope_chain].ridx.tolist()
            epitope_positions = [
                ridx.index(r) for r in cfg.epitope_residues if r in ridx
            ]
        if agg_conf is not None:
            agg_conf["contact_stats"] = compute_interface_contacts(
                backbone_struct,
                binder_chain,
                paratope_positions,
                epitope_chain=epitope_chain,
                epitope_positions=epitope_positions,
            )
            with open(f"{sample_dir}/backbone_confidence.json", "w") as f:
                f.write(json.dumps(agg_conf, indent=4))

        iptm_str = (
            f" iptm={agg_conf['complex_iptm']:.3f}"
            if agg_conf and "complex_iptm" in agg_conf
            else ""
        )
        _log(f"{name} b{b_idx}: pf={t1-t0:.2f}s diff={t2-t1:.2f}s{iptm_str}")

        if cfg.inverse_folder.type != "none":
            with tempfile.TemporaryDirectory() as ifold_dir:
                redesigned_seqs = _inverse_fold(
                    backbone_pdb,
                    binder_chain,
                    cfg.inverse_folder,
                    binder_seq,
                    ifold_dir,
                )

            for j, seq in enumerate(redesigned_seqs):
                design_id = f"sample{b_idx}_design{j}"
                with open(f"{sample_dir}/{design_id}.fasta", "w") as f:
                    f.write(f">{design_id}\n{seq}\n")

                t_r0 = time.time()
                struct_m, samples_m, refold_confs = self._refold_with_seq(
                    model,
                    struct,
                    binder_chain,
                    seq,
                    cfg.recycling_steps,
                    refold_diff_cfg,
                    device,
                )
                self._save_samples(
                    struct_m,
                    samples_m,
                    lambda r, _id=design_id: (
                        f"{refold_dir}/{_id}_refold{r}.cif",
                        f"{refold_dir}/{_id}_refold{r}.pdb",
                    ),
                )
                for r, conf in enumerate(refold_confs):
                    refold_struct = copy.deepcopy(struct_m)
                    _copy_sample_to_struct(refold_struct, samples_m[r])
                    conf["contact_stats"] = compute_interface_contacts(
                        refold_struct,
                        binder_chain,
                        paratope_positions,
                        epitope_chain=epitope_chain,
                        epitope_positions=epitope_positions,
                    )
                    conf["binder_scRMSD"] = compute_self_consistency_rmsd(
                        backbone_struct,
                        refold_struct,
                        binder_chain,
                    )
                    conf["scDockQ"] = compute_dockq(backbone_struct, refold_struct)
                    if self._target_template_chain is not None:
                        conf["target_template_lddt"] = compute_target_lddt(
                            self._target_template_chain,
                            refold_struct,
                            self._target_template_chain_id,
                            epitope_positions=epitope_positions,
                        )
                    with open(
                        f"{refold_dir}/{design_id}_refold{r}_confidence.json", "w"
                    ) as f:
                        f.write(json.dumps(conf, indent=4))
                best = max((c.get("complex_iptm", 0) for c in refold_confs), default=0)
                _log(
                    f"  {design_id}: refold={time.time()-t_r0:.2f}s best_iptm={best:.3f}"
                )

        _log(f"{name} b{b_idx}: total_batch={time.time()-t0:.2f}s")
        self._last_batch_end = time.time()
