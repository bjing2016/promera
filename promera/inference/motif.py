"""De novo motif scaffolding via the Promera cofolding model.

Given a reference mmCIF motif structure and a set of motif residue ranges, generate a
single designed chain that:

  1. Embeds residues at fixed sequence/position, with X-scaffold
     flanks and linkers of (optionally sampled) length around/between them.
  2. Conditions the trunk on the motif's intra Cα geometry via the
     distogram embedding and `is_epitope` flag.
  3. Runs the cofolding pass (pairformer -> diffusion) to scaffold a backbone.
  4. (Optional) Inverse-folds the scaffold (X) positions with LigandMPNN /
     ProteinMPNN / SolubleMPNN / AbMPNN while keeping the motif sequence
     fixed and refolds each designed sequence.

Run:
    python -m promera \\
        --task promera.inference.MotifScaffolding \\
        --task_config examples/motif_scaffold.yaml \\
        output=out/
"""

import copy
import json
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch

from tinyprot.feature import AF3Featurizer
from tinyprot.geometry import compute_rmsd
from tinyprot.msa import construct_paired_msa, load_msa_from_dir
from tinyprot.structure import Structure

from .design import _inverse_fold, _sample_dir
from .utils import (
    _AA3TO1,
    _copy_sample_to_struct,
    _log,
    ca_distogram,
    compute_agg_confidence,
    compute_contact_stats,
    compute_self_consistency_rmsd,
    finalize_feats,
    refold_with_seq,
    save_samples,
)


def _sample_len(spec, rng):
    """spec is an int (fixed) or [min, max] (sampled inclusive)."""
    if isinstance(spec, int):
        return spec
    return rng.randint(int(spec[0]), int(spec[1]))


def _parse_ranges(ranges):
    """Parse ["10-25", "40", ...] into a list of (start, end) inclusive int pairs."""
    out = []
    for r in ranges:
        if isinstance(r, int):
            out.append((r, r))
            continue
        s = str(r).strip()
        if "-" in s:
            a, b = s.split("-", 1)
            out.append((int(a), int(b)))
        else:
            out.append((int(s), int(s)))
    return out


def _load_motif_segments(chain, ranges):
    """Extract motif segments from a reference chain.

    Returns a list of segments, each a list of (one_letter, ca_xyz_or_None)
    tuples, in the order given by `ranges`. Residue numbers are matched against
    the chain's `ridx` (the reference's own numbering).
    """
    ridx = chain.ridx.tolist()
    segments = []
    for start, end in _parse_ranges(ranges):
        seg = []
        for resnum in range(start, end + 1):
            if resnum not in ridx:
                raise ValueError(
                    f"motif residue {resnum} not found in reference chain "
                    f"(available {ridx[0]}..{ridx[-1]})"
                )
            i = ridx.index(resnum)
            rname = str(chain.rname[i])
            one = _AA3TO1.get(rname, "X")
            anames = [str(a) for a in chain.aname[i]]
            ca = None
            if "CA" in anames:
                j = anames.index("CA")
                if chain.mask[i, j]:
                    ca = np.asarray(chain.coords[i, j], dtype=np.float32)
            seg.append((one, ca))
        if not seg:
            raise ValueError(f"empty motif segment for range {start}-{end}")
        segments.append(seg)
    return segments


def _load_motif_segments_pdb(pdb_path, chain_id, ranges):
    """Like _load_motif_segments but reads a .pdb via prody """
    import prody

    st = prody.parsePDB(pdb_path)
    if st is None:
        raise ValueError(f"could not parse {pdb_path}")
    chain = st.getHierView()[chain_id]
    if chain is None:
        raise ValueError(f"chain {chain_id!r} not in {pdb_path}")
    res_by_num = {res.getResnum(): res for res in chain.iterResidues()}

    segments = []
    for start, end in _parse_ranges(ranges):
        seg = []
        for resnum in range(start, end + 1):
            res = res_by_num.get(resnum)
            if res is None:
                raise ValueError(
                    f"motif residue {resnum} not found in chain {chain_id} of {pdb_path}"
                )
            one = _AA3TO1.get(res.getResname(), "X")
            ca_atom = res.select("name CA")
            ca = (
                None
                if ca_atom is None
                else np.asarray(ca_atom.getCoords()[0], dtype=np.float32)
            )
            seg.append((one, ca))
        segments.append(seg)
    return segments


def _build_motif_schema(segments, scaffold_cfg, design_chain, rng, total_length=None):
    """Build a single-chain schema with motif residues fixed and X scaffold.

    Returns (schema, motif_positions, motif_ca) where motif_positions are the
    0-based token indices of motif residues in the design chain and motif_ca is
    the aligned list of reference Cα coords (entries may be None if unresolved).
    """
    n_term = scaffold_cfg.get("n_term", [10, 20])
    c_term = scaffold_cfg.get("c_term", [10, 20])
    linker = scaffold_cfg.get("linker", [4, 8])

    for _ in range(1000):
        seq = "X" * _sample_len(n_term, rng)
        motif_positions = []
        motif_ca = []
        for si, seg in enumerate(segments):
            if si > 0:
                seq += "X" * _sample_len(linker, rng)
            for one, ca in seg:
                motif_positions.append(len(seq))
                motif_ca.append(None if ca is None else ca.tolist())
                seq += one
        seq += "X" * _sample_len(c_term, rng)
        if total_length is None or total_length[0] <= len(seq) <= total_length[1]:
            break
    else:
        raise ValueError(
            f"could not sample scaffold within total_length {list(total_length)} "
            "after 1000 tries"
        )

    schema = {design_chain: {"type": "protein", "sequence": seq}}
    return schema, motif_positions, motif_ca


def _compute_motif_distogram(motif_positions, motif_ca, n_tokens):
    """Build (distogram_emb, distogram_mask) over intra-motif Cα pairs.

    Bins match the model's training: torch.linspace(1, 50, 50), bin index 0..50.
    Only motif-motif pairs with resolved Cα are conditioned; all else is masked.
    """
    template_coords = np.zeros((n_tokens, 3), dtype=np.float32)
    has_template = np.zeros(n_tokens, dtype=bool)
    for pos, ca in zip(motif_positions, motif_ca):
        if ca is not None:
            template_coords[pos] = ca
            has_template[pos] = True

    return ca_distogram(template_coords, has_template)


def _motif_ca_rmsd(struct, design_chain, motif_positions, motif_ca):
    """Kabsch-aligned Cα RMSD between the reference motif and a structure's
    motif positions. Returns None if too few resolved positions."""
    chain = struct.chains[design_chain]
    ref, pred = [], []
    for pos, ca in zip(motif_positions, motif_ca):
        if ca is None:
            continue
        anames = [str(a) for a in chain.aname[pos]]
        if "CA" in anames:
            ref.append(ca)
            pred.append(chain.coords[pos, anames.index("CA")])
    if len(pred) < 3:
        return None
    return float(
        compute_rmsd(
            np.asarray(ref, dtype=np.float32), np.asarray(pred, dtype=np.float32)
        )
    )


class MotifScaffolding:
    def __init__(self, cfg):
        self.cfg = cfg

        motif = cfg.design.motif
        ref_path = motif.reference
        motif_chain_id = getattr(motif, "motif_chain", "A")
        ranges = list(motif.motif_ranges)
        if ref_path.lower().endswith((".cif", ".mmcif")):
            ref = Structure.from_mmcif(ref_path)
            if motif_chain_id not in ref.chains:
                raise ValueError(
                    f"motif_chain {motif_chain_id!r} not in {ref_path}; "
                    f"available: {list(ref.chains)}"
                )
            self._segments = _load_motif_segments(ref.chains[motif_chain_id], ranges)
        else:
            self._segments = _load_motif_segments_pdb(ref_path, motif_chain_id, ranges)
        self._name = getattr(cfg, "name", None) or os.path.splitext(
            os.path.basename(ref_path)
        )[0]

        savedir = cfg.output
        self.items = []
        for b_idx in range(cfg.num_backbones):
            last_refold = (
                f"{_sample_dir(savedir, self._name, b_idx)}"
                f"/refolds/sample{b_idx}_design0_refold4.cif"
            )
            if cfg.skip_existing and os.path.exists(last_refold):
                continue
            self.items.append(b_idx)
        self._last_batch_end = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cfg = self.cfg
        b_idx = self.items[idx]
        rng = random.Random(b_idx)

        design = cfg.design
        t0 = time.time()
        schema, motif_positions, motif_ca = _build_motif_schema(
            self._segments,
            design.scaffold,
            design.chain,
            rng,
            total_length=design.get("total_length", None),
        )
        struct = Structure.from_schema(schema)

        msas = load_msa_from_dir(cfg.msa_dir, struct.chains)
        pairing = construct_paired_msa(msas)
        feats = AF3Featurizer(struct, msas, pairing).featurize(compute_frames=True)
        feats = finalize_feats(feats, struct, self._name, seed_idx=b_idx)
        feats["backbone_idx"] = b_idx

        n_tokens = len(feats["restype"])
        if motif_positions:
            disto_emb, disto_mask = _compute_motif_distogram(
                motif_positions, motif_ca, n_tokens
            )
            feats["distogram_emb"] = disto_emb
            feats["distogram_mask"] = disto_mask
            feats["is_epitope"][motif_positions] = 1

        feats["motif_positions"] = motif_positions
        feats["motif_ca"] = motif_ca

        _log(
            f"__getitem__ {self._name} b{b_idx}: t={time.time()-t0:.2f}s  "
            f"len={len(schema[design.chain]['sequence'])}  "
            f"motif_tokens={len(motif_positions)}"
        )
        return feats


    def run_batch(self, model, batch):
        cfg = self.cfg
        savedir = cfg.output

        name = batch["name"][0]
        b_idx = int(batch["backbone_idx"][0])
        struct = batch["struct"][0]
        device = batch["restype"].device
        design_chain = cfg.design.chain
        motif_positions = list(batch["motif_positions"][0])
        motif_ca = batch["motif_ca"][0]
        design_seq = "".join(
            _AA3TO1.get(str(r), "X") for r in struct.chains[design_chain].rname
        )

        sample_dir = _sample_dir(savedir, name, b_idx)
        refold_dir = f"{sample_dir}/refolds"
        os.makedirs(sample_dir, exist_ok=True)

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

        bb_paths = save_samples(
            struct,
            all_samples,
            lambda n: (f"{sample_dir}/backbone.cif", f"{sample_dir}/backbone.pdb"),
        )
        backbone_pdb = bb_paths[0]

        backbone_struct = copy.deepcopy(struct)
        _copy_sample_to_struct(backbone_struct, all_samples[0])
        agg_conf["backbone_motif_rmsd"] = _motif_ca_rmsd(
            backbone_struct, design_chain, motif_positions, motif_ca
        )
        with open(f"{sample_dir}/backbone_confidence.json", "w") as f:
            f.write(json.dumps(agg_conf, indent=4))

        _log(
            f"{name} b{b_idx}: pf={t1-t0:.2f}s diff={t2-t1:.2f}s "
            f"motif_rmsd={agg_conf['backbone_motif_rmsd']}"
        )

        if cfg.inverse_folder.type != "none":
            import tempfile

            with tempfile.TemporaryDirectory() as ifold_dir:
                redesigned_seqs = _inverse_fold(
                    backbone_pdb,
                    design_chain,
                    cfg.inverse_folder,
                    design_seq,
                    ifold_dir,
                )

            for j, seq in enumerate(redesigned_seqs):
                design_id = f"sample{b_idx}_design{j}"
                with open(f"{sample_dir}/{design_id}.fasta", "w") as f:
                    f.write(f">{design_id}\n{seq}\n")

                t_r0 = time.time()
                struct_m, samples_m, refold_confs = refold_with_seq(
                    model,
                    self.cfg.msa_dir,
                    struct,
                    design_chain,
                    seq,
                    cfg.recycling_steps,
                    refold_diff_cfg,
                    device,
                )
                save_samples(
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
                    conf["scRMSD"] = compute_self_consistency_rmsd(
                        backbone_struct, refold_struct, design_chain
                    )
                    conf["motif_rmsd"] = _motif_ca_rmsd(
                        refold_struct, design_chain, motif_positions, motif_ca
                    )
                    with open(
                        f"{refold_dir}/{design_id}_refold{r}_confidence.json", "w"
                    ) as f:
                        f.write(json.dumps(conf, indent=4))
                best = min(
                    (c["motif_rmsd"] for c in refold_confs if c.get("motif_rmsd")),
                    default=None,
                )
                _log(f"  {design_id}: refold={time.time()-t_r0:.2f}s best_motif_rmsd={best}")

        _log(f"{name} b{b_idx}: total_batch={time.time()-t0:.2f}s")
        self._last_batch_end = time.time()
