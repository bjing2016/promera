import importlib
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from scipy.special import softmax
from tinyprot.structure import Structure

def tm_function(d, Nres):
    d0 = 1.24 * (max(Nres, 19) - 15) ** (1 / 3) - 1.8
    return 1 / (1 + (d / d0) ** 2)


def _ipsae_d0(L):
    L = float(max(L, 0))
    d0 = 1.24 * (L - 15) ** (1 / 3) - 1.8 if L > 27 else 1.0
    return max(1.0, d0)


def compute_ipsae(pae, asym_id, frame_mask, pae_cutoff=10.0):
    """Compute ipSAE_d0res_max for each inter-chain pair with at least one framed chain.

    Returns dict mapping "{id1}_{id2}" -> float for id1 < id2.
    """
    if hasattr(pae, "cpu"):
        pae = pae.cpu().numpy()
    if hasattr(frame_mask, "cpu"):
        frame_mask = frame_mask.cpu().numpy()
    pae = np.asarray(pae, dtype=float)
    frame_mask = np.asarray(frame_mask, dtype=bool)[: pae.shape[0]]
    asym_id = np.asarray(asym_id)[: pae.shape[0]]

    unique_ids = np.unique(asym_id)
    if len(unique_ids) < 2:
        return {}

    ipsae_asym = {}
    for id1 in unique_ids:
        for id2 in unique_ids:
            if id1 == id2:
                continue
            if not (
                frame_mask[asym_id == id1].any() or frame_mask[asym_id == id2].any()
            ):
                continue
            sub = pae[np.ix_(asym_id == id1, asym_id == id2)]
            best = 0.0
            for row in sub:
                valid = row < pae_cutoff
                n0 = int(valid.sum())
                if n0 == 0:
                    continue
                d0 = _ipsae_d0(n0)
                score = float((1.0 / (1.0 + (row[valid] / d0) ** 2)).mean())
                if score > best:
                    best = score
            ipsae_asym[(id1, id2)] = best

    result = {}
    for id1 in unique_ids:
        for id2 in unique_ids:
            if id1 >= id2:
                continue
            if (id1, id2) in ipsae_asym or (id2, id1) in ipsae_asym:
                result[f"{id1}_{id2}"] = max(
                    ipsae_asym.get((id1, id2), 0.0),
                    ipsae_asym.get((id2, id1), 0.0),
                )
    return result


def compute_agg_confidence(
    pde,
    pae,
    plddt,
    pae_logits,
    asym_id,
    frame_mask,
    use_torch=False,
):
    if use_torch:
        device = pae_logits.device
        pae_value = torch.arange(0.25, 32, 0.5, device=device, dtype=pae_logits.dtype)
        sm = lambda x: torch.softmax(x, dim=-1)
        to_mask = lambda x: torch.tensor(x, device=device)
    else:
        pae_value = np.arange(0.25, 32, 0.5)
        sm = lambda x: softmax(x, axis=-1)
        to_mask = lambda x: x

    asym_id = np.asarray(asym_id)

    out = {}
    out["complex_plddt"] = float(plddt.mean())

    N_res = len(asym_id)
    ptm_arr = (sm(pae_logits) * tm_function(pae_value, N_res)).sum(-1)
    out["complex_ptm"] = float(ptm_arr.mean(-1)[frame_mask].max())

    unique_ids = np.unique(asym_id)
    out["chain_plddt"] = {
        str(idx): float(plddt[asym_id == idx].mean()) for idx in unique_ids
    }

    if len(unique_ids) > 1:
        diff_chain_mask_np = asym_id[:, None] != asym_id[None, :]
        diff_chain_mask = to_mask(diff_chain_mask_np)
        iptm_vec = (ptm_arr * diff_chain_mask).sum(-1) / diff_chain_mask.sum(-1)
        out["complex_iptm"] = float(iptm_vec[frame_mask].max())
        out["iptm"] = {}

    out["ptm"] = {}

    for idx1 in unique_ids:
        for idx2 in unique_ids:
            if idx1 > idx2:
                continue

            this_mask = (asym_id == idx1) | (asym_id == idx2)
            this_frame_mask = frame_mask[this_mask]
            if not bool(
                this_frame_mask.any() if use_torch else np.any(this_frame_mask)
            ):
                continue

            this_N_res = int(this_mask.sum())
            this_pae_logits = pae_logits[this_mask][:, this_mask]
            this_ptm_arr = (
                sm(this_pae_logits) * tm_function(pae_value, this_N_res)
            ).sum(-1)

            if idx1 == idx2:
                out["ptm"][str(idx1)] = float(
                    this_ptm_arr.mean(-1)[this_frame_mask].max()
                )
            else:
                this_diff_chain_mask = to_mask(
                    diff_chain_mask_np[this_mask][:, this_mask]
                )
                this_iptm_vec = (this_ptm_arr * this_diff_chain_mask).sum(
                    -1
                ) / this_diff_chain_mask.sum(-1)
                out["iptm"][f"{idx1}_{idx2}"] = float(
                    this_iptm_vec[this_frame_mask].max()
                )

    if len(unique_ids) > 1:
        out["ipsae"] = compute_ipsae(pae, asym_id, frame_mask)

    return out


def compute_contact_stats(contact_logits, pred_dist, asym_id):
    asym_id = np.asarray(asym_id)
    chain_pair_stats = {}
    for ci in np.unique(asym_id).tolist():
        for cj in np.unique(asym_id).tolist():
            if ci >= cj:
                continue
            mask_i = torch.tensor(asym_id == ci)
            mask_j = torch.tensor(asym_id == cj)
            pair_mask = mask_i[:, None] & mask_j[None, :]
            pred_contact = (pred_dist < 8.0) & pair_mask
            n_contacts = int(pred_contact.sum().item())
            if n_contacts == 0:
                continue
            avg_prob = float(torch.sigmoid(contact_logits[pred_contact]).mean().item())
            chain_pair_stats[f"{ci}_{cj}"] = {
                "n_pred_contacts": n_contacts,
                "avg_contact_prob": avg_prob,
            }
    return chain_pair_stats


def msa_summary(msas):
    """Per-chain MSA depth (number of sequences) and source path.

    `msas` maps chain_id -> tinyprot MSA. Dummy MSAs (no precomputed
    alignment) report a depth of 1 (the query only) and a path of None.
    Returns a dict with "msa_depth" and "msa_path" sub-dicts keyed by chain.
    """
    return {
        "msa_depth": {cid: int(len(msa.seqs)) for cid, msa in msas.items()},
        "msa_path": {cid: msa.path for cid, msa in msas.items()},
    }


def finalize_feats(feats, struct, name, seed_idx):
    feats["atom_pad_mask"] = np.ones_like(feats["ref_element"])
    feats["token_pad_mask"] = np.ones_like(feats["restype"])
    feats["is_epitope"] = np.zeros_like(feats["restype"])
    M = len(feats["restype"])
    bond_mat = np.zeros((M, M), dtype=int)
    bond_mat[*feats["token_bonds"].T] = 1
    bond_mat[*feats["token_bonds"].T[::-1]] = 1
    feats["token_bonds"] = bond_mat
    feats["name"] = name
    feats["seed_idx"] = seed_idx
    feats["atom_is_protein"] = feats["is_protein"][feats["atom_to_token"]]
    feats["atom_is_rna"] = feats["is_rna"][feats["atom_to_token"]]
    feats["atom_is_dna"] = feats["is_dna"][feats["atom_to_token"]]
    feats["atom_is_ligand"] = feats["is_ligand"][feats["atom_to_token"]]
    feats["struct"] = struct
    feats["atom_is_std"] = feats["is_std"][feats["atom_to_token"]]
    return feats


# ---------------------------------------------------------------------------
# Design utilities: target parsing, schema construction, output, metrics
# ---------------------------------------------------------------------------

_AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _seqs_from_pdb(path: str) -> dict:
    chains: dict = {}
    seen: set = set()
    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            res3 = line[17:20].strip()
            resnum = line[22:27].strip()
            if res3 not in _AA3TO1:
                continue
            key = (chain, resnum)
            if key not in seen:
                seen.add(key)
                chains.setdefault(chain, []).append(_AA3TO1[res3])
    return {c: "".join(seq) for c, seq in chains.items()}


def _seqs_from_cif(path: str) -> dict:
    struct = Structure.from_mmcif(path)
    seqs = {}
    for chain_id, chain in struct.chains.items():
        if not chain.type.startswith("polymer:polypeptide"):
            continue
        seq = "".join(_AA3TO1.get(str(r), "X") for r in chain.rname)
        if seq:
            seqs[chain_id] = seq
    return seqs


def _resnum_map_from_pdb(path: str) -> dict:
    result: dict = {}
    seen: set = set()
    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            res3 = line[17:20].strip()
            resnum = line[22:27].strip()
            if res3 not in _AA3TO1:
                continue
            key = (chain, resnum)
            if key not in seen:
                seen.add(key)
                chain_map = result.setdefault(chain, {})
                chain_map[resnum] = len(chain_map)
    return result


def _resnum_map_from_cif(path: str) -> dict:
    struct = Structure.from_mmcif(path)
    result: dict = {}
    for chain_id, chain in struct.chains.items():
        if not chain.type.startswith("polymer:polypeptide"):
            continue
        result[chain_id] = {str(ridx): i for i, ridx in enumerate(chain.ridx.tolist())}
    return result


def build_binder_schema(
    target_path: str,
    binder_chain: str,
    binder_length: int,
    target_chains=None,
) -> dict:
    if target_path.endswith(".cif"):
        raw_seqs = _seqs_from_cif(target_path)
    else:
        raw_seqs = _seqs_from_pdb(target_path)

    schema = {}
    for chain_id, seq in raw_seqs.items():
        if target_chains and chain_id not in target_chains:
            continue
        schema[chain_id] = {"type": "protein", "sequence": seq}

    if not schema:
        raise ValueError(
            f"No protein chains found in {target_path} "
            f"(target_chains={target_chains})"
        )
    if binder_chain in schema:
        raise ValueError(
            f"binder_chain '{binder_chain}' conflicts with a target chain "
            f"({list(schema.keys())}). Pick a different binder_chain."
        )

    schema[binder_chain] = {"type": "protein", "sequence": "A" * binder_length}
    return schema


def _resolve_residue_idx(
    schema: dict, chain: str, resnums: list, resnum_map: dict = None
) -> list:
    offset = 0
    for key in schema:
        if key == "connections":
            continue
        seq_len = len(schema[key]["sequence"])
        if key == chain:
            if resnum_map and chain in resnum_map:
                chain_map = resnum_map[chain]
                return [
                    offset + chain_map[str(r)] for r in resnums if str(r) in chain_map
                ]
            return [offset + (r - 1) for r in resnums if 1 <= r <= seq_len]
        offset += seq_len
    return []


def _copy_sample_to_struct(struct, samp: np.ndarray) -> None:
    i = 0
    for chain in struct.chains.values():
        for j in range(len(chain.aname)):
            for k in range(len(chain.aname[j])):
                if chain.aname[j][k] != "":
                    chain.coords[j, k] = samp[i]
                    chain.mask[j, k] = True
                    i += 1


def _struct_to_pdb(struct, path: str) -> None:
    """Write a tinyprot Structure to PDB format (for LigandMPNN consumption)."""
    lines = []
    atom_num = 1
    for asym_id, chain in struct.chains.items():
        is_protein = chain.type.startswith("polymer:polypeptide")
        record = "ATOM  " if is_protein else "HETATM"
        chain_id = asym_id[0] if asym_id else "A"
        for i in range(len(chain.rname)):
            rname = str(chain.rname[i])[:3]
            ridx = int(chain.ridx[i])
            for j in range(len(chain.aname[i])):
                aname = str(chain.aname[i][j])
                if not aname or not chain.mask[i, j]:
                    continue
                aname_pdb = f" {aname:<3s}" if len(aname) < 4 else aname[:4]
                x = float(chain.coords[i, j, 0])
                y = float(chain.coords[i, j, 1])
                z = float(chain.coords[i, j, 2])
                lines.append(
                    f"{record}{atom_num:5d} {aname_pdb} {rname:>3s} {chain_id}{ridx:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
                )
                atom_num += 1
    lines.append("END\n")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.writelines(lines)


_LMPNN_DIR = os.environ.get(
    "LIGANDMPNN_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "LigandMPNN")),
)


def _parse_pdb_atoms(path: str) -> list:
    atoms = []
    with open(path) as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                atoms.append({
                    "record": line[0:6].strip(),
                    "chain": line[21],
                    "resnum": int(line[22:26]),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                })
            except (ValueError, IndexError):
                continue
    return atoms


def _detect_model_type(pdb_path: str) -> str:
    atoms = _parse_pdb_atoms(pdb_path)
    return "ligand_mpnn" if any(a["record"] == "HETATM" for a in atoms) else "soluble_mpnn"


def _get_interface_indices(
    pdb_path: str, binder_chain: str, cutoff: float = 6.0, non_protein_target: bool = True
) -> list:
    atoms = _parse_pdb_atoms(pdb_path)
    binder_res_atoms: dict = {}
    for a in atoms:
        if a["chain"] == binder_chain and a["record"] == "ATOM":
            binder_res_atoms.setdefault(a["resnum"], []).append([a["x"], a["y"], a["z"]])
    sorted_res = sorted(binder_res_atoms)
    if not sorted_res:
        return []
    target_coords = [
        [a["x"], a["y"], a["z"]]
        for a in atoms
        if (non_protein_target and a["record"] == "HETATM")
        or (not non_protein_target and a["record"] == "ATOM" and a["chain"] != binder_chain)
    ]
    if not target_coords:
        return []
    target_arr = np.array(target_coords)
    interface = []
    for i, rn in enumerate(sorted_res):
        b = np.array(binder_res_atoms[rn])
        dists = np.sqrt(((b[:, None, :] - target_arr[None, :, :]) ** 2).sum(-1))
        if dists.min() < cutoff:
            interface.append(i)
    return interface

def _lmpnn_import():
    """Import LigandMPNN's run.main from $LIGANDMPNN_DIR."""
    if _LMPNN_DIR not in sys.path:
        sys.path.insert(0, _LMPNN_DIR)
    try:
        from run import main as lmpnn_main
    except ImportError as e:
        raise ImportError(f"Could not import LigandMPNN from {_LMPNN_DIR}.\n{e}")
    return lmpnn_main


def _lmpnn_resolve(model_type):
    """Map an inverse-folder type to (lmpnn_model_type, checkpoint_protein_mpnn, mp)."""
    requested = str(model_type).lower().replace("-", "").replace("_", "")
    mp = os.path.join(_LMPNN_DIR, "model_params")
    protein_mpnn_ckpt = os.path.join(mp, "proteinmpnn_v_48_020.pt")
    abmpnn_ckpt = os.environ.get("ABMPNN_CHECKPOINT", os.path.join(mp, "abmpnn.pt"))
    if requested == "proteinmpnn":
        return "protein_mpnn", protein_mpnn_ckpt, mp
    if requested == "solublempnn":
        return "soluble_mpnn", protein_mpnn_ckpt, mp
    if requested == "abmpnn":
        if not os.path.exists(abmpnn_ckpt):
            raise FileNotFoundError(
                "AbMPNN checkpoint not found. Expected either:\n"
                f"  {os.path.join(mp, 'abmpnn.pt')}\n"
                "or set:\n  export ABMPNN_CHECKPOINT=/path/to/abmpnn.pt"
            )
        return "protein_mpnn", abmpnn_ckpt, mp
    if requested in {"ligandmpnn", "ligand"}:
        return "ligand_mpnn", protein_mpnn_ckpt, mp
    raise ValueError(
        f"Unknown inverse_folder/model_type: {model_type!r}. "
        "Expected one of: proteinmpnn, solublempnn, ligandmpnn, abmpnn."
    )


def _lmpnn_config(lmpnn_model_type, checkpoint_protein_mpnn, mp, lmpnn_dir, num_seqs, binder_chain):
    """Build the full LigandMPNN run config with our defaults.

    Caller sets the per-run pdb/fixed fields (single: pdb_path/fixed_residues;
    batched: pdb_path_multi/fixed_residues_multi)."""
    return SimpleNamespace(
        model_type=lmpnn_model_type,
        checkpoint_protein_mpnn=checkpoint_protein_mpnn,
        checkpoint_ligand_mpnn=os.path.join(mp, "ligandmpnn_v_32_010_25.pt"),
        checkpoint_soluble_mpnn=os.path.join(mp, "solublempnn_v_48_020.pt"),
        pdb_path="",
        pdb_path_multi="",
        fixed_residues="",
        fixed_residues_multi="",
        redesigned_residues="",
        redesigned_residues_multi="",
        bias_AA="",
        bias_AA_per_residue="",
        bias_AA_per_residue_multi="",
        omit_AA="C",
        omit_AA_per_residue="",
        omit_AA_per_residue_multi="",
        symmetry_residues="",
        symmetry_weights="",
        homo_oligomer=0,
        out_folder=lmpnn_dir,
        file_ending="",
        zero_indexed=0,
        seed=0,
        batch_size=1,
        number_of_batches=int(num_seqs),
        temperature=0.1,
        save_stats=0,
        ligand_mpnn_use_atom_context=1,
        ligand_mpnn_cutoff_for_score=8.0,
        ligand_mpnn_use_side_chain_context=0,
        chains_to_design=binder_chain,
        parse_these_chains_only="",
        transmembrane_buried="",
        transmembrane_interface="",
        global_transmembrane_label=0,
        parse_atoms_with_zero_occupancy=0,
        verbose=0,
        fasta_seq_separation=":",
        force_hetatm=0,
        pack_side_chains=0,
        pack_with_ligand_context=1,
        repack_everything=0,
        packed_suffix="_packed",
        number_of_packs_per_design=4,
        sc_num_denoising_steps=3,
        sc_num_samples=16,
        checkpoint_path_sc=os.path.join(mp, "ligandmpnn_sc_v_32_002_16.pt"),
        checkpoint_per_residue_label_membrane_mpnn=os.path.join(
            mp, "per_residue_label_membrane_mpnn_v_48_020.pt"
        ),
        checkpoint_global_label_membrane_mpnn=os.path.join(
            mp, "global_label_membrane_mpnn_v_48_020.pt"
        ),
    )


def _sanitize_pdb(pdb_path, out_path):
    """Copy a PDB to out_path, rewriting UNK residues to ALA (LigandMPNN rejects UNK)."""
    with open(pdb_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM  ", "HETATM")) and line[17:20] == "UNK":
                line = line[:17] + "ALA" + line[20:]
            fout.write(line)


def _binder_fasta_idx(pdb_path, binder_chain):
    """Column index of the binder chain in LigandMPNN's ':'-joined FASTA output."""
    atoms = _parse_pdb_atoms(pdb_path)
    pdb_chains = sorted(set(a["chain"] for a in atoms))
    return pdb_chains.index(binder_chain)


def _lmpnn_fixed_tokens(pdb_path, binder_chain, fixed_residues, is_ligand_mpnn):
    """Fixed-residue token string for a backbone; autodetects the interface if None."""
    if fixed_residues is not None:
        return fixed_residues
    atoms = _parse_pdb_atoms(pdb_path)
    interface_idx = _get_interface_indices(
        pdb_path, binder_chain, cutoff=6.0, non_protein_target=is_ligand_mpnn
    )
    binder_resnums = sorted(
        set(
            a["resnum"]
            for a in atoms
            if a["chain"] == binder_chain and a["record"] == "ATOM"
        )
    )
    return " ".join(f"{binder_chain}{binder_resnums[i]}" for i in interface_idx)


def _parse_lmpnn_seqs(fasta_path, binder_fasta_idx):
    """Read redesigned binder-chain sequences from a LigandMPNN output FASTA."""
    if not os.path.exists(fasta_path):
        print(f"[lmpnn] no output FASTA at {fasta_path}")
        return []
    seqs = []
    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith(">"):
                parts = line.split(":")
                if binder_fasta_idx < len(parts):
                    seqs.append(parts[binder_fasta_idx])
    return seqs[1:]


def run_lmpnn_redesign(
    pdb_path, binder_chain, lmpnn_dir, num_seqs=8, model_type=None, fixed_residues=None
):
    """Run ProteinMPNN/SolubleMPNN/LigandMPNN/AbMPNN redesign on a binder+target PDB."""
    os.makedirs(lmpnn_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdb_path))[0]
    sanitized_pdb = os.path.join(lmpnn_dir, f"{base}_sanitized.pdb")
    _sanitize_pdb(pdb_path, sanitized_pdb)
    pdb_path = sanitized_pdb

    if model_type is None:
        model_type = _detect_model_type(pdb_path)
    is_ligand_mpnn = (
        str(model_type).lower().replace("-", "").replace("_", "")
        in {"ligandmpnn", "ligand"}
    )

    binder_fasta_idx = _binder_fasta_idx(pdb_path, binder_chain)
    fixed_tokens = _lmpnn_fixed_tokens(pdb_path, binder_chain, fixed_residues, is_ligand_mpnn)

    lmpnn_main = _lmpnn_import()
    lmpnn_model_type, checkpoint_protein_mpnn, mp = _lmpnn_resolve(model_type)
    config = _lmpnn_config(
        lmpnn_model_type, checkpoint_protein_mpnn, mp, lmpnn_dir, num_seqs, binder_chain
    )
    config.pdb_path = pdb_path
    config.fixed_residues = fixed_tokens
    lmpnn_main(config)

    base = os.path.splitext(os.path.basename(pdb_path))[0]
    fasta_path = os.path.join(lmpnn_dir, "seqs", f"{base}.fa")
    return _parse_lmpnn_seqs(fasta_path, binder_fasta_idx)


def _struct_to_protein_dict(struct, device):
    """Build a LigandMPNN parse_PDB-style protein_dict from a tinyprot Structure in
    memory (no PDB round-trip). Returns (protein_dict, ordered_chain_ids).

    Only protein chains are included; chain ids follow asym_id[0] (matching
    _struct_to_pdb). X is [L,4,3] backbone N,CA,C,O; residues missing any backbone
    atom are dropped (parse_PDB requires all four). Non-protein context (ligands,
    nucleotides) is not carried, so ligand_mpnn runs without atom context here."""
    if _LMPNN_DIR not in sys.path:
        sys.path.insert(0, _LMPNN_DIR)
    from data_utils import restype_str_to_int

    Xs, Ss, ridxs, labels, letters, order = [], [], [], [], [], []
    ci = 0
    for asym_id, chain in struct.chains.items():
        if not chain.type.startswith("polymer:polypeptide"):
            continue
        chain_id = asym_id[0] if asym_id else "A"
        order.append(chain_id)
        for i in range(len(chain.rname)):
            coord = {}
            for j, a in enumerate(chain.aname[i]):
                a = str(a)
                if a and bool(chain.mask[i, j]):
                    coord[a] = chain.coords[i, j]
            if not all(k in coord for k in ("N", "CA", "C", "O")):
                continue
            Xs.append(np.stack([coord["N"], coord["CA"], coord["C"], coord["O"]], 0))
            aa1 = _AA3TO1.get(str(chain.rname[i])[:3], "X")
            Ss.append(restype_str_to_int.get(aa1, restype_str_to_int["X"]))
            ridxs.append(int(chain.ridx[i]))
            labels.append(ci)
            letters.append(chain_id)
        ci += 1

    L = len(Xs)
    pd = {
        "X": torch.tensor(np.stack(Xs, 0), dtype=torch.float32, device=device),
        "S": torch.tensor(Ss, dtype=torch.int32, device=device),
        "mask": torch.ones(L, dtype=torch.int32, device=device),
        "R_idx": torch.tensor(ridxs, dtype=torch.int32, device=device),
        "chain_labels": torch.tensor(labels, dtype=torch.int32, device=device),
        "chain_letters": letters,
        "mask_c": [
            torch.tensor([cl == cid for cl in letters], dtype=torch.bool, device=device)
            for cid in order
        ],
    }
    return pd, order


def run_lmpnn_redesign_batched(
    structs, binder_chain, model_type, binder_seqs, num_seqs=1, temperature=0.1, device=None
):
    """In-memory, GPU-batched binder redesign over many backbones (no PDB I/O).

    Builds LigandMPNN protein dicts directly from tinyprot structs and runs a single
    cached model over the whole batch via ``run.design_batched`` — the MPNN model is
    loaded once and reused across calls (see LIGANDMPNN_NO_CACHE), instead of being
    rebuilt from the checkpoint on every backbone. Each backbone's non-X binder
    positions are held fixed; ``num_seqs`` designs per backbone are produced by
    replicating it in the batch (distinct per-element noise → distinct sequences).

    Returns a list aligned with ``structs``, each a list of ``num_seqs`` redesigned
    binder-chain sequences. Requires the promera-compatible LigandMPNN fork
    (https://github.com/bjing2016/LigandMPNN) which provides ``design_batched``."""
    if not structs:
        return []
    if _LMPNN_DIR not in sys.path:
        sys.path.insert(0, _LMPNN_DIR)
    try:
        lmpnn_run = importlib.import_module("run")
    except ImportError as e:
        raise ImportError(f"Could not import LigandMPNN from {_LMPNN_DIR}.\n{e}")
    if not hasattr(lmpnn_run, "design_batched"):
        raise ImportError(
            f"LigandMPNN at {_LMPNN_DIR} has no design_batched; use the "
            "promera-compatible fork: https://github.com/bjing2016/LigandMPNN"
        )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    lmpnn_model_type, checkpoint_protein_mpnn, mp = _lmpnn_resolve(model_type)
    cfg = SimpleNamespace(
        model_type=lmpnn_model_type,
        checkpoint_protein_mpnn=checkpoint_protein_mpnn,
        checkpoint_soluble_mpnn=os.path.join(mp, "solublempnn_v_48_020.pt"),
        checkpoint_ligand_mpnn=os.path.join(mp, "ligandmpnn_v_32_010_25.pt"),
        ligand_mpnn_use_side_chain_context=0,
        ligand_mpnn_use_atom_context=1,
        ligand_mpnn_cutoff_for_score=8.0,
        chains_to_design=binder_chain,
        omit_AA="C",
        temperature=temperature,
        seed=0,
        fasta_seq_separation=":",
    )

    n = max(1, int(num_seqs))
    protein_dicts, orders, owner = [], [], []
    for i, struct in enumerate(structs):
        for _ in range(n):
            pd, order = _struct_to_protein_dict(struct, device)
            protein_dicts.append(pd)
            orders.append(order)
            owner.append(i)
    fixed_list = [
        " ".join(
            f"{binder_chain}{k+1}"
            for k, aa in enumerate(binder_seqs[owner[j]])
            if aa != "X"
        )
        for j in range(len(owner))
    ]
    redes_list = [""] * len(protein_dicts)

    raw = lmpnn_run.design_batched(cfg, protein_dicts, redes_list, fixed_list)

    out = [[] for _ in structs]
    for j, full_seq in enumerate(raw):
        parts = full_seq.split(cfg.fasta_seq_separation)
        out[owner[j]].append(parts[orders[j].index(binder_chain)])
    return out

from tinyprot.geometry import get_contact_mask, compute_rmsd
from tinyprot.metrics import dockQ as _tp_dockq, LDDT as _tp_lddt


def _binder_ca_coords(struct, binder_chain):
    chain = struct.chains[binder_chain]
    ca = []
    for i in range(len(chain.aname)):
        anames = [str(a) for a in chain.aname[i]]
        if "CA" in anames:
            ca.append(chain.coords[i, anames.index("CA")])
    return np.asarray(ca, dtype=np.float64)


def compute_interface_contacts(
    struct,
    binder_chain,
    paratope_positions,
    epitope_chain=None,
    epitope_positions=None,
    thresh=5.0,
):
    binder = struct.chains[binder_chain]
    target_chains = {
        k: c
        for k, c in struct.chains.items()
        if k != binder_chain and c.type.startswith("polymer")
    }

    binder_contact = np.zeros(len(binder.rname), dtype=bool)
    target_contact = {
        k: np.zeros(len(c.rname), dtype=bool) for k, c in target_chains.items()
    }

    for k, tc in target_chains.items():
        mask = np.asarray(get_contact_mask(binder, tc, thresh=thresh))
        binder_contact |= mask.any(axis=1)
        target_contact[k] |= mask.any(axis=0)

    epitope_contacts = 0
    epitope_residues = 0
    if epitope_chain and epitope_positions is not None:
        epi_mask = target_contact.get(epitope_chain, np.zeros(0, dtype=bool))
        epitope_contacts = int(epi_mask[epitope_positions].sum())
        epitope_residues = int(len(epitope_positions))

    return {
        "paratope_residues": int(len(paratope_positions)),
        "paratope_contacts": int(binder_contact[paratope_positions].sum()),
        "binder_contacts": int(binder_contact.sum()),
        "target_contacts": int(sum(m.sum() for m in target_contact.values())),
        "epitope_residues": epitope_residues,
        "epitope_contacts": epitope_contacts,
    }


def compute_self_consistency_rmsd(ref_struct, pred_struct, binder_chain):
    a = _binder_ca_coords(ref_struct, binder_chain)
    b = _binder_ca_coords(pred_struct, binder_chain)
    return float(compute_rmsd(a, b))


_BB_ATOMS = ("N", "CA", "C", "O")


def _strip_to_bb(struct):
    """Return a deep copy with every protein chain reduced to N/Cα/C/O atoms.

    The diffusion backbone has UNK residues at paratope positions and only BB
    atoms placed there — stripping the refold to BB too lets tinyprot's dockQ
    compare equivalent atom sets without tripping on UNK aname mismatches.
    """
    import copy as _copy
    out = _copy.deepcopy(struct)
    for chain in out.chains.values():
        if not chain.type.startswith("polymer:polypeptide"):
            continue
        n_res = len(chain.aname)
        new_aname = np.full((n_res, len(_BB_ATOMS)), "", dtype=chain.aname.dtype)
        new_coords = np.zeros((n_res, len(_BB_ATOMS), 3), dtype=chain.coords.dtype)
        new_mask = np.zeros((n_res, len(_BB_ATOMS)), dtype=chain.mask.dtype)
        for i in range(n_res):
            anames_i = [str(a) for a in chain.aname[i]]
            for k, atom in enumerate(_BB_ATOMS):
                new_aname[i, k] = atom
                if atom in anames_i:
                    j = anames_i.index(atom)
                    new_coords[i, k] = chain.coords[i, j]
                    new_mask[i, k] = chain.mask[i, j]
        chain.aname = new_aname
        chain.coords = new_coords
        chain.mask = new_mask
    return out


def compute_dockq(ref_struct, pred_struct):
    """Per-pair DockQ on backbone atoms only so UNK paratope residues in the
    diffusion backbone match the refold's standard residues at the atom level.

    Uses tinyprot dockQ's bb_only mode, which widens the contact/interface
    thresholds (8 Å / 13 Å) for backbone-only atoms so the interface contact
    mask doesn't collapse to empty."""
    ref_bb = _strip_to_bb(ref_struct)
    pred_bb = _strip_to_bb(pred_struct)
    result = _tp_dockq(ref_bb, pred_bb, exclude_nonstd=False, bb_only=True)
    return {
        f"{a}_{b}": {k: float(v) for k, v in scores.items()}
        for (a, b), scores in result.items()
    }


def compute_target_lddt(template_chain, pred_struct, target_chain_id, epitope_positions=None):
    """All-atom LDDT of the predicted target chain against the template chain.

    If epitope_positions is given (0-based indices), LDDT is computed only over
    those residues.
    """
    pred_chain = pred_struct.chains[target_chain_id]
    if epitope_positions is not None and len(epitope_positions) > 0:
        mask = np.zeros(len(template_chain.rname), dtype=bool)
        mask[epitope_positions] = True
        template_chain = template_chain.residue_slice(mask)
        pred_chain = pred_chain.residue_slice(mask)
    result = _tp_lddt(template_chain, pred_chain)
    return float(result["LDDT"])
