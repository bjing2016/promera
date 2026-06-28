import torch
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from typing import Optional
import functools
import pickle
from tinyprot.ccd import _get_attrs


def mask_chain(chain, mask_frac=0.15, msa=None):

    mask = np.random.rand(len(chain.rname)) < mask_frac
    for i in np.argwhere(mask):
        chain.rname[i] = "UNK"

        new_aname = _get_attrs("UNK", lambda at: at.GetProp("name"), False)
        new_asym = _get_attrs("UNK", lambda at: at.GetSymbol(), False)
        new_mask = np.zeros(len(new_aname), dtype=bool)
        new_coords = np.zeros((len(new_aname), 3), dtype=np.float32)

        for j in range(chain.aname.shape[1]):
            if chain.aname[i, j] in new_aname:
                k = new_aname.index(chain.aname[i, j])
                new_mask[k] = True
                new_coords[k] = chain.coords[i, j]

        chain.aname[i, :] = ""
        chain.asym[i, :] = ""
        chain.mask[i, :] = False
        chain.coords[i, :] = 0.0

        chain.aname[i, : len(new_aname)] = new_aname
        chain.asym[i, : len(new_aname)] = new_asym
        chain.mask[i, : len(new_aname)] = new_mask
        chain.coords[i, : len(new_aname)] = new_coords

        if msa is not None:
            msa.seqs[0, chain.ridx[i] - 1] = "X"
            msa.seqs[1:, chain.ridx[i] - 1] = "-"


_atom_keys = [
    "ref_pos",
    "ref_element",
    "ref_charge",
    "ref_space_uid",
    "ref_hydrogens",
    "atom_to_token",
    "ref_atom_name_chars",
    "atom_pad_mask",
    "atom_coords",
    "atom_resolved_mask",
    "atom_is_protein",
    "atom_is_rna",
    "atom_is_dna",
    "atom_is_ligand",
    "atom_is_std",
    "atom_supervise",
    "alt_coords",
    "alt_coords_mask",
]
_msa_keys = [
    "msa",
    "msa_mask",
    "msa_chars",
    "msa_paired",
    "deletion_value",
    "has_deletion",
]

_token_pair_keys = [
    "token_bonds",
    "token_contacts",
    "token_pair_supervise",
    "distogram_supervise",
    "contact_supervise",
]
_exclude_keys = [
    "label",
    "chain_sym_mapping",
    "chain_sym_mask",
]


def _pad_to_max(values, dim=0, multiple_of=1, pad_to=None):
    lens = [v.shape[dim] for v in values]
    pad_len = max(lens)
    if pad_to is not None:
        assert pad_to >= pad_len
        pad_len = max(pad_len, pad_to)
    if pad_len % multiple_of > 0:
        pad_len += multiple_of - pad_len % multiple_of
    for i in range(len(values)):
        v = values[i]
        shape = list(v.shape)
        shape[dim] = pad_len - shape[dim]
        zeros = np.zeros(shape, dtype=v.dtype)
        values[i] = np.concatenate([v, zeros], dim)
    return values


def get_collate_to_crop_max(cfg):
    return lambda data: collate(
        data, max_tokens=cfg.max_tokens, max_atoms=cfg.max_atoms, max_seqs=cfg.max_seqs
    )


def collate(data, max_tokens=None, max_atoms=None, max_seqs=None, token_multiple=1):
    """Stack a list of feature dicts into one padded batch.

    token_multiple: round the padded token count up to a multiple of this value
    (default 1 = pad to the batch max). Set >1 to stabilize the token dimension
    across batches so `torch.compile`d, token-dimensioned modules (the diffusion
    token transformer) see a constant input shape and recompile far less often —
    e.g. 32 collapses a narrow length spread (binder design) to a single compiled
    shape. Atoms already round to a multiple of 32; the compiled module is
    token-dimensioned, so only the token dim needs this."""

    if type(data[0]) is list:  # unpack lists
        data = [d for datum in data for d in datum]

    float_types = [np.float64, np.float32, np.float16]
    int_types = [np.int64, np.int32, np.int16, np.int8, np.uint8]

    # Get the keys
    keys = data[0].keys()
    # print(keys)
    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]
        if (type(values[0]) is not np.ndarray) or (
            values[0].dtype not in float_types + int_types + [np.bool_]
        ):
            collated[key] = values
            continue

        if key in _msa_keys:
            # Pad MSA depth (dim 0) to a common size so items with different
            # numbers of aligned sequences can be stacked in one batch. With
            # max_seqs unset this is the per-batch max depth; padded rows are
            # masked out via msa_mask. Tokens (dim 1) pad to max_tokens / batch.
            _pad_to_max(values, 0, pad_to=max_seqs)
            _pad_to_max(values, 1, multiple_of=token_multiple, pad_to=max_tokens)

        elif key in _atom_keys:
            _pad_to_max(values, 0, multiple_of=32, pad_to=max_atoms)

        elif key in _token_pair_keys:
            _pad_to_max(values, 0, multiple_of=token_multiple, pad_to=max_tokens)
            _pad_to_max(values, 1, multiple_of=token_multiple, pad_to=max_tokens)

        elif key in _exclude_keys:
            pass

        else:
            _pad_to_max(values, 0, multiple_of=token_multiple, pad_to=max_tokens)
        values = np.stack(values)
        if values.dtype in float_types:
            values = values.astype(np.float32)
        elif values.dtype in int_types:
            values = values.astype(np.int64)

        # Stack the values
        collated[key] = torch.from_numpy(values)
    return collated
