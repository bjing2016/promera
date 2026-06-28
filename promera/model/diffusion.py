# Adapted from https://github.com/jwohlwend/boltz
# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

from __future__ import annotations


import types

from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module


def _make_layernorms_lowp(module):
    """Patch every nn.LayerNorm in `module` to run in the active autocast dtype
    (e.g. bf16) instead of being upcast to fp32 by autocast (which wraps each in
    an fp32<->bf16 cast pair). F.layer_norm accumulates in fp32 internally, so
    bf16 IO stays accurate; with autocast off it is byte-identical to before."""

    def _lowp_layernorm_forward(self, x):
        if torch.is_autocast_enabled() and x.dtype != torch.float32:
            with torch.autocast("cuda", enabled=False):
                w = self.weight.to(x.dtype) if self.weight is not None else None
                b = self.bias.to(x.dtype) if self.bias is not None else None
                return F.layer_norm(x, self.normalized_shape, w, b, self.eps)
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    for m in module.modules():
        if isinstance(m, nn.LayerNorm):
            m.forward = types.MethodType(_lowp_layernorm_forward, m)

from .layers import initialize as init
from .loss.diffusion import (
    smooth_lddt_loss,
    weighted_rigid_align,
)
from .encoders import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    PairwiseConditioning,
    SingleConditioning,
)
from .transformers import (
    DiffusionTransformer,
)
from .utils import (
    LinearNoBias,
    log,
)


class DiffusionModule(Module):
    """Diffusion module"""

    def __init__(
        self,
        token_s: int,
        token_z: int,
        atom_s: int,
        atom_z: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        sigma_data: int = 16,
        dim_fourier: int = 256,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        atom_feature_dim: int = 128,
        conditioning_transition_layers: int = 2,
        activation_checkpointing: bool = False,
        offload_to_cpu: bool = False,
        has_alt_update: bool = False,
        **kwargs,
    ) -> None:

        super().__init__()

        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.sigma_data = sigma_data

        self.single_conditioner = SingleConditioning(
            sigma_data=sigma_data,
            token_s=token_s,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers,
        )
        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=conditioning_transition_layers,
        )

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_feature_dim=atom_feature_dim,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=True,
            activation_checkpointing=activation_checkpointing,
        )

        self.s_to_a_linear = nn.Sequential(
            nn.LayerNorm(2 * token_s), LinearNoBias(2 * token_s, 2 * token_s)
        )
        init.final_init_(self.s_to_a_linear[1].weight)

        self.token_transformer = DiffusionTransformer(
            dim=2 * token_s,
            dim_single_cond=2 * token_s,
            dim_pairwise=token_z,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
            activation_checkpointing=activation_checkpointing,
            offload_to_cpu=offload_to_cpu,
        )

        self.a_norm = nn.LayerNorm(2 * token_s)

        self.atom_attention_decoder = AtomAttentionDecoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
            has_alt_update=has_alt_update,
        )

    def forward(
        self,
        s_inputs,
        s_trunk,
        z_trunk,
        r_noisy,
        times,
        relative_position_encoding,
        feats,
        multiplicity=1,
        model_cache=None,
    ):

        s, normed_fourier = self.single_conditioner(
            times=times,
            s_trunk=s_trunk.repeat_interleave(multiplicity, 0),
            s_inputs=s_inputs.repeat_interleave(multiplicity, 0),
        )

        if model_cache is None or len(model_cache) == 0:
            z = self.pairwise_conditioner(
                z_trunk=z_trunk, token_rel_pos_feats=relative_position_encoding
            )
        else:
            z = None

        # Compute Atom Attention Encoder and aggregation to coarse-grained tokens
        a, q_skip, c_skip, p_skip, to_keys = self.atom_attention_encoder(
            feats=feats,
            s_trunk=s_trunk,
            z=z,
            r=r_noisy,
            multiplicity=multiplicity,
            model_cache=model_cache,
        )

        # Full self-attention on token level
        a = a + self.s_to_a_linear(s)

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        a = self.token_transformer(
            a,
            mask=mask.float(),
            s=s,
            z=z,  # note z is not expanded with multiplicity until after bias is computed
            multiplicity=multiplicity,
            model_cache=model_cache,
        )
        a = self.a_norm(a)

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        r_update = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            p=p_skip,
            feats=feats,
            multiplicity=multiplicity,
            to_keys=to_keys,
            model_cache=model_cache,
        )

        return {"r_update": r_update, "token_a": a}


class AtomDiffusion(Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.score_model = DiffusionModule(**cfg.score_model_args, **cfg.dims)
        # Run the denoiser's LayerNorms in the autocast (bf16) dtype rather than
        # letting autocast upcast them to fp32 and wrap every one in a cast pair.
        # F.layer_norm accumulates in fp32 internally so bf16 IO stays accurate;
        # the fp32 path is byte-identical. This removes the dominant cast overhead
        # that otherwise caps bf16 diffusion throughput at saturation.
        _make_layernorms_lowp(self.score_model)
        # Whether to torch.compile the token transformer; done lazily on the first
        # forward (see preconditioned_network_forward) so it happens AFTER the
        # checkpoint is loaded — compiling here in __init__ renames the submodule's
        # params (_orig_mod.*) and the checkpoint then fails to load into it.
        self._compile_token_transformer = bool(cfg.compile_score)
        self._tt_compiled = False
        # Lazily-built CUDA-graph runner for the score model (inference only).
        # Built on first forward (after weights are loaded) so score_model stays
        # a normal submodule for checkpoint loading. See cudagraph.py.
        self._cg_runner = None

    @property
    def device(self):
        return next(self.score_model.parameters()).device

    def c_skip(self, sigma):
        sigma_data = self.cfg.diffusion.sigma_data
        return (sigma_data**2) / (sigma**2 + sigma_data**2)

    def c_out(self, sigma):
        sigma_data = self.cfg.diffusion.sigma_data
        return sigma * sigma_data / torch.sqrt(sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        sigma_data = self.cfg.diffusion.sigma_data
        return 1 / torch.sqrt(sigma**2 + sigma_data**2)

    def c_noise(self, sigma):
        sigma_data = self.cfg.diffusion.sigma_data
        return log(sigma / sigma_data) * 0.25

    def preconditioned_network_forward(
        self,
        noised_atom_coords,
        sigma,
        network_condition_kwargs: dict,
        training: bool = True,
    ):
        batch, device = noised_atom_coords.shape[0], noised_atom_coords.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1")

        # Lazily compile ONLY the token transformer (the 24-layer compute bulk of
        # the denoiser, all standard ops) on the first inference forward, after the
        # checkpoint is loaded. Compiling the *whole* score model is numerically
        # unsafe: Inductor mis-lowers the conditioning/atom path (Fourier cos
        # embedding + atom-encoder index gathers) and the small per-step error
        # compounds over the ~200-step rollout into a broken structure (verified:
        # whole-model compile -> LDDT ~0.07; token transformer only -> ~0.86). The
        # token transformer alone fuses safely and yields ~2.2x diffusion at
        # saturation.
        if self._compile_token_transformer and not self._tt_compiled and not training:
            self.score_model.token_transformer = torch.compile(
                self.score_model.token_transformer, dynamic=False, fullgraph=False
            )
            self._tt_compiled = True

        # CUDA-graph the score model during inference: the rollout calls it
        # identically every step (only r_noisy/times change), so replaying a
        # captured graph removes the per-step kernel-launch overhead that
        # dominates the (memory/launch-bound) diffusion. Numerically identical to
        # eager. Disabled during training and when alt updates are used.
        use_cudagraph = (
            bool(getattr(self.cfg, "cudagraph_score", False))
            and not training
            and not self.cfg.score_model_args.has_alt_update
        )
        if use_cudagraph:
            if self._cg_runner is None:
                from .cudagraph import CUDAGraphScoreModel

                self._cg_runner = CUDAGraphScoreModel(self.score_model)
            score_fn = self._cg_runner
        else:
            score_fn = self.score_model

        net_out = score_fn(
            r_noisy=self.c_in(padded_sigma) * noised_atom_coords,
            times=self.c_noise(sigma),
            **network_condition_kwargs,
        )
        if self.cfg.score_model_args.has_alt_update:
            r_update, r_update_alt = net_out["r_update"]
        else:
            r_update = net_out["r_update"]

        out = {
            "denoised_atom_coords": self.c_skip(padded_sigma) * noised_atom_coords
            + self.c_out(padded_sigma) * r_update
        }
        if self.cfg.score_model_args.has_alt_update:
            out["denoised_atom_coords_alt"] = (
                self.c_skip(padded_sigma) * noised_atom_coords
                + self.c_out(padded_sigma) * r_update_alt
            )
        return out
        # return denoised_coords, net_out["token_a"]

    def loss_weight(self, sigma):
        sigma_data = self.cfg.diffusion.sigma_data
        return (sigma**2 + sigma_data**2) / ((sigma * sigma_data) ** 2)

    def forward(
        self, feats, out, multiplicity=1
    ):  # this forward is never used in inference
        network_condition_kwargs = dict(
            s_inputs=out["s_inputs"],
            s_trunk=out["s_condition"],
            z_trunk=out["z_condition"],
            relative_position_encoding=out["relpos"],
            feats=feats,
            multiplicity=multiplicity,
        )
        network_out = self.preconditioned_network_forward(
            feats["noised_atom_coords"],
            feats["sigmas"],
            training=True,
            network_condition_kwargs=network_condition_kwargs,
        )

        return dict(
            noised_atom_coords=feats["noised_atom_coords"],
            **network_out,
            sigmas=feats["sigmas"],
            aligned_true_atom_coords=feats["augmented_coords"],
        )

    def compute_loss(
        self,
        feats,
        out_dict,
        nucleotide_loss_weight=5.0,
        ligand_loss_weight=10.0,
        multiplicity=1,
        alt=False,
        align=True,
    ):
        if alt:
            denoised_atom_coords = out_dict["denoised_atom_coords_alt"]
        else:
            denoised_atom_coords = out_dict["denoised_atom_coords"]
        noised_atom_coords = out_dict["noised_atom_coords"]
        sigmas = out_dict["sigmas"]

        resolved_atom_mask = feats["atom_resolved_mask"]
        resolved_atom_mask = resolved_atom_mask.repeat_interleave(multiplicity, 0)

        align_weights = (
            feats["atom_is_protein"]
            + nucleotide_loss_weight * (feats["atom_is_rna"] + feats["atom_is_dna"])
            + ligand_loss_weight * feats["atom_is_ligand"]
        )
        align_weights = align_weights.repeat_interleave(multiplicity, 0)

        if align:
            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                atom_coords = out_dict["aligned_true_atom_coords"]
                atom_coords_aligned_ground_truth = weighted_rigid_align(
                    atom_coords.detach().float(),
                    denoised_atom_coords.detach().float(),
                    align_weights.detach().float(),
                    mask=resolved_atom_mask.detach().float(),
                )

            # Cast back
            atom_coords_aligned_ground_truth = atom_coords_aligned_ground_truth.to(
                denoised_atom_coords
            )
        else:
            atom_coords_aligned_ground_truth = out_dict["aligned_true_atom_coords"]

        # weighted MSE loss of denoised atom positions
        mse_loss = ((denoised_atom_coords - atom_coords_aligned_ground_truth) ** 2).sum(
            dim=-1
        )

        mse_loss = mse_loss * feats["atom_supervise"].repeat_interleave(multiplicity, 0)
        mse_loss = torch.sum(
            mse_loss * align_weights * resolved_atom_mask, dim=-1
        ) / torch.sum(3 * align_weights * resolved_atom_mask, dim=-1)

        # weight by sigma factor
        loss_weights = self.loss_weight(sigmas)
        mse_loss = (mse_loss * loss_weights).mean()

        # total_loss = mse_loss

        # proposed auxiliary smooth lddt loss
        lddt_loss = smooth_lddt_loss(
            denoised_atom_coords,
            feats["atom_coords"],
            feats["atom_is_rna"] + feats["atom_is_dna"],
            coords_mask=feats["atom_resolved_mask"] * feats["atom_supervise"],
            multiplicity=multiplicity,
        )

        # total_loss = total_loss + lddt_loss

        return dict(
            mse_loss=mse_loss,
            smooth_lddt_loss=lddt_loss,
        )
