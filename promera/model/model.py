import torch
from torch import nn
from .layers import initialize as init
from .loss.distogram import distogram_loss
from .loss.contact import contact_module_loss
from .loss.confidence import symmetry_correction, compute_confidence_loss
from .encoders import RelativePositionEncoder
from .trunk import (
    DistogramModule,
    InputEmbedder,
    MSAModule,
    PairformerModule,
)
from .confidence import ConfidenceModule, ContactModule
import numpy as np
from .template import LightningModuleTemplate
from .diffusion import AtomDiffusion
from tinyprot.feature import _ntoks
from ..diffusion.sampler import Sampler, EDMDiffusionStepper, get_edm_sched_fn


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim, layers=2, act=nn.ReLU):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.LayerNorm(dim))
        if act == SwiGLU:
            out_mul = 2
        else:
            out_mul = 1
        self.layers.append(nn.Linear(dim, ff_dim * out_mul))
        for i in range(layers - 2):
            self.layers.append(act())
            self.layers.append(nn.Linear(ff_dim, ff_dim * out_mul))
        self.layers.append(act())
        self.layers.append(nn.Linear(ff_dim, dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PromeraModel(LightningModuleTemplate):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.cfg = cfg

        cfg = cfg.model

        # Input projections
        s_input_dim = cfg.dims.token_s + 2 * _ntoks + 2

        self.s_init = nn.Linear(s_input_dim, cfg.dims.token_s, bias=False)
        self.z_init_1 = nn.Linear(s_input_dim, cfg.dims.token_z, bias=False)
        self.z_init_2 = nn.Linear(s_input_dim, cfg.dims.token_z, bias=False)

        self.input_embedder = InputEmbedder(cfg.input_embedder)
        self.rel_pos = RelativePositionEncoder(cfg.dims.token_z)
        self.token_bonds = nn.Linear(1, cfg.dims.token_z, bias=False)

        # Normalization layers
        self.s_norm = nn.LayerNorm(cfg.dims.token_s)
        self.z_norm = nn.LayerNorm(cfg.dims.token_z)

        # Recycling projections
        self.s_recycle = nn.Linear(cfg.dims.token_s, cfg.dims.token_s, bias=False)
        self.z_recycle = nn.Linear(cfg.dims.token_z, cfg.dims.token_z, bias=False)

        init.gating_init_(self.s_recycle.weight)
        init.gating_init_(self.z_recycle.weight)

        # Pairwise stack
        self.msa_module = MSAModule(
            token_z=cfg.dims.token_z,
            s_input_dim=s_input_dim,
            **cfg.msa_args,
        )

        if cfg.disto_embed:
            self.disto_embed = nn.Embedding(51, cfg.dims.token_z)

        self.pairformer_module = PairformerModule(cfg.pairformer_args)

        self.distogram_module = DistogramModule(cfg.dims.token_z, cfg.num_bins)

        # Output modules
        if cfg.has_structure_module:
            self.structure_module = AtomDiffusion(cfg.structure_module_args)

        if cfg.has_confidence:

            self.inp_confidence_module = ConfidenceModule(
                cfg.confidence_module_args, distogram=False, inp_only=True
            )
            self.trunk_confidence_module = ConfidenceModule(
                cfg.confidence_module_args, distogram=False
            )
            self.sm_confidence_module = ConfidenceModule(
                cfg.confidence_module_args,
                distogram=True,
            )

        if getattr(cfg, "has_contact_module", False):
            self.contact_module = ContactModule(cfg.contact_module_args)

        if hasattr(self.cfg, "training") and self.cfg.training.confidence_only:
            self.requires_grad_(False)
            if self.cfg.model.has_confidence:
                self.inp_confidence_module.requires_grad_(True)
                self.trunk_confidence_module.requires_grad_(True)
                self.sm_confidence_module.requires_grad_(True)
            if getattr(self.cfg.model, "has_contact_module", False):
                self.contact_module.requires_grad_(True)

    def subsample_msa(self, feats, subsample=None):
        subsample = subsample or self.cfg.model.msas_per_trunk_iter
        new = {**feats}

        msa = feats["msa"]
        B, S = msa.shape[0], msa.shape[1]
        k = min(S, subsample)

        # With batch_size > 1, collate pads shallower MSAs up to the batch-max
        # depth S with masked zero rows. Subsampling must therefore be done
        # PER batch element over each item's own real rows -- a single shared
        # index set drawn from arange(S) would spend most of a shallow item's
        # quota on its padded (masked) rows, silently shrinking its effective
        # MSA. Real rows are those with any unmasked position; collate appends
        # padding at the end, and any leftover slots point at a padded (masked)
        # row so they contribute nothing -- reproducing the bs=1 behaviour
        # (each item keeps min(real_depth, subsample) real sequences) exactly.
        row_real = feats["msa_mask"].any(dim=-1).cpu().numpy()  # [B, S] bool
        idx = np.zeros((B, k), dtype=np.int64)
        for b in range(B):
            real = np.nonzero(row_real[b])[0]
            if real.size == 0:
                real = np.array([0])
            kb = min(k, real.size)
            idx[b, :kb] = np.random.choice(real, size=kb, replace=False)
            if kb < k:
                padded = np.nonzero(~row_real[b])[0]
                idx[b, kb:] = padded[0] if padded.size else 0

        idx_t = torch.from_numpy(idx).to(msa.device)
        for key in [
            "msa",
            "msa_mask",
            "msa_paired",
            "deletion_value",
            "has_deletion",
        ]:
            v = feats[key]
            gather_idx = idx_t.view(B, k, *([1] * (v.dim() - 2))).expand(
                B, k, *v.shape[2:]
            )
            new[key] = torch.nn.functional.pad(
                torch.gather(v, 1, gather_idx), (0, 0, 0, subsample - k)
            )

        return new

    def pairformer_forward(self, feats, recycling_steps, no_grad=False, subsample=None):
        dict_out = {}

        # Compute input embeddings
        s_inputs = self.input_embedder(feats)

        # Initialize the sequence and pairwise embeddings
        s_init = self.s_init(s_inputs)
        z_init = (
            self.z_init_1(s_inputs)[:, :, None] + self.z_init_2(s_inputs)[:, None, :]
        )
        relative_position_encoding = self.rel_pos(feats)
        z_init = z_init + relative_position_encoding

        if len(feats["token_bonds"].shape) == 4:
            token_bonds = feats["token_bonds"].float()
        else:  # normally the case
            token_bonds = feats["token_bonds"].unsqueeze(-1).float()
        z_init = z_init + self.token_bonds(token_bonds)

        if self.cfg.model.disto_embed and "distogram_emb" in feats:
            z_init = z_init + torch.where(
                feats["distogram_mask"][..., None],
                self.disto_embed(feats["distogram_emb"]),
                0.0,
            )

        # Perform rounds of the pairwise stack
        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)

        # Compute pairwise mask
        mask = feats["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]

        pdistogram = []
        pcontact = []
        logits = []
        pair_logits = []

        for i in range(recycling_steps):
            enable_grad = self.training and i == recycling_steps - 1 and not no_grad
            with torch.set_grad_enabled(enable_grad):
                # Fixes an issue with unused parameters in autocast
                if enable_grad and torch.is_autocast_enabled():
                    torch.clear_autocast_cache()

                s = s_init + self.s_recycle(self.s_norm(s))
                z = z_init + self.z_recycle(self.z_norm(z))

                if self.cfg.model.subsample_msa_per_recycle:
                    feats_this_recycle = self.subsample_msa(feats, subsample=subsample)
                    s_msa, z_msa = self.msa_module(z, s_inputs, feats_this_recycle)
                else:
                    s_msa, z_msa = self.msa_module(z, s_inputs, feats)

                z = z + z_msa

                s, z = self.pairformer_module(s, z, mask=mask, pair_mask=pair_mask)
                pdistogram.append(self.distogram_module(z))

        dict_out = {
            "s": s,
            "z": z,
            "z_init": z_init,
            "s_init": s_init,
            "relpos": relative_position_encoding,
            "s_inputs": s_inputs,
            "pdistogram": pdistogram,
            "pcontact": pcontact,
            "logits": logits,
            "pair_logits": pair_logits,
        }

        return dict_out

    def add_noise(self, feats, multiplicity=1):

        from .utils import center_random_augmentation

        def noise_distribution(cfg, batch_size):
            rand = torch.randn((batch_size,), device=self.device)
            return cfg.sigma_data * (cfg.P_mean + cfg.P_std * rand).exp()

        batch_size = feats["atom_coords"].shape[0]

        if self.cfg.training.synchronize_sigmas:
            sigmas = noise_distribution(
                self.cfg.diffusion, batch_size
            ).repeat_interleave(multiplicity, 0)
        else:
            sigmas = self.noise_distribution(batch_size * multiplicity)

        padded_sigmas = sigmas.reshape(-1, 1, 1)

        atom_coords = feats["atom_coords"]
        B, L, _ = atom_coords.shape
        atom_coords = atom_coords.repeat_interleave(multiplicity, 0)
        feats["atom_coords"] = atom_coords

        atom_mask = feats["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        atom_coords = center_random_augmentation(
            atom_coords,
            atom_mask,
            augmentation=self.cfg.training.coordinate_augmentation,
        )
        feats["augmented_coords"] = atom_coords

        noise = torch.randn_like(atom_coords)
        noised_atom_coords = atom_coords + padded_sigmas * noise

        feats["noised_atom_coords"] = noised_atom_coords
        feats["sigmas"] = sigmas
        return feats

    # dict_keys(['ref_pos', 'ref_element', 'ref_charge', 'ref_atom_name', 'ref_space_uid', 'atom_to_token', 'token_to_rep_atom', 'frames_idx', 'frames_mask', 'atom_coords', 'atom_resolved_mask', 'token_bonds', 'ref_atom_name_chars', 'residue_index', 'residue_name', 'asym_id', 'entity_id', 'asym_id_', 'entity_id_', 'sym_id', 'restype', 'is_protein', 'is_rna', 'is_dna', 'is_ligand', 'is_std', 'token_index', 'token_pos', 'token_pos_mask', 'msa_chars', 'msa', 'msa_mask', 'msa_paired', 'deletion_value', 'profile', 'deletion_mean', 'has_deletion', 'atom_pad_mask', 'token_pad_mask', 'token_pair_supervise', 'atom_is_protein', 'atom_is_rna', 'atom_is_dna', 'atom_is_ligand', 'atom_supervise', 'name', 'is_pseudo_complex'])

    def prepare_batch(self, batch):

        if np.random.rand() < self.cfg.training.partial_distogram_prob:
            t_dists = torch.cdist(batch["token_pos"], batch["token_pos"])
            boundaries = torch.linspace(1, 50, 50).to(t_dists)
            target = (t_dists.unsqueeze(-1) > boundaries).sum(dim=-1).long()
            mask = torch.rand_like(batch["token_pad_mask"], dtype=float) < 0.15
            mask = mask[:, None] & mask[:, :, None] & batch["distogram_supervise"]
            batch["distogram_mask"] = mask
            batch["distogram_emb"] = torch.where(mask, target, 0)
        else:
            batch["distogram_mask"] = torch.zeros_like(batch["distogram_supervise"])
            batch["distogram_emb"] = batch["distogram_mask"].long()

        if np.random.rand() < self.cfg.training.epitope_prob:

            contacts = torch.cdist(batch["token_pos"], batch["token_pos"]) < 10.0
            mask = contacts & (
                batch["asym_id"][..., None] != batch["asym_id"][..., None, :]
            )
            mask = mask.sum(-1) > 0
            batch["is_epitope"] = mask & (
                torch.rand_like(mask, dtype=float)
                < 0.2  # magic number from RFDiffusion
            )  # hotspot
        else:
            batch["is_epitope"] = torch.zeros_like(batch["token_pad_mask"])

    def training_step(self, batch, batch_idx):

        self.prepare_batch(batch)

        loss = 0

        recycling_steps = int(
            self.rng.integers(1, self.cfg.training.recycling_steps + 1)
        )
        # Compute the forward pass
        out = self.pairformer_forward(batch, recycling_steps)

        self._logger.log(
            "s_norm",
            out["s"].square().mean().sqrt(),
        )
        self._logger.log(
            "z_norm",
            out["z"].square().mean().sqrt(),
        )

        if not self.cfg.training.confidence_only:
            # Distogram loss
            disto_loss = distogram_loss(out["pdistogram"][-1], batch)

            self._logger.log(
                "distogram_loss",
                disto_loss,
                mask=batch["distogram_supervise"].sum((-1, -2)) > 0,
            )
            loss = loss + self.cfg.loss.distogram_weight * disto_loss

        # Diffusion loss
        if (
            self.cfg.model.has_structure_module
            and not self.cfg.training.confidence_only
        ):
            mul = self.cfg.training.diffusion_multiplicity
            self.add_noise(batch, multiplicity=mul)

            out["s_condition"] = out["s"]
            out["z_condition"] = out["z"]

            out |= self.structure_module(batch, out, multiplicity=mul)

            self._logger.log(
                "denoised_coords_norm",
                out["denoised_atom_coords"].square().mean().sqrt(),
            )

            with torch.autocast("cuda", enabled=False):
                diffusion_loss = self.structure_module.compute_loss(
                    batch,
                    out,
                    multiplicity=mul,
                    **self.cfg.loss.diffusion,
                )

            loss = loss + self.cfg.loss.mse_weight * diffusion_loss["mse_loss"]
            loss = loss + self.cfg.loss.lddt_weight * diffusion_loss["smooth_lddt_loss"]

            # can add mask to these as well
            self._logger.log("mse_loss", diffusion_loss["mse_loss"])
            self._logger.log("lddt_loss", diffusion_loss["smooth_lddt_loss"])

        if self.cfg.model.has_confidence or self.cfg.model.has_contact_module:

            with torch.no_grad():
                out |= self.sample_diffusion(batch, out, self.cfg.training)

            if "alt_coords" in batch:
                with torch.no_grad():
                    symmetry_corrected_coords, symmetry_corrected_mask = (
                        symmetry_correction(
                            out["sample_atom_coords"],
                            batch,
                            multiplicity=self.cfg.training.diffusion_samples,
                        )
                    )
            else:
                symmetry_corrected_coords = batch["atom_coords"]
                symmetry_corrected_mask = batch["atom_resolved_mask"]

        if self.cfg.model.has_confidence:
            inp_confidence = self.inp_confidence_module(
                batch,
                out,
                multiplicity=self.cfg.training.diffusion_samples,
            )
            trunk_confidence = self.trunk_confidence_module(
                batch,
                out,
                multiplicity=self.cfg.training.diffusion_samples,
            )

            sm_confidence = self.sm_confidence_module(
                batch,
                out,
                multiplicity=self.cfg.training.diffusion_samples,
            )

            inp_confidence_loss = compute_confidence_loss(
                out | inp_confidence,
                batch,
                symmetry_corrected_coords,
                symmetry_corrected_mask,
                multiplicity=self.cfg.training.diffusion_samples,
                alpha_pae=1.0,
            )
            trunk_confidence_loss = compute_confidence_loss(
                out | trunk_confidence,
                batch,
                symmetry_corrected_coords,
                symmetry_corrected_mask,
                multiplicity=self.cfg.training.diffusion_samples,
                alpha_pae=1.0,
            )
            sm_confidence_loss = compute_confidence_loss(
                out | sm_confidence,
                batch,
                symmetry_corrected_coords,
                symmetry_corrected_mask,
                multiplicity=self.cfg.training.diffusion_samples,
                alpha_pae=1.0,
            )

            for key in inp_confidence_loss:
                if key == "loss":
                    continue
                self._logger.log(f"inp_{key}", inp_confidence_loss[key])
                self._logger.log(f"trunk_{key}", trunk_confidence_loss[key])
                self._logger.log(f"sm_{key}", sm_confidence_loss[key])

            loss = loss + self.cfg.loss.confidence_weight * (
                inp_confidence_loss["loss"]
                + trunk_confidence_loss["loss"]
                + sm_confidence_loss["loss"]
            )

        if self.cfg.model.has_contact_module:
            contact_out = self.contact_module(
                batch, out, multiplicity=self.cfg.training.diffusion_samples
            )
            cl, frac = contact_module_loss(
                contact_out["contact_logits"],
                contact_out["pred_dist"],
                batch,
                multiplicity=self.cfg.training.diffusion_samples,
            )
            self._logger.log("contact_loss", cl)
            loss = loss + self.cfg.loss.contact_weight * cl
            self._logger.log("frac_contacts", frac)

        self._logger.log("loss", loss)

        if (
            self.cfg.model.has_structure_module
            and (not self.cfg.training.confidence_only)
            and getattr(self.cfg.model, "center_of_mass_loss", False)
        ):
            com = (
                out["denoised_atom_coords"] * batch["atom_resolved_mask"][..., None]
            ).sum(1) / (1e-5 + batch["atom_resolved_mask"][..., None].sum(1))
            com_loss = com.square().sum(-1).sqrt()

            loss = loss + self.cfg.model.center_of_mass_loss_weight * com_loss.mean()
            self._logger.log("com_loss", com_loss)
            self._logger.log("loss_with_com_loss", loss)

        return loss

    def sample_diffusion(self, batch, out, cfg):
        # Diffusion sampling runs in fp32 by default (autocast disabled) for
        # numerical stability — the EDM stepper's weighted_rigid_align SVD is
        # always done in fp32 (see weighted_rigid_align), but with fp16 the
        # denoiser network overflows and feeds NaN coords into that SVD. bf16 has
        # fp32 range so it is stable and accurate. The diffusion precision follows
        # cfg.amp; set cfg.amp_diffusion to override it (e.g. "fp32" to keep the
        # denoiser fp32 while the trunk runs bf16). fp16 is NOT recommended.
        amp = getattr(cfg, "amp_diffusion", None) or getattr(cfg, "amp", None)
        amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
        with torch.autocast(
            "cuda", enabled=amp_dtype is not None, dtype=amp_dtype or torch.float16
        ):
            return self._sample_diffusion_impl(batch, out, cfg)

    def _sample_diffusion_impl(self, batch, out, cfg):

        edm_sched_fn = get_edm_sched_fn(cfg.diffusion)
        sampler = Sampler(
            schedules={"coords": edm_sched_fn},
            steppers=[EDMDiffusionStepper(cfg.diffusion)],
        )
        atom_mask = batch["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(cfg.diffusion_samples, 0)
        shape = (*atom_mask.shape, 3)
        batch["coords"] = edm_sched_fn(0) * torch.randn(shape, device=self.device)
        batch["coords_sigma"] = edm_sched_fn(0) * torch.ones(
            shape[0], device=self.device
        )
        model_cache = {}

        def model_func(batch):
            from .utils import compute_random_augmentation

            atom_coords = batch["coords"]
            random_R, random_tr = compute_random_augmentation(
                atom_coords.shape[0], device=self.device, dtype=atom_coords.dtype
            )
            atom_coords = atom_coords - atom_coords.mean(dim=-2, keepdims=True)
            atom_coords = (
                torch.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr
            )
            batch["coords"] = atom_coords

            key = "denoised_atom_coords"
            pred = self.structure_module.preconditioned_network_forward(
                batch["coords"],
                batch["coords_sigma"],
                training=False,
                network_condition_kwargs=dict(
                    s_inputs=batch["s_inputs"],
                    s_trunk=batch["s"],
                    z_trunk=batch["z"],
                    relative_position_encoding=batch["relpos"],
                    feats=batch,
                    multiplicity=cfg.diffusion_samples,
                    model_cache=model_cache,
                ),
            )[key]
            return {"coords": pred}

        sample, extra = sampler.sample(
            model_func,
            batch | out,
            cfg.diffusion_steps,
            pbar=False,
        )

        result = {
            "sample_atom_coords": sample["coords"],
            "sample_trajectory": torch.stack(extra["traj"]).transpose(0, 1),
            "sample_noisy": torch.stack(extra["noisy"]).transpose(0, 1),
        }
        return result
