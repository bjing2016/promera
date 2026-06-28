import torch, tqdm
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_edm_sched_fn(cfg):
    def edm_sched_fn(t):
        p = cfg.rho
        sigma_max = cfg.sigma_max * cfg.sigma_data
        sigma_min = cfg.sigma_min * cfg.sigma_data
        return (
            sigma_min ** (1 / p)
            + (1 - t) * (sigma_max ** (1 / p) - sigma_min ** (1 / p))
        ) ** p

    return edm_sched_fn


class Sampler:
    def __init__(
        self,
        schedules,
        steppers,
    ):
        self.schedules = schedules
        self.steppers = steppers

    def sample(self, model, noisy_batch, steps=100, trunc=None, pbar=True):

        from collections import defaultdict

        extra = defaultdict(list)
        steps = np.linspace(0, 1, steps + 1)
        steps = list(zip(steps[:-1], steps[1:]))
        if pbar:
            it = tqdm.tqdm(steps)
        else:
            it = iter(steps)
        for t, s in it:
            if trunc is not None and t < trunc:
                continue
            sched = {key: (sched(t), sched(s)) for key, sched in self.schedules.items()}

            noisy_batch = self.single_step(model, noisy_batch, sched, extra)

        return noisy_batch, extra

    # step from t to s
    def single_step(self, model_fn, noisy_batch, sched, extra={}, sc=True):

        for stepper in self.steppers:
            stepper.set_step(noisy_batch, sched, extra)

        readout = model_fn(noisy_batch)

        for stepper in self.steppers:
            stepper.advance(noisy_batch, sched, readout, extra)

        return noisy_batch


class EDMDiffusionStepper:
    def __init__(self, cfg=None, mask=None):
        self.cfg = cfg
        self.mask = mask

    def set_step(self, batch, sched, extra={}):

        t, s = sched["coords"]

        cfg = self.cfg
        if cfg.edm_churn:
            gamma = cfg.gamma_0 if s > cfg.gamma_min else 0
            t_hat = t * (gamma + 1)

            noise = (
                cfg.noise_scale
                * np.sqrt(t_hat**2 - t**2)
                * torch.randn_like(batch["coords"])
            )

            batch["coords"] += noise
            batch["coords_sigma"][:] = t_hat
        else:
            batch["coords_sigma"][:] = t

    def advance(self, batch, sched, out, extra={}):

        cfg = self.cfg
        if cfg.edm_churn:
            t, s = sched["coords"]
            x = batch["coords"]
            x0 = out["coords"]
            # boltz alignment
            from ..model.loss.diffusion import weighted_rigid_align

            # coords carry the multiplicity-expanded batch (B * diffusion_samples)
            # while atom_pad_mask is still the un-expanded batch (B). Repeat it to
            # match so the per-sample alignment broadcasts correctly for B > 1
            # (at B == 1 the mask broadcast happened to work already).
            atom_mask = batch["atom_pad_mask"].float()
            if atom_mask.shape[0] != x.shape[0]:
                atom_mask = atom_mask.repeat_interleave(
                    x.shape[0] // atom_mask.shape[0], 0
                )

            with torch.autocast("cuda", enabled=False):
                x = weighted_rigid_align(
                    x.float(),
                    x0.float(),
                    atom_mask,
                    atom_mask,
                )
            t_hat = batch["coords_sigma"]
            delta = (x0 - x) / t_hat[..., None, None]
            dt = t_hat - s

            dx = cfg.step_scale * dt[..., None, None] * delta

        else:
            x = batch["coords"]
            t2, t1 = sched["coords"]

            dt = t2 - t1
            g = np.sqrt(2 * t2)

            x0 = out["coords"]

            s = (x0 - x) / t2**2  # score
            noise = torch.randn_like(x)
            gamma = self.cfg.temp_factor

            w = 2 * (t2 / self.cfg.sigma_data + 1)
            w = max(w, 1)
            w = 0
            ode = 0.5 * g**2 * s * dt
            sde = 0.5 * g**2 * s * dt + g * gamma * np.sqrt(dt) * noise
            dx = ode + w * sde

        extra["traj"].append(x0)
        extra["noisy"].append(x)
        batch["coords"] = x + dx
