"""CUDA-graph capture/replay for the diffusion score model.

The reverse-diffusion rollout calls the score (denoiser) network the same way
~`diffusion_steps` times: identical shapes, identical control flow, and the same
trunk conditioning (s_trunk/z_trunk/s_inputs/relpos/feats) every step — only the
noisy coordinates `r_noisy` and the noise level `times` change. That is the ideal
shape for a CUDA graph: capture the kernel sequence once, then replay it for every
subsequent step with the launch overhead removed.

Why this and not torch.compile: a CUDA graph replays the *exact same* kernels, so
it is numerically identical to eager (no Inductor fusion/precision drift, which
compounds over the 200-step rollout). Capture is ~milliseconds, so there is no
multi-minute compile startup, and it re-captures cheaply when the target changes.

Correctness requirements that hold here:
- The score model is pure NN ops (no cuSOLVER/SVD, no host syncs); the per-step
  random augmentation, EDM noise, and rigid-align SVD all run in the sampler,
  *outside* this captured region.
- In eval there is no dropout/RNG inside the score model, so replay is exact.
- The cross-step cache (z projection / atom keys) depends only on the constant
  conditioning, so warming it up before capture and replaying the cache-hit path
  is equivalent to recomputing it every step.
"""

import torch


class CUDAGraphScoreModel:
    """Callable wrapper that CUDA-graphs `score_model(r_noisy, times, **const)`.

    Only `r_noisy` and `times` vary between steps; everything else is treated as
    constant for the lifetime of one capture and is re-captured automatically when
    it changes (new target/shape).
    """

    def __init__(self, score_model, warmup: int = 3):
        self.score_model = score_model
        self.warmup = warmup
        self.graph = None
        self._static_r = None
        self._static_t = None
        self._static_out = None
        self._sig = None
        self._disabled = False  # set if capture ever fails -> fall back to eager

    @staticmethod
    def _sig_of(r_noisy, times, kwargs):
        st = kwargs["s_trunk"]
        z = kwargs["z_trunk"]
        cache = kwargs.get("model_cache", None)
        # shapes + identity of the conditioning + the per-rollout cache dict
        # uniquely identify a capture. A new rollout/target changes the trunk
        # tensor addresses and/or installs a fresh cache dict, which forces a
        # re-capture (cheap) so we never replay a graph that points at freed
        # conditioning or stale cached tensors.
        return (
            tuple(r_noisy.shape),
            tuple(times.shape),
            r_noisy.dtype,
            tuple(st.shape),
            st.data_ptr(),
            z.data_ptr(),
            id(cache),
            int(kwargs.get("multiplicity", 1)),
        )

    def __call__(self, r_noisy, times, **kwargs):
        if self._disabled:
            return self.score_model(r_noisy=r_noisy, times=times, **kwargs)
        sig = self._sig_of(r_noisy, times, kwargs)
        if self.graph is None or sig != self._sig:
            try:
                self._capture(r_noisy, times, kwargs)
                self._sig = sig
            except Exception as e:  # fall back to eager so results stay correct
                print(f"[cudagraph] capture failed ({e}); falling back to eager")
                self._disabled = True
                self.graph = None
                return self.score_model(r_noisy=r_noisy, times=times, **kwargs)
        self._static_r.copy_(r_noisy)
        self._static_t.copy_(times)
        self.graph.replay()
        # Clone so the sampler can keep using the result while the next replay
        # overwrites the static output buffer.
        return {"r_update": self._static_out.clone()}

    def _capture(self, r_noisy, times, kwargs):
        import gc

        # Release any previous graph first so its handle on the shared memory
        # pool is dropped before we begin a new capture (otherwise capture_begin
        # asserts on the pool's use_count). Re-capture happens once per new
        # rollout/target, so this keeps memory bounded across a long run.
        self.graph = None
        self._static_out = None
        gc.collect()
        torch.cuda.synchronize()

        self._static_r = r_noisy.clone()
        self._static_t = times.clone()

        # Warm up on a side stream: this populates the cross-step cache (miss ->
        # hit) and lets cuBLAS/cuDNN pick algorithms before capture.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup):
                _ = self.score_model(
                    r_noisy=self._static_r, times=self._static_t, **kwargs
                )["r_update"]
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        # Use a private pool per capture (no shared graph_pool_handle): a fresh
        # capture must not share a pool that the just-released graph still held,
        # or capture_begin asserts on the pool use_count. The old graph was
        # dropped above, so its pool is freed before this allocates a new one.
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._static_out = self.score_model(
                r_noisy=self._static_r, times=self._static_t, **kwargs
            )["r_update"]
