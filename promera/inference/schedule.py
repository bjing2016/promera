import errno
import os
import time


def _log(msg):
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    print(f"[{time.strftime('%H:%M:%S')} rank{rank}] {msg}", flush=True)


class DynamicClaimBatchSampler:
    """A DDP-friendly work-queue batch sampler for embarrassingly-parallel inference.

    The default Lightning behaviour stripes dataset indices across ranks with a
    ``DistributedSampler`` *before* anything runs. That static split ignores how
    long each item actually takes: a rank that draws the short jobs goes idle
    while another rank is still grinding through big ones.

    This sampler replaces the static split with a single shared queue. Every rank
    pulls the next still-unclaimed items and processes them, so a rank that
    finishes a batch early immediately picks up more work — dynamic load
    balancing across both GPUs on a node and across nodes. Items are claimed in
    the dataset's existing order (the dataset sorts largest-first), so the longest
    jobs are dispatched before the short ones, which is the classic
    longest-processing-time greedy for minimising end-of-run stragglers.

    Coordination is a per-item directory atomically created on a shared
    filesystem (``mkdir`` is atomic on POSIX/NFS/Lustre), so no extra
    communication is needed — it composes with whatever process group Lightning
    sets up. Prediction issues no collective ops here, so ranks processing
    unequal item counts is fine.

    Note: this sampler intentionally does **not** implement ``__len__``. That
    makes the ``DataLoader`` report an unknown length, so Lightning runs the
    predict loop until each rank's iterator is exhausted (``StopIteration``)
    instead of capping every rank at a fixed batch count — which is what lets
    ranks legitimately process different numbers of items.
    """

    def __init__(self, num_items, batch_size, claim_dir, world_size=1, rank=0):
        self.num_items = int(num_items)
        self.batch_size = max(1, int(batch_size))
        self.claim_dir = claim_dir
        self.world_size = int(world_size)
        self.rank = int(rank)

        # A single process owns the whole queue, so it is safe to wipe any stale
        # claims left by a crashed run. With more than one rank we instead rely on
        # a run-unique claim_dir (keyed by the job id) to avoid stale entries.
        if self.world_size <= 1 and os.path.isdir(claim_dir):
            import shutil

            shutil.rmtree(claim_dir, ignore_errors=True)
        os.makedirs(claim_dir, exist_ok=True)

    def _claim(self, idx):
        """Atomically claim item `idx`; True if this rank won it, False if taken."""
        try:
            os.mkdir(os.path.join(self.claim_dir, f"{idx:09d}"))
            return True
        except OSError as e:
            if e.errno == errno.EEXIST:
                return False
            raise

    def __iter__(self):
        # Forward-only cursor: claims are monotonic (an item never un-claims), so a
        # rank never has to revisit an index it has already passed. Each rank does
        # at most one O(num_items) sweep total across the whole run.
        pos = 0
        n_claimed = 0
        while pos < self.num_items:
            batch = []
            # Grab up to batch_size consecutive unclaimed items. Scanning in order
            # keeps a batch size-coherent (the dataset is size-sorted), preserving
            # the low-padding batching that sort_by_size buys.
            while pos < self.num_items and len(batch) < self.batch_size:
                idx = pos
                pos += 1
                if self._claim(idx):
                    batch.append(idx)
            if batch:
                n_claimed += len(batch)
                yield batch
        _log(f"scheduler: this rank claimed {n_claimed}/{self.num_items} item(s)")
