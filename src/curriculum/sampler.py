"""Update-indexed curriculum sampler.

Sample position p belongs to optimizer update u = p // batch_size. The active pool is chosen from u, so the
schedule is defined in optimizer updates regardless of pool size. Within a stage, indices are drawn by cycling
through fresh random permutations of the pool (i.e. ordinary shuffled epochs over the pool; with replacement
across cycles). For schedule E1 this is identical to standard random shuffling.
"""
import numpy as np
import torch

from .schedules import stage_boundaries


class CurriculumSampler(torch.utils.data.Sampler):
    def __init__(self, pools, schedule, batch_size, updates_per_epoch, total_updates, seed):
        self.pools, self.schedule = pools, schedule
        self.bs, self.upe, self.total = batch_size, updates_per_epoch, total_updates
        self.bounds = stage_boundaries(schedule, total_updates)
        self.seed = seed
        self.pos = 0          # global sample position (persists across epochs)
        self.log = []         # (update, stage, dataset_index)
        self._stage, self._queue = None, []

    def __len__(self):
        return self.upe * self.bs

    def set_epoch(self, epoch):
        """Jump to the start of `epoch` (used when resuming). Regenerates the deterministic stream."""
        target = epoch * len(self)
        self.pos, self._stage, self._queue, self.log = 0, None, [], []
        while self.pos < target:
            self._next()

    def stage_of(self, update):
        for s in range(len(self.schedule)):
            if update < self.bounds[s + 1]:
                return s
        return len(self.schedule) - 1

    def _next(self):
        u = self.pos // self.bs
        s = self.stage_of(u)
        if s != self._stage:  # new stage: new deterministic RNG stream
            self._stage, self._queue = s, []
            self._rng = np.random.RandomState(self.seed * 1000 + s)
        if not self._queue:
            self._queue = list(self._rng.permutation(self.pools[self.schedule[s][1]]))
        i = int(self._queue.pop())
        self.log.append((u, s, i))
        self.pos += 1
        return i

    def __iter__(self):
        for _ in range(len(self)):
            yield self._next()
