"""Deterministic, task-homogeneous batches shared across DDP ranks."""
import random
from torch.utils.data import Dataset, Sampler

TASKS = ("sar", "nir", "season")


class JointTaskDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return sum(len(d) for d in self.datasets.values())

    def __getitem__(self, key):
        task, index = key
        return {**self.datasets[task][index], "task_id": task}


class JointTaskBatchSampler(Sampler):
    """Every three optimizer updates visit each task once, in shuffled order.

    All accumulation microbatches use the same task. Rank slices of each global
    batch are disjoint. Each task reshuffles its samples only when exhausted.
    Lightning must NOT wrap this sampler in a DistributedSampler.
    """
    def __init__(self, sizes, batch_size, accumulation, updates, rank=0,
                 world_size=1, seed=42):
        self.sizes = dict(sizes)
        self.batch_size, self.accumulation = int(batch_size), int(accumulation)
        self.updates, self.rank, self.world_size = int(updates), int(rank), int(world_size)
        self.seed, self.epoch = int(seed), 0
        if min(self.batch_size, self.accumulation, self.updates, self.world_size) < 1:
            raise ValueError("Batch size, accumulation, updates and world size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("Invalid DDP rank")
        if set(self.sizes) != set(TASKS):
            raise ValueError(f"Expected datasets {TASKS}")
        if min(self.sizes.values()) < self.batch_size * self.world_size:
            raise ValueError("Each dataset must contain at least one global microbatch")

    @property
    def sampler(self):
        # Lightning discovers batch_sampler.sampler and sets its epoch before
        # DataLoader workers prefetch the next epoch.
        return self

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.updates * self.accumulation

    def __iter__(self):
        task_rng = random.Random(self.seed + self.epoch * 1000003)
        rngs = {t: random.Random(self.seed + self.epoch * 1000003 + 101 * (i + 1))
                for i, t in enumerate(TASKS)}
        pools, positions = {}, {}
        for task in TASKS:
            pools[task] = list(range(self.sizes[task]))
            rngs[task].shuffle(pools[task])
            positions[task] = 0
        order = []
        width = self.batch_size * self.world_size
        for update in range(self.updates):
            if update % len(TASKS) == 0:
                order = list(TASKS)
                task_rng.shuffle(order)
            task = order[update % len(TASKS)]
            for _ in range(self.accumulation):
                if positions[task] + width > self.sizes[task]:
                    rngs[task].shuffle(pools[task])
                    positions[task] = 0
                start = positions[task] + self.rank * self.batch_size
                indices = pools[task][start:start + self.batch_size]
                positions[task] += width
                yield [(task, index) for index in indices]
