"""Curriculum schedules expressed as fractions of the total optimizer-update budget."""
import numpy as np

# Each schedule: list of (fraction_of_updates, pool_name). Fractions sum to 1.
SCHEDULES = {
    "E1": [(1.0, "all")],
    "E2": [(0.3, "random_em"), (0.7, "all")],
    "E3": [(0.3, "easy+medium"), (0.7, "all")],
    "E4": [(0.3, "medium+hard"), (0.7, "all")],
    "E5": [(0.2, "easy"), (0.2, "easy+medium"), (0.6, "all")],
}


def build_pools(difficulty, subset_seed=1234):
    """Map pool name -> array of dataset indices.

    difficulty: sequence of 'easy'/'medium'/'hard', one per training-dataset index.
    'random_em' is a fixed random subset (same for every training seed) with |easy+medium| images.
    """
    difficulty = np.asarray(difficulty)
    idx = np.arange(len(difficulty))
    pools = {
        "all": idx,
        "easy": idx[difficulty == "easy"],
        "medium": idx[difficulty == "medium"],
        "hard": idx[difficulty == "hard"],
    }
    pools["easy+medium"] = np.concatenate([pools["easy"], pools["medium"]])
    pools["medium+hard"] = np.concatenate([pools["medium"], pools["hard"]])
    rng = np.random.RandomState(subset_seed)
    pools["random_em"] = np.sort(rng.choice(idx, size=len(pools["easy+medium"]), replace=False))
    return pools


def stage_boundaries(schedule, total_updates):
    """Return update index at which each stage starts (rounded), plus total."""
    cum = np.cumsum([0.0] + [f for f, _ in schedule])
    assert abs(cum[-1] - 1.0) < 1e-9, "schedule fractions must sum to 1"
    return [int(round(c * total_updates)) for c in cum]
