"""Stage-adaptive candidate routing for GEARS."""

import math
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class RoutingResult:
    elite_indices: List[int]
    recycle_indices: List[int]
    discard_indices: List[int]


def get_stage_focus(checkpoint_idx: int, transition_alpha: int) -> str:
    """Return the reward dimension used for recycling at a checkpoint."""
    return "MQ" if checkpoint_idx <= transition_alpha else "VQ"


def route_candidates(
    reward_dicts: List[dict],
    checkpoint_idx: int,
    transition_alpha: int,
    keep_size: int,
    editing_ratio: float = 0.5,
) -> RoutingResult:
    """Split candidates into discarded, recycled, and elite groups."""
    if not 0.0 <= editing_ratio <= 1.0:
        raise ValueError("editing_ratio must be in [0, 1].")

    num_candidates = len(reward_dicts)
    keep_size = min(max(keep_size, 0), num_candidates)

    ranked_by_alignment = sorted(
        ((idx, reward["TA"]) for idx, reward in enumerate(reward_dicts)),
        key=lambda item: item[1],
    )
    num_discarded = num_candidates - keep_size
    discard_indices = [idx for idx, _ in ranked_by_alignment[:num_discarded]]
    valid_indices = [idx for idx, _ in ranked_by_alignment[num_discarded:]]

    stage_focus = get_stage_focus(checkpoint_idx, transition_alpha)
    ranked_by_focus = sorted(
        ((idx, reward_dicts[idx][stage_focus]) for idx in valid_indices),
        key=lambda item: item[1],
    )
    num_recycled = min(
        len(ranked_by_focus),
        math.ceil(len(ranked_by_focus) * editing_ratio),
    )
    recycle_indices = [idx for idx, _ in ranked_by_focus[:num_recycled]]
    elite_indices = [idx for idx, _ in ranked_by_focus[num_recycled:]]

    return RoutingResult(
        elite_indices=elite_indices,
        recycle_indices=recycle_indices,
        discard_indices=discard_indices,
    )
