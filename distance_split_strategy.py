import random
import pandas as pd
from typing import Dict
from protest.base.split_strategy.abstract_split_strategy import (
    AbstractSplitStrategy,
    split_name,
)


def distance(sequence: str, reference: str):
    if len(sequence) != len(reference):
        raise ValueError(
            "Only calculate the distance between the same length of strings"
        )
    return sum(a != b for a, b in zip(sequence, reference))


def assign_distance(wide: pd.DataFrame, reference: str):
    return wide.assign(
        distance=lambda d: d["sequence"].map(lambda x: distance(x, reference))
    )


class DistanceSplitStrategy(AbstractSplitStrategy):
    """
    Split so that train contains up to N distance and valid contains the rest
    """

    def __init__(
        self,
        reference: str,
        distance_threshold: int = 1,
        **kwargs,
    ):
        self.reference = reference
        self.distance_threshold = distance_threshold
        super().__init__(**kwargs)

    def create_split_map(
        self, data: pd.DataFrame, target: str = None
    ) -> Dict[str, str]:
        if self.distance_threshold <= 0:
            raise ValueError("The distance threshold needs to be over 1")
        sequences = data["sequence"].drop_duplicates()
        sizes = self.n_train_valid_test(len(sequences))
        if all(list(len(s) == len(self.reference) for s in sequences)):
            data = assign_distance(data, self.reference)
        else:
            raise ValueError("Only the same length reference as sequences supported")
        sequences = list(data["sequence"].unique())
        counting = data["distance"]
        counts = data[["sequence", "distance"]]
        if max(counting) <= self.distance_threshold:
            raise ValueError(
                f"The distance threshold needs to be smaller than max distance "
                f"(max distance: {max(counting)})."
            )
        train = list(counts[counts["distance"] <= self.distance_threshold].sequence)
        rest = list(counts[counts["distance"] > self.distance_threshold].sequence)
        random.shuffle(train)
        random.shuffle(rest)
        train, valid, test = (
            train[: sizes.n_train],
            rest[: sizes.n_valid],
            rest[sizes.n_valid :],
        )
        split_map = {s: split_name(s, train, valid, test) for s in sequences}
        return split_map
