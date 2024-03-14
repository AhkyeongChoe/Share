import pandas as pd
import numpy as np
import math
import random
from typing import List, Dict
from biotite.sequence import ProteinSequence
from protest.base.split_strategy.abstract_split_strategy import (
    AbstractSplitStrategy,
    split_name,
)


def get_pdb_reference(wide: pd.DataFrame) -> str:
    pdb_reference = (
        wide[["residue_number", "residue_name"]]
        .drop_duplicates()
        .assign(
            residue_name=wide["residue_name"].map(ProteinSequence.convert_letter_3to1)
        )
    )
    return "".join(pdb_reference.residue_name)


def euclidean_distance(n: pd.DataFrame, m: pd.DataFrame) -> float:
    return math.sqrt(
        math.pow(n.x_coord - m.x_coord, 2)
        + math.pow(n.y_coord - m.y_coord, 2)
        + math.pow(n.z_coord - m.z_coord, 2)
    )


class DistanceToActiveSiteSplitStrategy(AbstractSplitStrategy):
    """
    Split strategy based on active site. Train contains sequence
    with mutants near active site, and valid contains the rest
    """

    def __init__(
        self,
        pdb: pd.DataFrame,
        active_sites: List[int],
        euclidean_distance_threshold: float = 7.3,  # need to be adjusted
        **kwargs,
    ):
        if not isinstance(active_sites, list) or not all(
            isinstance(x, int) for x in active_sites
        ):
            raise TypeError("Active sites should be a list of integers")
        if (
            not isinstance(euclidean_distance_threshold, float)
            or euclidean_distance_threshold < 0
        ):
            raise ValueError(
                "Euclidean distance threshold should be a non-negative float"
            )
        self.pdb = pdb
        self.active_sites = active_sites
        self.euclidean_distance_threshold = euclidean_distance_threshold
        super().__init__(**kwargs)

    def create_split_map(
        self, data: pd.DataFrame, target: str = None
    ) -> Dict[str, str]:
        sequences = data["sequence"].drop_duplicates()
        sizes = self.n_train_valid_test(len(sequences))
        reference = get_pdb_reference(self.pdb)
        if not all(len(s) == len(reference) for s in sequences):
            raise ValueError("Only the same length reference as sequences supported")
        if max(self.active_sites) > len(reference):
            raise ValueError("Active sites need to be within the reference length")
        pdb_coordinates = (
            self.pdb[
                [
                    "residue_number",
                    "atom_name",
                    "alt_loc",
                    "x_coord",
                    "y_coord",
                    "z_coord",
                ]
            ]
            .loc[lambda d: d["atom_name"] == "CA"]
            .loc[lambda d: (d["alt_loc"] == "A") | (d["alt_loc"] == "")]
        )
        pdb_coordinates = pdb_coordinates.assign(
            dist_to_active_site=[
                min(
                    euclidean_distance(
                        pdb_coordinate, pdb_coordinates.iloc[active_site]
                    )
                    for active_site in self.active_sites
                )
                for pdb_coordinate in pdb_coordinates.itertuples()
            ]
        )
        mutated_positions = np.array(
            list(data.sequence.apply(lambda x: [a != b for a, b in zip(x, reference)]))
        )
        index = pdb_coordinates.dist_to_active_site < self.euclidean_distance_threshold
        train = list(data.sequence[index] & mutated_positions.sum(axis=1) >= 1)
        rest = list(set(data.sequence) - set(train))
        random.shuffle(train)
        random.shuffle(rest)
        train, valid, test = (
            train[: sizes.n_train],
            rest[: sizes.n_valid],
            rest[sizes.n_valid :],
        )
        split_map = {s: split_name(s, train, valid, test) for s in sequences}
        return split_map
