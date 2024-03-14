import pytest
import protest.base.split_strategy.distance_to_active_site_split_strategy as pp
import pandas as pd
from typing import Tuple, List


@pytest.mark.parametrize(
    "data,expected",
    [
        (
            pd.DataFrame(
                {
                    "residue_number": [1, 1, 2, 2, 3],
                    "residue_name": ["ALA", "ALA", "TRP", "TRP", "TRP"],
                }
            ),
            "AWW",
        )
    ],
)
def test_get_pdb_reference(data, expected):
    result = pp.get_pdb_reference(data)
    assert result == expected


@pytest.mark.parametrize(
    "n,m,expected",
    [
        (
            pd.DataFrame({"x_coord": [0], "y_coord": [0], "z_coord": [0]}),
            pd.DataFrame({"x_coord": [2], "y_coord": [2], "z_coord": [1]}),
            3,
        )
    ],
)
def test_euclidean_distance(n, m, expected):
    result = pp.euclidean_distance(n, m)
    assert result == expected


class TestActiveSplitStrategy:
    @pytest.fixture
    def pdb_and_data_and_active_sites(self) -> Tuple[pd.DataFrame, pd.DataFrame, List]:
        pdb = pd.DataFrame(
            {
                "residue_number": [1, 2, 3, 4],
                "residue_name": ["ALA", "ALA", "ALA", "ALA"],
                "x_coord": [0, 3, 6, 8],
                "y_coord": [0, 9, 4, 3],
                "z_coord": [0, 2, 2, 6],
                "atom_name": ["CA", "CA", "CA", "CA"],
                "alt_loc": ["", "", "", ""],
            }
        )
        data = pd.DataFrame({"sequence": ["AAAA", "AAAB", "AAAA", "AAAC"]})
        active_sites = [1, 3]
        return pdb, data, active_sites

    def test_active_threshold_positive(self, pdb_and_data_and_active_sites):
        pdb, data, active_sites = pdb_and_data_and_active_sites
        with pytest.raises(ValueError):
            pp.DistanceToActiveSiteSplitStrategy(
                pdb, active_sites, euclidean_distance_threshold=-2
            )

    def test_reference_sequences_same_length(self, pdb_and_data_and_active_sites):
        pdb, _, active_sites = pdb_and_data_and_active_sites
        data = pd.DataFrame({"sequence": ["AAAA", "AAAB", "AAAA", "AAA"]})
        strategy = pp.DistanceToActiveSiteSplitStrategy(pdb, active_sites)
        with pytest.raises(ValueError):
            strategy.split(data, "output")

    def test_active_site_within_reference(self, pdb_and_data_and_active_sites):
        pdb, data, _ = pdb_and_data_and_active_sites
        active_sites = [1, 10]
        strategy = pp.DistanceToActiveSiteSplitStrategy(pdb, active_sites)
        with pytest.raises(ValueError):
            strategy.split(data, "output")
