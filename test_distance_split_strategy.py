import pytest
import protest.base.split_strategy.distance_split_strategy as pp
from protest.base.protein_engineering_dataset import ProteinEngineeringDataset


@pytest.mark.parametrize("sequence,reference,expected", [("ABBB", "AAAA", 3)])
def test_distance(sequence, reference, expected):
    result = pp.distance(sequence, reference)
    assert result == expected


class TestDistanceSplitStrategy:
    @pytest.fixture()
    def reference(self, charge_ladder):
        reference = charge_ladder["sequence"].iloc[0]
        return reference

    @pytest.fixture()
    def data_with_split(self, charge_ladder):
        reference = charge_ladder["sequence"].iloc[0]
        split_strategy = pp.DistanceSplitStrategy(reference)
        data = ProteinEngineeringDataset.from_data_frame(
            charge_ladder, ["charge"], split_strategy
        )
        return data

    def test_train_valid_no_overlap(self, data_with_split):
        train_seq = data_with_split.data_frame.query(
            "split == 'train'"
        ).sequence.unique()
        valid_seq = data_with_split.data_frame.query(
            "split == 'valid'"
        ).sequence.unique()
        assert len(set(train_seq).intersection(valid_seq)) == 0

    def test_valid_not_implemented(self, data_with_split, reference):
        mutation_counts = data_with_split.data_frame["sequence"].map(
            lambda x: pp.distance(x, reference)
        )
        strategy = pp.DistanceSplitStrategy(
            reference, distance_threshold=max(mutation_counts)
        )
        with pytest.raises(ValueError):
            strategy.split(data_with_split.data_frame)

    def test_train_not_implemented(self, data_with_split, reference):
        strategy = pp.DistanceSplitStrategy(reference, distance_threshold=0)
        with pytest.raises(ValueError):
            strategy.split(data_with_split.data_frame)

    def test_valid_on_more_mutations(self, data_with_split, reference):
        train_max = max(
            data_with_split.data_frame.query("split == 'train'")["sequence"].map(
                lambda x: pp.distance(x, reference)
            )
        )
        valid_min = min(
            data_with_split.data_frame.query("split == 'valid'")["sequence"].map(
                lambda x: pp.distance(x, reference)
            )
        )
        assert train_max < valid_min

    def test_reference_sequences_same_length(self, data_with_split, reference):
        strategy = pp.DistanceSplitStrategy(reference[:-1])
        with pytest.raises(ValueError):
            strategy.split(data_with_split.data_frame)
