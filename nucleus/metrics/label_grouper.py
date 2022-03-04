from typing import Any, List

import numpy as np
import pandas as pd


class LabelsGrouper:
    def __init__(self, annotations_or_predictions_list: List[Any]):
        self.items = annotations_or_predictions_list
        if len(self.items) > 0:
            assert hasattr(
                self.items[0], "label"
            ), f"Expected items to have attribute 'label' found none on {repr(self.items[0])}"
        self.codes, self.labels = pd.factorize(
            [item.label for item in self.items]
        )
        self.group_idx = 0

    def __iter__(self):
        self.group_idx = 0
        return self

    def __next__(self):
        if self.group_idx >= len(self.labels):
            raise StopIteration
        label = self.labels[self.group_idx]
        label_items = list(
            np.take(self.items, np.where(self.codes == self.group_idx)[0])
        )
        self.group_idx += 1
        return label, label_items

    def label_group(self, label: str) -> List[Any]:
        if len(self.items) == 0:
            return []
        idx = np.where(self.labels == label)[0]
        if idx >= 0:
            label_items = list(
                np.take(self.items, np.where(self.codes == idx)[0])
            )
            return label_items
        else:
            return []
