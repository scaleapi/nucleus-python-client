from nucleus.dataset import Dataset

PANDASET_ID = "ds_bwhjbyfb8mjj0ykagxf0"


def test_get_pandaset_items(CLIENT):
    dataset: Dataset = CLIENT.get_dataset(PANDASET_ID)
    items = dataset.items
    items_and_annotations = dataset.items_and_annotations()

    target_item = items[0]
    assert {_["item"].reference_id for _ in items_and_annotations} == set(
        [i.reference_id for i in items]
    )
    ref_item = dataset.refloc(target_item.reference_id)
    assert ref_item["item"] == target_item
    index_item = dataset.iloc(0)
    assert index_item["item"] in items
