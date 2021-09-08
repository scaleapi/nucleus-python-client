"""
Cleans up messy test/demo artifacts. 
- Delete empty Slices with <1k Items
- Delete Slices with name like "test" (often repeated demo flows)
"""
import argparse
import logging

import nucleus

log = logging.getLogger()
log.setLevel(logging.INFO)


def cleanup_slices(api_key, dataset_id):
    client = nucleus.NucleusClient(api_key)

    dataset = client.get_dataset(dataset_id)

    log.info("======\nChecking for Slices to delete\n======")
    for slc_id in dataset.slices:
        slc = client.get_slice(slc_id)
        slc_info = slc.info()
        slc_name = slc_info["name"]

        if any(kw in slc_name.lower() for kw in ["test", "night", "police", "truck"]):
            log.info("Deleting Slice '%s' because it contains night/police/truck keyword. Slice id: %s",
                     slc_name, slc_id)
            client.delete_slice(slc_id)

        if len(slc_info['dataset_items']) < 1000:
            log.info("Deleting Slice '%s' because it contains fewer than 1000 items. Slice id: %s",
                     slc_name, slc_id)
            client.delete_slice(slc_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('api_key', type=str)
    parser.add_argument('dataset_id', type=str)
    args = parser.parse_args()

    cleanup_slices(args.api_key, args.dataset_id)
