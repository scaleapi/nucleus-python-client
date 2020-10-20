# References

### utils.py
Helper functions for batching uploads.  Invoke these functions through the run_batching.py script, or
use them in your own code.

Use the use_notebook argument to toggle between command line and ipython notebook versions of the tqdm progress bar.

### run_batching.py
Supports uploading dataset items or annotations in batch, using command line arguments.

Arguments:
--action ACTION: Which dataset action to perform.  Current options are: ("append", "annotate")
--api_key API_KEY: User's Scale API_KEY for access control
--dataset_id DATASET_ID: identifier of existing dataset to which to upload payload.
--payload_json_file PAYLOAD_JSON_FILE: filepath to json file containing payload to upload
[--batch_size BATCH_SIZE]

Example usage
```python
python3 references/run_batching.py --api_key 'YOUR_API_KEY' --payload_json_file ../payload_append.json --action append --dataset_id ds_by6e7z7pf3w0040d
```
