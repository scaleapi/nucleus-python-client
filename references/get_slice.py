# Example usage of batching helper_functions
import json
import argparse
from utils import (
    batch_upload_append,
    batch_upload_annotation,
    batch_upload_prediction,
)

APPEND_ACTION = "append"
ANNOTATE_ACTION = "annotate"
PREDICT_ACTION = "predict"

commands_dict = {
    APPEND_ACTION: batch_upload_append,
    ANNOTATE_ACTION: batch_upload_annotation,
    PREDICT_ACTION: batch_upload_prediction,
}


def main(args):