"""Shared stateless utility function library for chipping images"""

import io
import json
import os
from itertools import product
from typing import Dict, List

import boto3
import numpy as np
from botocore.exceptions import ClientError
from PIL import Image

from .constants import (
    ANNOTATION_LOCATION_KEY,
    BOX_TYPE,
    GEOMETRY_KEY,
    HEIGHT_KEY,
    IMAGE_LOCATION_KEY,
    LABEL_KEY,
    TYPE_KEY,
    WIDTH_KEY,
    X_KEY,
    Y_KEY,
)


def split_s3_bucket_key(s3_path: str):
    s3_bucket, s3_key = s3_path.split("//", 1)[-1].split("/", 1)
    return s3_bucket, s3_key


def fetch_image(s3_url: str):
    s3_bucket, s3_key = split_s3_bucket_key(s3_url)
    image = Image.open(
        boto3.resource("s3").Bucket(s3_bucket).Object(s3_key).get()["Body"]
    )
    return image


def fetch_chip(ref_id: str):
    """
    Fetches the locations of the image and its corresponding annotations.

    This function checks if the reference ID starts with "s3" to determine if the
    image and annotations are stored on S3, otherwise it checks the local filesystem.
    If the image or annotations do not exist, it returns None for their locations.

    Args:
        ref_id (str): The reference ID for the image and annotations.

    Returns:
        A tuple containing the location of the image and the annotations.
        If either is not found, None is returned in their place.
    """
    image_loc = None
    annotation_loc = None
    if ref_id.startswith("s3"):
        s3_bucket, s3_key = split_s3_bucket_key(ref_id)
        try:
            boto3.resource("s3").Bucket(s3_bucket).Object(
                s3_key + ".jpeg"
            ).load()
            image_loc = ref_id + ".jpeg"
        except ClientError:
            return None, None
        try:
            boto3.resource("s3").Bucket(s3_bucket).Object(
                s3_key + ".json"
            ).load()
            annotation_loc = ref_id + ".json"
        except ClientError:
            return image_loc, None
    else:
        if os.path.exists(ref_id + ".jpeg"):
            image_loc = ref_id + ".jpeg"
            if os.path.exists(ref_id + ".json"):
                annotation_loc = ref_id + ".json"
    return image_loc, annotation_loc


def write_chip(
    ref_id: str, image: Image.Image, annotations: List[Dict[str, str]]
):
    if ref_id.startswith("s3"):
        s3_bucket, s3_key = split_s3_bucket_key(ref_id)
        byteio = io.BytesIO()
        image.save(byteio, format="jpeg")
        byteio.seek(0)
        boto3.resource("s3").Bucket(s3_bucket).Object(
            s3_key + ".jpeg"
        ).upload_fileobj(byteio)
        annotation_loc = None
        if len(annotations) > 0:
            boto3.resource("s3").Bucket(s3_bucket).Object(
                s3_key + ".json"
            ).put(
                Body=json.dumps(annotations, ensure_ascii=False).encode(
                    "UTF-8"
                ),
                ContentType="application/json",
            )
            annotation_loc = ref_id + ".json"
        return ref_id + ".jpeg", annotation_loc
    else:
        dirs = ref_id.rsplit("/", 1)[0]
        os.makedirs(dirs, exist_ok=True)
        image_loc = ref_id + ".jpeg"
        annotation_loc = None
        image.save(image_loc)
        if len(annotations) > 0:
            annotation_loc = ref_id + ".json"
            with open(annotation_loc, "w", encoding="utf-8") as f:
                json.dump(annotations, f, ensure_ascii=False)
        return image_loc, annotation_loc


def generate_offsets(w: int, h: int, chip_size: int, stride_size: int):
    xs = np.arange(0, w - stride_size, stride_size)
    ys = np.arange(0, h - stride_size, stride_size)
    if len(xs) > 1:
        xs = np.round(xs * (w - chip_size) / xs[-1]).astype(int)
    if len(ys) > 1:
        ys = np.round(ys * (h - chip_size) / ys[-1]).astype(int)
    yield from product(xs, ys)


def chip_annotations(data, x0: int, y0: int, x1: int, y1: int):
    """
    Adjusts the annotations to fit within the chip defined by the rectangle
    with top-left corner (x0, y0) and bottom-right corner (x1, y1).

    Parameters:
        data: List of annotation dictionaries to be adjusted.
        x0: The x-coordinate of the top-left corner of the chip.
        y0: The y-coordinate of the top-left corner of the chip.
        x1: The x-coordinate of the bottom-right corner of the chip.
        y1: The y-coordinate of the bottom-right corner of the chip.

    Returns:
        A list of adjusted annotation dictionaries that fit within the chip.
    """
    annotations = []
    X_1_KEY = "x1"
    Y_1_KEY = "y1"
    for annotation in data:
        geometry = annotation[GEOMETRY_KEY].copy()
        geometry[X_1_KEY] = geometry[X_KEY] + geometry[WIDTH_KEY]
        geometry[Y_1_KEY] = geometry[Y_KEY] + geometry[HEIGHT_KEY]
        geometry[X_KEY] = max(min(geometry[X_KEY], x1), x0) - x0
        geometry[X_1_KEY] = max(min(geometry[X_1_KEY], x1), x0) - x0
        geometry[Y_KEY] = max(min(geometry[Y_KEY], y1), y0) - y0
        geometry[Y_1_KEY] = max(min(geometry[Y_1_KEY], y1), y0) - y0
        geometry[WIDTH_KEY] = geometry[X_1_KEY] - geometry[X_KEY]
        geometry[HEIGHT_KEY] = geometry[Y_1_KEY] - geometry[Y_KEY]
        geometry["area"] = geometry[WIDTH_KEY] * geometry[HEIGHT_KEY]
        if geometry["area"] > 0:
            annotations.append(
                {
                    LABEL_KEY: annotation[LABEL_KEY],
                    TYPE_KEY: BOX_TYPE,
                    GEOMETRY_KEY: {
                        X_KEY: geometry[X_KEY],
                        Y_KEY: geometry[Y_KEY],
                        WIDTH_KEY: geometry[WIDTH_KEY],
                        HEIGHT_KEY: geometry[HEIGHT_KEY],
                    },
                }
            )
    return annotations


def process_chip(chip_arg):
    (
        offset,
        chip_size,
        w,
        h,
        item_ref_id,
        cache_directory,
        image,
        annotations,
    ) = chip_arg
    x0, y0 = map(int, offset)
    x1 = min(x0 + chip_size, w)
    y1 = min(y0 + chip_size, h)
    ref_id = f"{cache_directory}/{item_ref_id}_{x0}_{y0}_{x1}_{y1}"
    chipped_image_loc, chipped_annotation_loc = fetch_chip(ref_id)
    if chipped_image_loc:
        return {
            IMAGE_LOCATION_KEY: chipped_image_loc,
            ANNOTATION_LOCATION_KEY: chipped_annotation_loc,
        }
    chipped_image = image.crop((x0, y0, x1, y1))
    chipped_annotations = chip_annotations(annotations, x0, y0, x1, y1)
    chipped_image_loc, chipped_annotation_loc = write_chip(
        ref_id, chipped_image, chipped_annotations
    )
    return {
        IMAGE_LOCATION_KEY: chipped_image_loc,
        ANNOTATION_LOCATION_KEY: chipped_annotation_loc,
    }
