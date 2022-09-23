import io
from typing import Any, Callable, Dict, List, Set

from PIL import Image, ImageDraw

# From scaleapi/server/src/lib/select/api/types.ts
_TOP_LEVEL_BOX_REQUIRED_KEYS = {"geometry", "type"}
_TOP_LEVEL_OPTIONAL_KEYS = {"label", "confidence", "classPdf", "metadata"}
# TODO idk which ones are right for nucleus
_TOP_LEVEL_BOX_ALL_KEYS = _TOP_LEVEL_BOX_REQUIRED_KEYS.union(
    _TOP_LEVEL_OPTIONAL_KEYS
)
_BOX_GEOMETRY_KEYS = {"x", "y", "width", "height"}
_TOP_LEVEL_CATEGORY_REQUIRED_KEYS = {"geometry", "type"}
_TOP_LEVEL_LINE_REQUIRED_KEYS = {"geometry", "type"}


def verify_output(
    annotation_list: List[Dict[str, Any]],
    required_keys: Set,
    all_keys: Set,
    expected_geometry_keys: Set,
    annotation_type: str,
):
    for annotation in annotation_list:
        keys = set(annotation.keys())
        missing_keys = required_keys.difference(keys)
        extra_keys = keys.difference(all_keys)
        if len(missing_keys):
            raise ValueError(f"Missing keys {missing_keys} in annotation")
        if len(extra_keys):
            raise ValueError(f"Extra keys {extra_keys} in annotation")
        # All annotations we care about have this geometry field, so we can do the check here
        actual_geometry_keys = set(annotation["geometry"].keys())
        if actual_geometry_keys != expected_geometry_keys:
            raise ValueError(
                f"Keys {actual_geometry_keys} in geometry not equal to expected {expected_geometry_keys}"
            )
        if annotation["type"] != annotation_type:
            raise ValueError(
                f"Bounding box type {annotation['type']} should equal {annotation_type}"
            )


def verify_box_output(bbox_list):
    required_keys = _TOP_LEVEL_BOX_REQUIRED_KEYS
    all_keys = _TOP_LEVEL_BOX_ALL_KEYS
    expected_geometry_keys = _BOX_GEOMETRY_KEYS
    annotation_type = "box"
    return verify_output(
        bbox_list,
        required_keys,
        all_keys,
        expected_geometry_keys,
        annotation_type,
    )


def verify_category_output(category_annotation):
    """"""
    pass


def visualize_box_launch_bundle(
    img_file: str,
    load_predict_fn: Callable,
    load_model_fn: Callable = None,
    model: Any = None,
    show_image: bool = False,
    max_boxes: int = 5,
) -> Image:
    """
    Run this function locally to visualize what your Launch bundle will do on a local image
    Intended to verify that your Launch bundle returns annotations in the correct format, as well as sanity check
    any coordinate systems used for the image.
    Will display the image in a separate window if show_image == True.
    Returns the image
    """
    # Basically do the same thing as what Launch does but locally

    if not (model is None) ^ (load_model_fn is None):
        raise ValueError(
            "Exactly one of `model` and `load_model_fn` must not be None."
        )

    if load_model_fn:
        model = load_model_fn()

    predict_fn = load_predict_fn(model)

    with open(img_file, "rb") as f:
        img_bytes = f.read()

    output = predict_fn(img_bytes)
    verify_box_output(output)

    image = Image.open(io.BytesIO(img_bytes))
    draw = ImageDraw.Draw(image)
    for bbox in output[:max_boxes]:
        geo = bbox["geometry"]
        x, y, w, h = geo["x"], geo["y"], geo["width"], geo["height"]
        draw.rectangle([(x, y), (x + w, y + h)])

    if show_image:
        image.show()

    return image
