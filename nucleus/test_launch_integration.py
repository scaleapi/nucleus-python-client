import io
from typing import Any, Callable, Dict

from PIL import Image, ImageDraw

# From scaleapi/server/src/lib/select/api/types.ts
_TOP_LEVEL_REQUIRED_KEYS = {"geometry", "type"}
_TOP_LEVEL_OPTIONAL_KEYS = {"label", "confidence", "classPdf", "metadata"}
# TODO idk which ones are right for nucleus
_TOP_LEVEL_ALL_KEYS = _TOP_LEVEL_REQUIRED_KEYS.union(_TOP_LEVEL_OPTIONAL_KEYS)
_GEOMETRY_KEYS = {"x", "y", "width", "height"}


def verify_box_output(bbox_list):
    for bbox in bbox_list:
        keys = set(bbox.keys())
        missing_keys = _TOP_LEVEL_REQUIRED_KEYS.difference(keys)
        extra_keys = keys.difference(_TOP_LEVEL_ALL_KEYS)
        if len(missing_keys):
            raise ValueError(f"Missing keys {missing_keys} in box annotation")
        if len(extra_keys):
            raise ValueError(f"Extra keys {extra_keys} in box annotation")
        geometry_keys = set(bbox["geometry"].keys())
        if geometry_keys != _GEOMETRY_KEYS:
            raise ValueError(
                f"Keys {geometry_keys} in geometry not equal to expected {_GEOMETRY_KEYS}"
            )
        if bbox["type"] != "box":
            raise ValueError(
                f"Bounding box type {bbox['type']} should equal 'box'"
            )


def visualize_bbox_launch_bundle(
    img_file: str,
    config: Dict[str, Any],
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
        model = load_model_fn(config)

    predict_fn = load_predict_fn(config, model)

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
