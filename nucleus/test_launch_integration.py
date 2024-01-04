import io
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type

from PIL import Image, ImageDraw

if TYPE_CHECKING:
    # Backwards compatibility is even uglier with mypy
    from pydantic.v1 import BaseModel, Extra, ValidationError
else:
    try:
        # NOTE: we always use pydantic v1 but have to do these shenanigans to support both v1 and v2
        from pydantic.v1 import BaseModel, Extra, ValidationError
    except ImportError:
        from pydantic import BaseModel, Extra, ValidationError

# From scaleapi/server/src/lib/select/api/types.ts
# These classes specify how user models must pass output to Launch + Nucleus.


class PointModel(BaseModel):  # pylint: disable=used-before-assignment
    x: float
    y: float

    class Config:
        extra = Extra.forbid  # pylint: disable=used-before-assignment


class BoxGeometryModel(BaseModel):
    x: float
    y: float
    width: float
    height: float

    class Config:
        extra = Extra.forbid


class BoxAnnotationModel(BaseModel):
    geometry: BoxGeometryModel
    type: str
    label: Optional[str] = None
    confidence: Optional[float] = None
    classPdf: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        extra = Extra.forbid


class NoneGeometryModel(BaseModel):
    class Config:
        extra = Extra.forbid


class CategoryAnnotationModel(BaseModel):
    geometry: NoneGeometryModel
    type: str
    label: Optional[str] = None
    confidence: Optional[float] = None
    classPdf: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        extra = Extra.forbid


class LineGeometryModel(BaseModel):
    vertices: List[PointModel]

    class Config:
        extra = Extra.forbid


class LineAnnotationModel(BaseModel):
    geometry: LineGeometryModel
    type: str
    label: Optional[str] = None
    confidence: Optional[float] = None
    classPdf: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        extra = Extra.forbid


class PolygonGeometryModel(BaseModel):
    vertices: List[PointModel]

    class Config:
        extra = Extra.forbid


class PolygonAnnotationModel(BaseModel):
    geometry: PolygonGeometryModel
    type: str
    label: Optional[str] = None
    confidence: Optional[float] = None
    classPdf: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        extra = Extra.forbid


def verify_output(
    annotation_list: List[Dict[str, Any]],
    model: Type[BaseModel],
    annotation_type: str,
):
    for annotation in annotation_list:
        try:
            model.parse_obj(annotation)
        except ValidationError as e:  # pylint: disable=used-before-assignment
            raise ValueError("Failed validation") from e
        if annotation["type"] != annotation_type:
            raise ValueError(
                f"Bounding box type {annotation['type']} should equal {annotation_type}"
            )


def verify_box_output(bbox_list):
    annotation_type = "box"
    return verify_output(
        bbox_list,
        BoxAnnotationModel,
        annotation_type,
    )


def verify_category_output(category_list):
    """I think the annotation needs to be a list with a single element in the Launch+Nucleus sfn."""
    annotation_type = "category"
    return verify_output(
        category_list, CategoryAnnotationModel, annotation_type
    )


def verify_line_output(line_list):
    annotation_type = "line"
    return verify_output(
        line_list,
        LineAnnotationModel,
        annotation_type,
    )


def verify_polygon_output(polygon_list):
    annotation_type = "polygon"
    return verify_output(
        polygon_list,
        PolygonAnnotationModel,
        annotation_type,
    )


def _run_model(
    input_bytes: bytes,
    load_predict_fn: Callable,
    load_model_fn: Optional[Callable],
    model: Optional[Any],
):
    if not (model is None) ^ (load_model_fn is None):
        raise ValueError(
            "Exactly one of `model` and `load_model_fn` must not be None."
        )

    if load_model_fn:
        model = load_model_fn()

    predict_fn = load_predict_fn(model)
    return predict_fn(input_bytes)


_FILL_COLOR = (0, 255, 0, 50)
_OUTLINE_COLOR = (0, 255, 0, 255)


def visualize_box_launch_bundle(
    img_file: str,
    load_predict_fn: Callable,
    load_model_fn: Optional[Callable] = None,
    model: Optional[Any] = None,
    show_image: bool = False,
    max_annotations: int = 5,
) -> Image:
    """
    Run this function locally to visualize what your Launch bundle will do on a local image
    Intended to verify that your Launch bundle returns annotations in the correct format, as well as sanity check
    any coordinate systems used for the image.
    Will display the image in a separate window if show_image == True.
    Returns the image as well.

    Parameters:
        img_file: The path to a local image file.
        load_predict_fn: The load_predict_fn as part of your Launch bundle
        load_model_fn: The load_model_fn as part of your Launch bundle
        model: The model as part of your Launch bundle. Note: exactly one of load_model_fn and model must be specified
        show_image: Whether to automatically pop up the image + predictions in a separate window. Can be useful in a
          script.
        max_annotations: How many annotations you want to draw

    Returns:
        Image: The image with annotations drawn on top.
    """
    # Basically do the same thing as what Launch does but locally

    with open(img_file, "rb") as f:
        img_bytes = f.read()

    output = _run_model(img_bytes, load_predict_fn, load_model_fn, model)
    verify_box_output(output)

    image = Image.open(io.BytesIO(img_bytes))
    draw = ImageDraw.Draw(image, "RGBA")
    for bbox in output[:max_annotations]:
        geo = bbox["geometry"]
        x, y, w, h = geo["x"], geo["y"], geo["width"], geo["height"]
        draw.rectangle(
            [(x, y), (x + w, y + h)], outline=_OUTLINE_COLOR, fill=_FILL_COLOR
        )

    if show_image:
        image.show()

    return image


def run_category_launch_bundle(
    img_file: str,
    load_predict_fn: Callable,
    load_model_fn: Optional[Callable] = None,
    model: Optional[Any] = None,
):
    """
    Run this function locally to test if your image categorization model returns a format consumable by Launch + Nucleus
    Parameters:
        img_file: The path to a local image file.
        load_predict_fn: The load_predict_fn as part of your Launch bundle
        load_model_fn: The load_model_fn as part of your Launch bundle
        model: The model as part of your Launch bundle. Note: exactly one of load_model_fn and model must be specified
    Returns:
        The raw output (as a json) of your categorization model.
    """
    with open(img_file, "rb") as f:
        img_bytes = f.read()

    output = _run_model(img_bytes, load_predict_fn, load_model_fn, model)
    verify_category_output(output)
    return output


def visualize_line_launch_bundle(
    img_file: str,
    load_predict_fn: Callable,
    load_model_fn: Optional[Callable] = None,
    model: Optional[Any] = None,
    show_image: bool = False,
    max_annotations: int = 5,
) -> Image:
    """
    Run this function locally to visualize what your Launch bundle will do on a local image
    Intended to verify that your Launch bundle returns annotations in the correct format, as well as sanity check
    any coordinate systems used for the image.
    Will display the image in a separate window if show_image == True.
    Returns the image as well.

    Parameters:
        img_file: The path to a local image file.
        load_predict_fn: The load_predict_fn as part of your Launch bundle
        load_model_fn: The load_model_fn as part of your Launch bundle
        model: The model as part of your Launch bundle. Note: exactly one of load_model_fn and model must be specified
        show_image: Whether to automatically pop up the image + predictions in a separate window. Can be useful in a
          script.
        max_annotations: How many annotations you want to draw

    Returns:
        Image: The image with annotations drawn on top.
    """
    # Basically do the same thing as what Launch does but locally

    with open(img_file, "rb") as f:
        img_bytes = f.read()

    output = _run_model(img_bytes, load_predict_fn, load_model_fn, model)
    verify_line_output(output)

    image = Image.open(io.BytesIO(img_bytes))
    draw = ImageDraw.Draw(image, "RGBA")
    for bbox in output[:max_annotations]:
        geo = bbox["geometry"]
        vertices = [(v["x"], v["y"]) for v in geo["vertices"]]
        draw.line(vertices, fill=_OUTLINE_COLOR)

    if show_image:
        image.show()

    return image


def visualize_polygon_launch_bundle(
    img_file: str,
    load_predict_fn: Callable,
    load_model_fn: Optional[Callable] = None,
    model: Optional[Any] = None,
    show_image: bool = False,
    max_annotations: int = 5,
) -> Image:
    """
    Run this function locally to visualize what your Launch bundle will do on a local image
    Intended to verify that your Launch bundle returns annotations in the correct format, as well as sanity check
    any coordinate systems used for the image.
    Will display the image in a separate window if show_image == True.
    Returns the image as well.

    Parameters:
        img_file: The path to a local image file.
        load_predict_fn: The load_predict_fn as part of your Launch bundle
        load_model_fn: The load_model_fn as part of your Launch bundle
        model: The model as part of your Launch bundle. Note: exactly one of load_model_fn and model must be specified
        show_image: Whether to automatically pop up the image + predictions in a separate window. Can be useful in a
          script.
        max_annotations: How many annotations you want to draw

    Returns:
        Image: The image with annotations drawn on top.
    """
    # Basically do the same thing as what Launch does but locally

    with open(img_file, "rb") as f:
        img_bytes = f.read()

    output = _run_model(img_bytes, load_predict_fn, load_model_fn, model)
    verify_polygon_output(output)

    image = Image.open(io.BytesIO(img_bytes))
    draw = ImageDraw.Draw(image, "RGBA")
    for bbox in output[:max_annotations]:
        geo = bbox["geometry"]
        vertices = [(v["x"], v["y"]) for v in geo["vertices"]]
        draw.polygon(vertices, outline=_OUTLINE_COLOR, fill=_FILL_COLOR)

    if show_image:
        image.show()

    return image
