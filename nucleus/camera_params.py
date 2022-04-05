from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

from .annotation import Point3D
from .constants import (
    CAMERA_MODEL_KEY,
    CX_KEY,
    CY_KEY,
    FX_KEY,
    FY_KEY,
    HEADING_KEY,
    K1_KEY,
    K2_KEY,
    K3_KEY,
    K4_KEY,
    P1_KEY,
    P2_KEY,
    POSITION_KEY,
)
from .quaternion import Quaternion

REQUIRED_CAMERA_PARAMS_KEYS = {
    POSITION_KEY,
    HEADING_KEY,
    FX_KEY,
    FY_KEY,
    CX_KEY,
    CY_KEY,
}


class CameraModels(str, Enum):
    BROWN_CONRADY = "brown_conrady"
    FISHEYE = "fisheye"

    def __contains__(self, item):
        try:
            self(item)
        except ValueError:
            return False
        return True


@dataclass
class CameraParams:
    """Camera position/heading used to record the image.

    Args:
        position (:class:`Point3D`): World-normalized position of the camera
        heading (:class:`Quaternion`): Vector4 indicating the quaternion of the
          camera direction; note that the z-axis of the camera frame
          represents the camera's optical axis.  See `Heading Examples
          <https://docs.scale.com/reference/data-types-and-the-frame-objects#heading-examples>`_.
        fx (float): Focal length in x direction (in pixels).
        fy (float): Focal length in y direction (in pixels).
        cx (float): Principal point x value.
        cy (float): Principal point y value.
    """

    position: Point3D
    heading: Quaternion
    fx: float
    fy: float
    cx: float
    cy: float
    camera_model: str
    k1: float
    k2: float
    k3: float
    k4: float
    p1: float
    p2: float

    def __post_init__(self):
        if self.camera_model is not None:
            if self.camera_model not in (k for k in CameraModels):
                raise ValueError(
                    f'Invalid Camera Model, the supported options are "{CameraModels.BROWN_CONRADY}" and "{CameraModels.FISHEYE}"'
                )

    @classmethod
    def from_json(cls, payload: Dict[str, Any]):
        """Instantiates camera params object from schematized JSON dict payload."""
        keys = set(payload.keys())
        if not keys.issuperset(REQUIRED_CAMERA_PARAMS_KEYS):
            raise ValueError(
                f"The following fields must be present in the camera_params dictionary: {REQUIRED_CAMERA_PARAMS_KEYS}"
            )

        return cls(
            Point3D.from_json(payload[POSITION_KEY]),
            Quaternion.from_json(payload[HEADING_KEY]),
            payload[FX_KEY],
            payload[FY_KEY],
            payload[CX_KEY],
            payload[CY_KEY],
            payload.get(CAMERA_MODEL_KEY, None),
            payload.get(K1_KEY, None),
            payload.get(K2_KEY, None),
            payload.get(K3_KEY, None),
            payload.get(K4_KEY, None),
            payload.get(P1_KEY, None),
            payload.get(P2_KEY, None),
        )

    def to_payload(self) -> dict:
        """Serializes camera params object to schematized JSON dict."""
        payload = {
            POSITION_KEY: self.position.to_payload(),
            HEADING_KEY: self.heading.to_payload(),
            FX_KEY: self.fx,
            FY_KEY: self.fy,
            CX_KEY: self.cx,
            CY_KEY: self.cy,
        }
        if self.k1:
            payload[K1_KEY] = self.k1
        if self.k2:
            payload[K2_KEY] = self.k2
        if self.k3:
            payload[K3_KEY] = self.k3
        if self.k4:
            payload[K4_KEY] = self.k4
        if self.p1:
            payload[P1_KEY] = self.p1
        if self.p2:
            payload[P2_KEY] = self.p2
        if self.camera_model:
            payload[CAMERA_MODEL_KEY] = self.camera_model
        return payload
