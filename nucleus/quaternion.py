from dataclasses import dataclass
from typing import Dict

from .constants import W_KEY, X_KEY, Y_KEY, Z_KEY


@dataclass
class Quaternion:
    """Quaternion objects are used to represent rotation.

    We use the Hamilton/right-handed quaternion convention, where
    ::

        i^2 = j^2 = k^2 = ijk = -1

    The quaternion represented by the tuple ``(x, y, z, w)`` is equal to
    ``w + x*i + y*j + z*k``.

    Parameters:
        x (float): The x value.
        y (float): The y value.
        x (float): The z value.
        w (float): The w value.
    """

    x: float
    y: float
    z: float
    w: float

    @classmethod
    def from_json(cls, payload: Dict[str, float]):
        """Instantiates quaternion object from schematized JSON dict payload."""
        return cls(
            payload[X_KEY], payload[Y_KEY], payload[Z_KEY], payload[W_KEY]
        )

    def to_payload(self) -> dict:
        """Serializes quaternion object to schematized JSON dict."""
        return {
            X_KEY: self.x,
            Y_KEY: self.y,
            Z_KEY: self.z,
            W_KEY: self.w,
        }
