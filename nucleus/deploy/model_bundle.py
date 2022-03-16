from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dataclasses_json import Undefined, dataclass_json


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class ModelBundle:
    """
    Represents a ModelBundle.
    """

    bundle_name: str
    bundle_id: Optional[str] = None
    env_params: Optional[Dict[str, str]] = None
    location: Optional[str] = None
    metadata: Optional[Dict[Any, Any]] = None
    packaging_type: Optional[str]
    requirements: Optional[List[str]]

    def __str__(self):
        return f"ModelBundle(bundle_name={self.bundle_name})"
