from typing import Any, Dict, List
from dataclasses import dataclass
from dataclasses_json import dataclass_json, Undefined


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class ModelBundle:
    """
    Represents a ModelBundle.
    """

    bundle_id: str
    bundle_name: str
    env_params: Dict[str, str]
    location: str
    metadata: Dict[Any, Any]
    packaging_type: str
    requirements: List[str]


    def __str__(self):
        return f"ModelBundle(bundle_name={self.bundle_name})"
