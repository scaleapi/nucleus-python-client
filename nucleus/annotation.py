class BoxAnnotation:

    def __init__(self, reference_id: str, label: str, x: int, y: int, width: int, height: int, metadata: dict={}):
        self.reference_id = reference_id
        self.label = label
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.metadata = metadata

    def to_payload(self) -> dict:
        return {
            "label": self.label,
            "type": "box",
            "geometry": {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height
            },
            "reference_id": self.reference_id,
            "metadata": self.metadata
        }