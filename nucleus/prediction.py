from .annotation import BoxAnnotation

class BoxPrediction(BoxAnnotation):
    
    def __init__(self, reference_id: str, label: str, x: int, y: int, width: int, height: int, confidence: float, metadata: dict={}):
        super().__init__(reference_id, label, x, y, width, height, metadata)
        self.confidence = confidence

    def to_payload(self) -> dict:
        payload = super().to_payload()
        if self.confidence: payload["confidence"] = self.confidence

        return payload