class PolygonAnnotationTypeError(Exception):
    def __init__(
        self,
        message="Annotation was expected to be of type 'BoxAnnotation' or 'PolygonAnnotation'.",
    ):
        self.message = message
        super().__init__(self.message)


class EverythingFilteredError(Exception):
    pass
