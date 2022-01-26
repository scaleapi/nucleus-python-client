class ModelBundle:
    """
    Represents a ModelBundle.
    TODO fill this out with more than just a name potentially.
    """

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"ModelBundle(name={self.name})"
