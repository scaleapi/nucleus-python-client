class ModelCreationError(Exception):
    def __init__(self, message="Could not create the model"):
        self.message = message
        super().__init__(self.message)


class ModelRunCreationError(Exception):
    def __init__(self, message="Could not create the model run"):
        self.message = message
        super().__init__(self.message)