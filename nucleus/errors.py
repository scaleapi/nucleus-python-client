class ModelCreationError(Exception):
    def __init__(self, message="Could not create the model"):
        self.message = message
        super().__init__(self.message)


class ModelRunCreationError(Exception):
    def __init__(self, message="Could not create the model run"):
        self.message = message
        super().__init__(self.message)


class NotFoundError(Exception):
    def __init__(
        self, message="Could not open file. Check the path or if it exists."
    ):
        self.message = message
        super().__init__(self.message)


class DatasetItemRetrievalError(Exception):
    def __init__(self, message="Could not retrieve dataset items"):
        self.message = message
        super().__init__(self.message)


class NucleusAPIError(Exception):
    def __init__(self, endpoint, command, response):
        message = f"Tried to {command.__name__} {endpoint}, but received {response.status_code}: {response.reason}."
        if hasattr(response, "text"):
            if response.text:
                message += f"\nThe detailed error is:\n{response.text}"
        super().__init__(message)
