"""
Contains client-side validation functions
"""


def validate_task_request(url, args):
    # A task request must have at least one of url or args, otherwise there's no input!
    if url is None and args is None:
        raise ValueError("Must specify at least one of url or args")
