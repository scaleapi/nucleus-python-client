"""
Contains client-side validation functions
"""

import logging

logger = logging.getLogger(__name__)
logging.basicConfig()


def validate_task_request(url, args):
    # A task request must have at least one of url or args, otherwise there's no input!
    if url is None and args is None:
        raise ValueError("Must specify at least one of url or args")
    if url is not None and args is not None:
        logger.warning(
            "Passing both url and args to task request; args will be ignored"
        )
