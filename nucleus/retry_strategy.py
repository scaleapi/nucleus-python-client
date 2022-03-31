# TODO: use retry library instead of custom code. Tenacity is one option.
import random


class RetryStrategy:
    statuses = {503, 524, 520, 504}

    @staticmethod
    def sleep_times():
        sleep_times = [1, 3, 9, 27]  # These are in seconds

        return [2 * random.random() * t for t in sleep_times]
