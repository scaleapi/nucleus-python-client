# TODO: use retry library instead of custom code. Tenacity is one option.
class RetryStrategy:
    statuses = {503, 524, 520, 504}
    sleep_times = [1, 3, 9]
