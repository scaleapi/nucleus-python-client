class RetryStrategy:
    statuses = {503, 504}
    sleep_times = [1, 3, 9]
