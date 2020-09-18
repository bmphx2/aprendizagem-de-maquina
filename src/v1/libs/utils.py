import time


def get_time():
    return time.time()


def get_time_diff(start_time):
    end_time = time.time()
    return round_float(end_time - start_time)


def round_float(value):
    return float("{:.3f}".format(value))


def get_num_rows(file):
    return sum(1 for line in open(file))


def generate_hash():
    return f"{get_time():.0f}"
