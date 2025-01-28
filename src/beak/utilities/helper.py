from datetime import datetime


def get_timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def get_runtime(function, *args, **kwargs):
    start_time = datetime.now()
    result = function(*args, **kwargs)
    end_time = datetime.now()

    duration = end_time - start_time
    print(f"\nRuntime: {duration.total_seconds() / 60:.1f} minutes.")

    return result


