import os
import json


def create_job_dir(root_dir, job_name, params=None):

    # Create a new folder in parent_dir with unique_name "job_name"
    job_dir = os.path.join(root_dir, job_name)
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)

    # Write parameters in json file
    if params is not None:
        json_path = os.path.join(job_dir, "params.json")
        with open(json_path, "w") as f:
            json.dump(params, f)

    return job_dir


# Use as a decorator to debug functions
def debug(func):
    def _debug(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"{func.__name__}(args: {args}, kwargs: {kwargs}) -> {result}")
        return result

    return _debug
