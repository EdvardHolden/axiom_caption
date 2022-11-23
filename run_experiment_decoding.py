"""
Script for evaluating different decoding approaches of a model
"""
from tqdm import tqdm
import config

import evaluate
from parser import get_evaluate_parser

# TODO break this down the middle and run it on 2-3 machines?
"""
SAMPLERS = [
    {"sampler": "greedy"},
    {"sampler": "temperature", "sampler_temperature": 1.0},
    {"sampler": "temperature", "sampler_temperature": 0.9},
]
"""
"""
SAMPLERS = [
    {"sampler": "temperature", "sampler_temperature": 0.8},
    {"sampler": "temperature", "sampler_temperature": 0.5},
    {"sampler": "top_k", "sampler_top_k": 32},
]
"""
SAMPLERS = [
    {"sampler": "top_k", "sampler_top_k": 64},
    {"sampler": "top_k", "sampler_top_k": 128},
    {"sampler": "top_k", "sampler_top_k": 256},
]

# Set no_samples as a static thing across experiments to avoid any mistakes
no_samples = {"no_samples": [1, 2, 4]}
# --sampler
# --no_samples
# --sampler_temperature
# --sampler_top_k


# TODO could this be improved?
def launch_evaluation_job_decoding(args):
    """
    Run model over the problem data with the given sampler
    """

    # The initial training cmd
    cmd = f"{config.PYTHON} evaluate.py "

    # Add all other remaining training parameters
    for arg in args.__dict__:
        if arg != "verbose":
            cmd += f" --{arg} {args.__dict__[arg]} "

    print(cmd)

    # res = check_call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    res = evaluate.main(**vars(args))

    return res


def main():

    print("Warning: not using any of the sampler arguments for the argparser")

    parser = get_evaluate_parser()
    args = parser.parse_args()

    # For each sampler
    results = {}
    for sampler in tqdm(SAMPLERS):

        sampler.update(no_samples)

        # Get the tag for the job
        sampler_tag = "_".join(k + "_" + str(sampler[k]) for k in sorted(sampler))
        print(f"Running experiment for sampler: {sampler_tag}")

        # Change the parameters
        for k, v in sampler.items():
            args.__dict__[k] = v

        # Run the job on the directory and embedding
        res = launch_evaluation_job_decoding(args)
        results[sampler_tag] = res

    """
    for k, v in results.items():
        print(k, v)
    """
    for stat in ["coverage", "jaccard", "avg_size"]:
        print(stat, results[stat])

    print("# Finito")
    # """


if __name__ == "__main__":
    main()
