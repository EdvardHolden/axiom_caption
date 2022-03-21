"""
Script for evaluating different decoding approaches of a model
"""
from tqdm import tqdm
import config

import evaluate
from evaluate import get_evaluate_parser

SAMPLERS = [
    {"sampler": "greedy", "no_samples": 1},
    {"sampler": "greedy", "no_samples": 3},
    {"sampler": "greedy", "no_samples": 5},
    {"sampler": "temperature", "sampler_temperature": 0.3, "no_samples": 1},
    {"sampler": "top_k", "sampler_top_k": 100, "no_samples": 1},
]
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

        # Get the tag for the job
        sampler_tag = "_".join(k + "_" + str(sampler[k]) for k in sorted(sampler))
        print(f"Running experiment for sampler: {sampler_tag}")

        # Change the parameters
        for k, v in sampler.items():
            args.__dict__[k] = v

        # Run the job on the directory and embedding
        res = launch_evaluation_job_decoding(args)
        results[sampler_tag] = res

    for k, v in results.items():
        print(k, v)

    print("# Finito")
    # """


if __name__ == "__main__":
    main()
