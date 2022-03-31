import json

from get_scores import get_problem_stats


# TODO extend
EXPERIMENTS = {115407: "iProver", 115377: "Upper Bound", 115408: "Sine 1 1", 115409: "Sine 3 0"}


def main():

    for exp_id, exp_name in EXPERIMENTS.items():

        exp_res = {"preamble": {"program": exp_name}}

        # Extract results
        stats_res = {}
        res = get_problem_stats(exp_id)
        for problem, time in res:
            stats_res[problem] = {"status": True, "rtime": time}

        # Join the two
        exp_res["stats"] = stats_res

        with open(f"data/{exp_id}.json", "w") as outfile:
            json.dump(exp_res, outfile, indent=4)


if __name__ == "__main__":
    main()
