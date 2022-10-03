import json

from get_scores import get_problem_stats


# TODO extend
'''
EXPERIMENTS = {
    115407: "iProver",
    115377: "Upper Bound",
    115408: "Sine 1 1",
    115409: "Sine 3 0",
    115451: "Sequence",
    115452: "Sequence + Sine 1 1",
    115453: "Sequence + Sine 3 0",
}
'''

'''
# Mix of nndata/deepmath and mizar40 for comparisions
EXPERIMENTS = {115497: "deepmath_clean",
               115498: "deepmath_ideal",
               115500: "deepmath_sine_1_1 ", # Rerun
               115501: "deepmath_sine_3_0",
               115502: "deepmath_caption_greedy_no_samples_4_length_22",
               115505: "deepmath_caption_sine_3_0",
               115506: "deepmath_caption_sine_1_1",
               115507: 'mizar_40_clean',
               115508: 'mizar_40_sine_1_1',
               115509: "mizar_40_sine_3_0",
               115510: "mizar_40_caption_sine_3_0",
               115511: "mizar_40_caption",
               115512: "mizar_40_caption_sine_1_1"}


# For the 6K vocab on Mizar
EXPERIMENTS = {115507: 'mizar_40_clean',
               115508: 'mizar_40_sine_1_1',
               115509: "mizar_40_sine_3_0",
               115514: "mizar_40_caption",
               115516: "mizar_40_caption_sine_1_1",
               115520: "mizar_40_caption_sine_3_0"}
'''

EXPERIMENTS = {
    #115498: "Ideal",  # Upper bound
    115507: "Clean",
    #115555: "Sine 1 1",
    116861: "Sine 3 0",
    115591: "Caption",
    115634: "Caption + Sine 3 0",
    117090: "Caption Remap",
    117091: "Caption Remap + sine 3 0",
}




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
