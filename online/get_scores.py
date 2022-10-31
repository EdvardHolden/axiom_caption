# Compute metrics of experiments in the dictionary in EXPERIMENTS.
# These scores are segmented based on the dataset provided
# in DATA_SETS

import mysql.connector as db
import db_cred
import traceback
import os

LTB_PROBLEMS = True
INCLUDE_INCORRECT = False

# Current library version of deepmath in the DB
LIBRARY_VERSION = 64

DATASET_DIR = "../data/deepmath/"

""" Error in problems
EXPERIMENTS = {115360: "test experiment",
               115362: "first test on clean",
               115363: "second test on clean",
               115364: "Ideal",
               115365: "Sine 1 1",
               115366: "Sine 3 0"}
"""
# """
# Good experiments with proper quoting of numbers
# EXPERIMENTS = {115376: "Clean", 115377: "Ideal", 115378: "Sine 1 1", 115379: "Sine 3 0"}
# """
# Experiments with 1000 extra axioms
# EXPERIMENTS = {115386: "Clean", 115387: "Sine 1 1", 115388: "Sine 3 0"}

# MPTP
# EXPERIMENTS = {115403: "Clean", 115404: "Sine 1 1", 115405: "Sine 3 0"}

# Mizar 40
# EXPERIMENTS = {115407: "Clean", 115408: "Sine 1 1", 115409: "Sine 3 0"}

# Sampling experiments
# With --schedule default
"""
EXPERIMENTS = {
    115446: "caption_greedy_samples_1",
    115448: "caption_temp_0.5_samples_2",
    115447: "caption_greedy_samples_4",
}
"""

'''
# With --schedule none
EXPERIMENTS = {
    115449: "caption_temp_0.5_samples_2",
    115450: "caption_greedy_samples_1",
    115451: "caption_greedy_samples_4",
    115452: "caption_greedy_samples_4_sine_1_1",
    115453: "caption_greedy_samples_4_sine_3_0",
}
'''

# Rerun with --schedule none on deepmath
EXPERIMENTS = {115496: "Positive Axioms",
               115497: "Clean",
               115498: "Ideal",
               #115499: "Sine 1 1",
               115500: "Sine 1 1 ", # Rerun
               115501: "Sine 3 0",
               115502: "caption_greedy_no_samples_4_length_22",
               #115503: "caption_sine_1_1_greedy_no_samples_4_length_22",
               #115504: "caption_sine_3_0_greedy_no_samples_4_length_22",
               115505: "caption_sine_3_0_greedy_no_samples_4_length_22 (rerun)",
               115506: "caption_sine_1_1_greedy_no_samples_4_length_22 (rerun)"}

# Mizar 40 problem versions
EXPERIMENTS = {115507: 'Clean', 115508: 'Sine 1 1', 115509: "Sine 3 0",
               115510: "mizar_40/caption_sine_3_0_greedy_no_samples_4_length_22",
               115511: "mizar_40/caption_greedy_no_samples_4_length_22/",
               115512: "mizar_40/caption_sine_1_1_greedy_no_samples_4_length_22/"}

# Vocab size 6K
EXPERIMENTS = {115513: "mizar_40/vocab_6k_/caption_greedy_no_samples_2_length_22",
               115514: "mizar_40/vocab_6k_/caption_greedy_no_samples_4_length_22",
               115515: "mizar_40/vocab_6k_/caption_sine_1_1_greedy_no_samples_2_length_22",
               115516: "mizar_40/vocab_6k_/caption_sine_1_1_greedy_no_samples_4_length_22",
               115519: "mizar_40/vocab_6k_/caption_sine_3_0_greedy_no_samples_2_length_22",
               115520: "mizar_40/vocab_6k_/caption_sine_3_0_greedy_no_samples_4_length_22"}


# Vocab size 6K Merged datasets
EXPERIMENTS = {115554: "Clean",
               115555: "Sine 1 1",
               115556: "Sine 3 0",
               115557: "Caption Model",
               115558: "Caption + Sine 1 1",
               115563: "Caption + Sine 3 0",
               }

# FIXME - clean all this up!
# New experiments where I have fixed the missing conjecture for captioning
# FIXME included the other ones for comparision
EXPERIMENTS = {115507: 'Clean',
               115514: "mizar_40/vocab_6k_/caption_greedy_no_samples_4_length_22",
               115516: "mizar_40/vocab_6k_/caption_sine_1_1_greedy_no_samples_4_length_22",
               115520: "mizar_40/vocab_6k_/caption_sine_3_0_greedy_no_samples_4_length_22",
               115588: "skolem_caption",
               115589: "skolem_caption_sine_1_1",
               115590: "skolem_caption_sine_3_0",
               115591: "merged_caption",
               115592: "merged_caption_sine_1_1",
               115593: "merged_caption_sine_3_0"}

"""
EXPERIMENTS = {115602: "fixed_mizar_caption_sine_1_1",
               115616: "fixed_mizar_caption_sine_3_0",
               115633: "fixed_merged_caption_sine_1_1",
               115634: "fixed_merged_caption_sine_3_0"}

EXPERIMENTS = {116971: "sine_0_1",
               116972: "sine_1_1",
               116973: "sine_2_1",
               116974: "sine_3_1",
               116975: "sine_4_1",
               116976: "sine_5_1",
               116977: "sine_6_1",
               117013: "sine_7_1",
               117014: "sine_8_1",
               117015: "sine_9_1",
               116978: "sine_0_2",
               116979: "sine_1_2",
               116980: "sine_2_2",
               116985: "sine_3_2",
               116986: "sine_4_2",
               116987: "sine_5_2",
               116988: "sine_6_2",
               117016: "sine_7_2",
               117017: "sine_8_2",
               117018: "sine_9_2",
               116989: "sine_0_3",
               116990: "sine_1_3",
               116991: "sine_2_3",
               116992: "sine_3_3",
               116993: "sine_4_3",
               116994: "sine_5_3",
               116995: "sine_6_3",
               117019: "sine_7_3",
               117020: "sine_8_3",
               117021: "sine_9_3",
               116996: "sine_0_4",
               116997: "sine_1_4",
               116998: "sine_2_4",
               116999: "sine_3_4",
               117000: "sine_4_4",
               117001: "sine_5_4",
               117002: "sine_6_4",
               117022: "sine_7_4",
               117023: "sine_8_4",
               117024: "sine_9_4",
               117003: "sine_0_0",
               117004: "sine_1_0",
               117005: "sine_2_0",
               117006: "sine_3_0",
               117007: "sine_4_0",
               117008: "sine_5_0",
               117009: "sine_6_0",
               117010: "sine_7_0",
               117011: "sine_8_0",
               117012: "sine_9_0"}
"""


# Comparing captioning with and without axiom_remapping
EXPERIMENTS = {
    115554: "Clean",
    115591: "Caption Model",
    115634: "Caption + Sine 3 0",
    117090: "Caption axiom_remapping",
    117091: "Caption axiom_remapping + Sine 3 0"
}


# Initial results on the ideal and merged+clean version of the dataset for different time-limits
EXPERIMENTS = {
    117207: "ideal_1s",
    117208: "ideal_2s:",
    117209: "ideal_10s",
    117210: "ideal_50s",
    117211: "ideal_100s",
    117212: "clean_1s",
    117213: "clean_2s",
    117214: "clean_10s",
    117221: "clean_50s",
    117222: "clean_100s",
}


DATA_SETS = {"total": None, "train": "train.txt", "test": "test.txt"}


def _base_query_solved_problemrun(select, exp_id, upper_time_bound, problem_set=None):

    try:
        db_conn_details = db_cred.db_connection_details
        conn = db.connect(**db_conn_details)
        curs = conn.cursor()

        sql_query = """
            {0}
            FROM ProblemRun PR
            """.format(
            select
        )

        # Join on appopriate tables to be able get status information
        sql_query += """
            JOIN ProblemVersion PV ON PV.ProblemVersionID = PR.Problem
            JOIN SZSStatus SL ON PV.Status = SL.SZSStatusID
            JOIN SZSStatus SR ON PR.Status = SR.SZSStatusID
            """

        # Join name information - TODO cannot make this work...
        sql_query += """
                     JOIN Problem P ON P.ProblemID = PV.Problem
                     """

        sql_query += """
                WHERE PR.Experiment={0}
                """.format(
            exp_id
        )

        if problem_set is not None:
            sql_query += """
                         AND PR.Problem in
                         (
                            SELECT ProblemVersionID
                            FROM Problem P
                            INNER JOIN ProblemVersion PV ON P.ProblemID = PV.Problem
                            WHERE ProblemName IN ({0}) and PV.Version={1}
                         )
                        """.format(
                '"' + '", "'.join(problem_set) + '"', LIBRARY_VERSION
            )

        # Apply upper time bound if set
        if upper_time_bound is not None:
            sql_query += """
                    AND PR.Runtime <= {0}
                    """.format(
                upper_time_bound
            )

        if LTB_PROBLEMS:
            sql_query += """
                 AND (PR.SZS_Status = 'Theorem'
                 OR PR.SZS_Status = 'Unsatisfiable')
                         """
        else:
            sql_query += """
                 AND SR.Solved = 1
                         """

        # Remove the incorrects by only including the correct
        if not INCLUDE_INCORRECT:
            sql_query += """
                AND ((SL.Unsat AND SR.Sat)
                    OR (SL.Sat AND SR.Unsat)) = 0"""

        # Execute
        sql_query += ";"
        curs.execute(sql_query)

        res = curs.fetchall()

        return res

    except Exception as err:
        print(err)
        print(traceback.format_exc())
    finally:
        if curs:
            curs.close()
        if conn:
            conn.close()


def get_no_solved(exp_id, upper_time_bound=None, problem_set=None):

    select = "SELECT IFNULL(COUNT(*), 0)"
    res = _base_query_solved_problemrun(select, exp_id, upper_time_bound, problem_set)
    return int(res[0][0])


def get_problem_stats(exp_id, upper_time_bound=None, problem_set=None):

    select = "SELECT PR.Problem, PR.Runtime"
    res = _base_query_solved_problemrun(select, exp_id, upper_time_bound, problem_set)
    return res


def get_solved_problem_name_time(exp_id, upper_time_bound=None, problem_set=None):

    select = "SELECT ProblemName, PR.Runtime"
    res = _base_query_solved_problemrun(select, exp_id, upper_time_bound, problem_set)
    return res


def get_solved_problem_name(exp_id, upper_time_bound=None, problem_set=None):

    select = "SELECT ProblemName"
    res = _base_query_solved_problemrun(select, exp_id, upper_time_bound, problem_set)
    return res



def get_avg_time_solved(exp_id, upper_time_bound=None, problem_set=None):

    select = "SELECT IFNULL(AVG(PR.RunTime), 0)"
    res = _base_query_solved_problemrun(select, exp_id, upper_time_bound, problem_set)
    return float(res[0][0])


def load_data_set(prob_set_file):
    with open(os.path.join(DATASET_DIR, prob_set_file), "r") as f:
        prob_set = f.readlines()

    prob_set = [prob.strip() for prob in prob_set]
    return prob_set


def main():

    for exp, desc in EXPERIMENTS.items():
        print(f"### {exp}: {desc}")

        results = {}
        for name, prob_set in DATA_SETS.items():

            # Load dataset file - None is for all problems
            if prob_set is not None:
                prob_set = load_data_set(prob_set)

            solved = get_no_solved(exp, problem_set=prob_set)
            time = get_avg_time_solved(exp, problem_set=prob_set)
            results[name] = {"solved": solved, "time": time}

        for name, res in results.items():
            print(name)
            print(f"  Solved  : {res['solved']}")
            print(f"  Avg Time: {res['time']:.2f}")
        print()


if __name__ == "__main__":
    main()
