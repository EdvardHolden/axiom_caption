
import mysql.connector as db
import db_cred
import traceback
import os

# _base_query_solved_problemrun_filter_collection
# First get for a single experiment
# Want numbr of problems solved (UNSAT) and average solving time
# Then  want divided by sets

LTB_PROBLEMS = True
INCLUDE_INCORRECT = False

# Current library version of deepmath in the DB
LIBRARY_VERSION = 64

DATASET_DIR = "../data/deepmath/"

EXPERIMENTS = {115360: 'test experiment'
}

DATA_SETS = {'total': None,
             'train': 'train.txt',
             'test': 'test.txt'}



def _base_query_solved_problemrun(select, exp_id, upper_time_bound, problem_set=None):

    try:
        db_conn_details = db_cred.db_connection_details
        conn = db.connect(**db_conn_details)
        curs = conn.cursor()

        sql_query = '''
            {0}
            FROM ProblemRun PR
            '''.format(select)

        # Join on appopriate tables to be able get status information
        sql_query += '''
            JOIN ProblemVersion PV ON PV.ProblemVersionID = PR.Problem
            JOIN SZSStatus SL ON PV.Status = SL.SZSStatusID
            JOIN SZSStatus SR ON PR.Status = SR.SZSStatusID
            '''

        sql_query += '''
                WHERE PR.Experiment={0}
                '''.format(exp_id)

        #SELECT ProblemID from Problem P INNER JOIN ProblemVersion PV ON P.ProblemID = PV.Problem WHERE ProblemName IN ("l13_euclid_8", "t23_waybel_4") and PV.Version=64
        # TODO make if on this?

        if problem_set is not None:
            sql_query += '''
                         AND PR.Problem in
                         (
                            SELECT ProblemVersionID
                            FROM Problem P
                            INNER JOIN ProblemVersion PV ON P.ProblemID = PV.Problem
                            WHERE ProblemName IN ({0}) and PV.Version={1}
                         )
                        '''.format("\"" + "\", \"".join(problem_set) + "\"",
                                   LIBRARY_VERSION)

        # Apply upper time bound if set
        if upper_time_bound is not None:
            sql_query += '''
                    AND PR.Runtime <= {0}
                    '''.format(upper_time_bound)

        if LTB_PROBLEMS:
            sql_query += '''
                 AND (PR.SZS_Status = 'Theorem'
                 OR PR.SZS_Status = 'Unsatisfiable')
                         '''
        else:
            sql_query += '''
                 AND SR.Solved = 1
                         '''

        # Remove the incorrects by only including the correct
        if not INCLUDE_INCORRECT:
            sql_query += '''
                AND ((SL.Unsat AND SR.Sat)
                    OR (SL.Sat AND SR.Unsat)) = 0'''

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


def get_avg_time_solved(exp_id, upper_time_bound=None, problem_set=None):

    select = "SELECT IFNULL(AVG(PR.RunTime), 0)"
    res = _base_query_solved_problemrun(select, exp_id, upper_time_bound, problem_set)
    return float(res[0][0])

def load_data_set(prob_set_file):
    with open(os.path.join(DATASET_DIR, prob_set_file), 'r') as f:
        prob_set = f.readlines()

    prob_set = [prob.strip() for prob in prob_set]
    return prob_set


def main():

    for exp, desc in EXPERIMENTS.items():
        print(f'### {exp}: {desc}')

        results = {}
        for name, prob_set in DATA_SETS.items():

            # Load dataset file - None is for all problems
            if prob_set is not None:
                prob_set = load_data_set(prob_set)

            solved = get_no_solved(exp, problem_set=prob_set)
            time = get_avg_time_solved(exp, problem_set=prob_set)
            results[name] = {'solved': solved, 'time': time}

        for name, res in results.items():
            print(name)
            print(f" Solved  : {res['solved']}")
            print(f" Avg Time: {res['time']:.2f}")
        print()


if __name__ == "__main__":
    main()


