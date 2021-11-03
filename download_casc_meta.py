"""
This program downloads the problem meta, which is the problem name, file,
and solving time of a division in the CASC results.
"""
import pickle
import requests
import re
import multiprocessing

from download_axioms import get_problem_files


# TODO - store this in some config?
META_PATHS = {'jjt': {
    'e': 'http://www.tptp.org/CASC/28/WWWFiles/Results/JJT/E---LTB-2.6/',
    'vampire': 'http://www.tptp.org/CASC/28/WWWFiles/Results/JJT/Vampire---LTB-4.6/',
    'iprover': 'http://www.tptp.org/CASC/28/WWWFiles/Results/JJT/iProver---LTB-3.5/'
},
    'hl4': {
        'e': 'http://www.tptp.org/CASC/J10/WWWFiles/Results/HL4/E---LTB-2.5/',
        'iprover': 'http://www.tptp.org/CASC/J10/WWWFiles/Results/HL4/iProver---LTB-3.3/'
}
}


def get_proof_meta(results_url, prob_name):
    # Properties: name, version, time
    # Extract problem name from url
    try:
        #print(results_url + prob_name)
        proof = requests.get(results_url + prob_name).text
    except Exception:
        return prob_name, {'version': None, 'time': None}

    version = get_problem_version(proof)
    time = get_solving_time(proof)

    return prob_name, {'version': version, 'time': time}


def get_solving_time(proof):
    try:
        time_regex = "^([0-9]*[.])?[0-9]+\\/(([0-9]*[.])?[0-9]+)\\sEOF$"
        time = re.findall(time_regex, proof, re.MULTILINE)
        if len(time) < 1:
            return None
        # Return WC time
        return round(float(time[0][1]), 2)
    except Exception:
        return None


def get_problem_version(proof):
    try:
        #version_regex = "^% SZS status [[:word:]]+ for ([[:alnum:]]|_)+$"
        version_regex = "^[%|#] SZS status .*$"
        version = re.findall(version_regex, proof, re.MULTILINE)
        if len(version) < 1:
            return None
        # Get the match and format into name_v
        version = version[0].split()[-1].split('/')[-1]
        if '.p' in version:
            version = version[:-2]

        return version
    except Exception:
        return None


def get_meta_data(results_url):
    problems = get_problem_files(results_url)

    pool = multiprocessing.Pool(processes=4)
    arguments = [(results_url, prob) for prob in problems]
    res = pool.starmap(get_proof_meta, arguments)
    pool.close()
    pool.join()

    # Build dict from results
    meta_data = {name: data for name, data in res}

    return meta_data


def main():

    for division, provers in META_PATHS.items():
        # Build meta-data for each division
        meta_data_division = {}
        for prover, result_url in provers.items():
            print(f"# Extracting meta for div {division} of prover {prover}")
            meta_data_division[prover] = get_meta_data(result_url)

        """
        for k, v in meta_data_division.items():
            print(k)
            print(v)
            print()
        """

        # Save for each division
        save_path = 'data/raw/' + division + '_meta.pkl'
        print("# Saving meta to: ", save_path)
        with open(save_path, 'wb') as f:
            pickle.dump(meta_data_division, f)
        print()


if __name__ == "__main__":
    main()
    """
    m = get_proof_meta('http://www.tptp.org/CASC/28/WWWFiles/Results/JJT/Vampire---LTB-4.6/', 'JJT00140')
    print(m)
    e = get_proof_meta('http://www.tptp.org/CASC/28/WWWFiles/Results/JJT/E---LTB-2.6/', 'JJT00001')
    print(e)
    #"""
