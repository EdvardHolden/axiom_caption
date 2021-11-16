'''
TODO write documentation
'''





from bs4 import BeautifulSoup
import requests
import re
from tqdm import tqdm
from pickle import dump
import os
import multiprocessing


# TODO tasklist
# Do not load things that already exists.
# But can force download!


"""
fof(f4,axiom,(
  ~p(s(tyop_2Emin_2Ebool,c_2Ebool_2EF_2E0))),
  file('/export/starexec/sandbox/benchmark/Problems/HL400003+4.p',unknown)).
"""


# Get list of all problem links
#BASE_URL = "http://www.tptp.org/CASC/J10/WWWFiles/Results/HL4/iProver---LTB-3.3/"
##BASE_URL = "http://www.tptp.org/CASC/28/WWWFiles/Results/JJT/iProver---LTB-3.5/"
#BASE_URL = "http://www.tptp.org/CASC/J10/WWWFiles/Results/HL4/E---LTB-2.5/"
BASE_URL = "http://www.tptp.org/CASC/28/WWWFiles/Results/JJT/E---LTB-2.6/"

# Vampire
#BASE_URL = "http://www.tptp.org/CASC/28/WWWFiles/Results/JJT/Vampire---LTB-4.6/"

def get_proof_links(url):

    # Get html content
    x = requests.get(url)
    html_content = x.text

    # Extract the hrefs in the table
    soup = BeautifulSoup(html_content, 'html.parser')
    links = []
    for link in soup.findAll('a'):
        links.append(link.get('href'))

    print("Number of links retrieved: ", len(links))
    # Append the base link
    links = [url + link for link in links]
    return links

def get_proof_from_link(proof_url):

    # Get html content
    try:
        x = requests.get(proof_url)
    except Exception:
        # Chance of requests.exceptions.ConnectionError
        return None
    return x.text

def get_axioms_from_proof(proof, e_prover_proof):
    if e_prover_proof:
        # For E
        axiom_regex = "^(fof|cnf|tff)\(\w+, axiom,(.*)$"
        result = re.findall(axiom_regex, proof, re.MULTILINE)
        axioms = [r[1].split("file")[0] for r in result]
    else:
        # For iProver|Vampire
        axiom_regex = "^((fof|cnf|tff)\\(f[0-9]+,axiom(.*\n){3})"
        result = re.findall(axiom_regex, proof, re.MULTILINE)
        axioms = [r[0].split("\n")[1] for r in result]

    return axioms


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



def compute_proof_data(proof_url, e_prover_proof):
    # Extract problem name from the link
    problem = proof_url.split('/')[-1]
    if problem == '':
        return problem, None

    # Get the proof output of the attempt
    proof = get_proof_from_link(proof_url)
    if proof is None:
        return problem, None

    # Get the proof data
    axioms = get_axioms_from_proof(proof, e_prover_proof)
    version = get_problem_version(proof)
    time = get_solving_time(proof)

    result = {'axioms': axioms,
              'version': version,
              'time': time}

    return problem, result


def get_proof_data(results_url, e_prover_proof):

    # Get list of all problems with an iProver proof
    proof_links = get_proof_links(results_url)
    print("Number of problems: ", len(proof_links))
    proof_links = proof_links[:10] # TODO

    # Make a process pool
    pool = multiprocessing.Pool(processes=os.cpu_count() - 1)

    # Compute the processes
    arguments = [(link, e_prover_proof) for link in proof_links]
    res = pool.starmap(compute_proof_data, arguments)

    # No more processes to the queue
    pool.close()
    # Wait for pool to finish processing before carrying on
    pool.join()

    # Join the data together in a dict, and remove any faulty results
    axioms = dict()
    for problem, data in res:
        # Check if problems are worth keeping
        if problem is None or data is None: # Faulty result
           continue
        elif data['version'] is None: # No problem version
           continue
        # Checks passed, keep proof
        axioms[problem] = data

    return axioms




def main():

    # Force all + force only some
    # TODO
    # Handle input arguments and prover
    # Get hold of links from config
    # For each link, run the script
    # Save the pickle in this script??
    URL = "http://www.tptp.org/CASC/J10/WWWFiles/Results/HL4/iProver---LTB-3.3/"
    prover = 'iprover'
    competition = 'hl4'

    print("Running: ", URL)
    e_prover_proof = False
    data = get_proof_data(URL, e_prover_proof) # TODO set
    print(data)

    # Save the dict
    with open(f'data/raw/{prover}_{competition}.pkl', 'wb') as f:
        dump(data, f)
    del data


if __name__ == "__main__":
    main()

    # Testing links
    #proof_url = "http://www.tptp.org/CASC/J10/WWWFiles/Results/HL4/iProver---LTB-3.3/HL400003"
    # proof_url = "http://www.tptp.org/CASC/J10/WWWFiles/Results/HL4/iProver---LTB-3.3/HL400013" # Where is the axiom?
