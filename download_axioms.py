# TODO distinguish between +4 and +5 problems?
from bs4 import BeautifulSoup
import requests
import re
from tqdm import tqdm
from pickle import dump
import os
import multiprocessing

# TODO rename file paths?

"""
fof(f4,axiom,(
  ~p(s(tyop_2Emin_2Ebool,c_2Ebool_2EF_2E0))),
  file('/export/starexec/sandbox/benchmark/Problems/HL400003+4.p',unknown)).
"""

E_PROOF = True

if E_PROOF:
    # For E
    axiom_regex = "^(fof|cnf|tff)\(\w+, axiom,(.*)$"
else:
    # For iProver|Vampire
    axiom_regex = "^((fof|cnf|tff)\\(f[0-9]+,axiom(.*\n){3})"


# Get list of all problem links
#BASE_URL = "http://www.tptp.org/CASC/J10/WWWFiles/Results/HL4/iProver---LTB-3.3/"
##BASE_URL = "http://www.tptp.org/CASC/28/WWWFiles/Results/JJT/iProver---LTB-3.5/"
#BASE_URL = "http://www.tptp.org/CASC/J10/WWWFiles/Results/HL4/E---LTB-2.5/"
BASE_URL = "http://www.tptp.org/CASC/28/WWWFiles/Results/JJT/E---LTB-2.6/"

# Vampire
#BASE_URL = "http://www.tptp.org/CASC/28/WWWFiles/Results/JJT/Vampire---LTB-4.6/"

def get_problem_files(url):

    # Get html content
    x = requests.get(url)
    html_content = x.text

    # Extract the hrefs in the table
    soup = BeautifulSoup(html_content, 'html.parser')
    links = []
    for link in soup.findAll('a'):
        links.append(link.get('href'))

    print("Number of links retrieved: ", len(links))
    return links


def get_proof_axioms(prob):
    # Function for extracting the axioms used in a proof file
    url = BASE_URL + prob

    # Get html content
    try:
        x = requests.get(url)
    except Exception:
        # Chance of requests.exceptions.ConnectionError
        return None
    proof_out = x.text

    # Find all axioms and extract the result
    result = re.findall(axiom_regex, proof_out, re.MULTILINE)
    if E_PROOF:
        result = [r[1].split("file")[0] for r in result]
    else:
        result = [r[0].split("\n")[1] for r in result]

    return prob, result


if __name__ == "__main__":

    # Testing links
    #proof_url = "http://www.tptp.org/CASC/J10/WWWFiles/Results/HL4/iProver---LTB-3.3/HL400003"
    # proof_url = "http://www.tptp.org/CASC/J10/WWWFiles/Results/HL4/iProver---LTB-3.3/HL400013" # Where is the axiom?

    # Get list of all problems with an iProver proof
    problems = get_problem_files(BASE_URL)
    print("Number of problems: ", len(problems))

    # Make a process pool
    pool = multiprocessing.Pool(processes=os.cpu_count() - 1)

    # Compute the processes
    res = pool.map(get_proof_axioms, problems)

    # No more processes to the queue
    pool.close()
    # Wait for pool to finish processing before carrying on
    pool.join()

    axioms = dict()
    for p, a in res:
        axioms[p] = a

    # Save the dict
    with open('data/axioms.pkl', 'wb') as f:
        dump(axioms, f)
