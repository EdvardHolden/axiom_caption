''' Downloads proof data from the CASC competitions

This script takes the result entries in config.COMPETITION_RESULTS amd downloads
the proof data of each attempt and store it as a pickle. The data contains the
problem version, the solving time reported in the attempt and the axioms used
in the proof. It can handle iProver, Vampire and E proofs. The resulting pickle
is stored in data/raw with a pickle object for each competition-prover.
'''


from bs4 import BeautifulSoup
import requests
import re
from pickle import dump
import os
import multiprocessing
import argparse
from dataclasses import dataclass
import config

RESULT_DIR = 'data/raw/'

parser = argparse.ArgumentParser()
parser.add_argument('--force_download_all', default=False, action='store_true',
                    help='Force (re-)download the proof data from each competition.')


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
    except requests.exceptions.ConnectionError:
        return None
    except Exception as err:
        print(f"Unexpected exception for {proof_url}: {err}")
        return None

    return x.text


def get_axioms_from_proof(proof, e_prover_proof):
    if e_prover_proof:
        # For E
        # TODO am I getting the correct group???
        axiom_regex = "^(fof|cnf|tff)\\(\\w+, axiom,(.*)$"
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
        if problem is None or data is None:  # Faulty result
            continue
        elif data['version'] is None:  # No problem version
            continue
        # Checks passed, keep proof
        axioms[problem] = data

    return axioms


@dataclass
class Entry:
    competition: str
    prover: str
    url: str

    def get_pickle_path(self):
        return f'{self.prover}_{self.competition}.pkl'

    def __repr__(self):
        return f'{self.prover}_{self.competition}'


def get_all_competition_entries():
    # Extract all entries from the results dict
    entries = []
    for competition, systems in config.COMPETITION_RESULTS.items():
        for prover, url in systems.items():
            entries += [Entry(competition, prover, url)]

    return entries


def main():

    args = parser.parse_args()

    # Get all links and competitions
    entries = get_all_competition_entries()

    if args.force_download_all:
        print(f'Downloading all {len(entries)} competition entries')

    for entry in entries:
        if not args.force_download_all and os.path.exists(RESULT_DIR + entry.get_pickle_path()):
            print(f'Proof data already exists for {entry}. Skipping...')
            continue

        print("Downloading the results from: ", entry)
        e_prover_proof = entry.prover == 'e'
        data = get_proof_data(entry.url, e_prover_proof)

        # Save the dict
        save_path = f'{RESULT_DIR}{entry.get_pickle_path()}'
        with open(save_path, 'wb') as f:
            dump(data, f)
        print(f'Saved to: {save_path}')
        del data


if __name__ == "__main__":
    main()
