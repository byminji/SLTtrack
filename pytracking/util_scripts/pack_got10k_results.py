import numpy as np
import os
import shutil
import argparse
import sys

env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)
from pytracking.evaluation.environment import env_settings


def pack_got10k_results(tracker_name, param_name, output_name=None, split='Test'):
    """ Packs got10k results into a zip folder which can be directly uploaded to the evaluation server. The packed
    file is saved in the folder env_settings().got_packed_results_path

    args:
        tracker_name - name of the tracker
        param_name - name of the parameter file
        train_name - name of the train version
        output_name - name of the packed zip file
        split - test/val
    """
    if output_name is None:
        output_name = '{}/{}'.format(tracker_name, param_name)

    output_path = os.path.join(env_settings().got_packed_results_path, output_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    results_path = env_settings().results_path
    for i in range(1,181):
        seq_name = 'GOT-10k_{}_{:06d}'.format(split, i)

        seq_output_path = '{}/{}'.format(output_path, seq_name)
        if not os.path.exists(seq_output_path):
            os.makedirs(seq_output_path)

        for run_id in range(3):
            res = np.loadtxt('{}/{}/{}_{:03d}/{}.txt'.format(results_path, tracker_name, param_name, run_id, seq_name), dtype=np.float64)
            times = np.loadtxt(
                '{}/{}/{}_{:03d}/{}_time.txt'.format(results_path, tracker_name, param_name, run_id, seq_name),
                dtype=np.float64)

            np.savetxt('{}/{}_{:03d}.txt'.format(seq_output_path, seq_name, run_id+1), res, delimiter=',', fmt='%f')
            np.savetxt('{}/{}_time.txt'.format(seq_output_path, seq_name), times, fmt='%f')

    # Generate ZIP file
    shutil.make_archive(output_path, 'zip', output_path)

    # Remove raw text files
    shutil.rmtree(output_path)



def main():
    parser = argparse.ArgumentParser(description='Pack got-10k results.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('--split', type=str, default='Test', help='split')
    parser.add_argument('--out', type=str, default=None, help='Output name')

    args = parser.parse_args()

    pack_got10k_results(args.tracker_name, args.tracker_param, args.out, args.split)


if __name__ == '__main__':
    main()