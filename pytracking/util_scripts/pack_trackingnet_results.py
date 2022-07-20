import numpy as np
import os
import shutil
import argparse

import sys
env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation.environment import env_settings
from pytracking.evaluation.datasets import get_dataset


def pack_trackingnet_results(tracker_name, param_name, run_id=None, output_name=None):
    """ Packs trackingnet results into a zip folder which can be directly uploaded to the evaluation server. The packed
    file is saved in the folder env_settings().tn_packed_results_path

    args:
        tracker_name - name of the tracker
        param_name - name of the parameter file
        run_id - run id for the tracker
        output_name - name of the packed zip file
    """

    if output_name is None:
        if run_id is None:
            output_name = '{}/{}'.format(tracker_name, param_name)
        else:
            output_name = '{}/{}_{:03d}'.format(tracker_name, param_name, run_id)

    output_path = os.path.join(env_settings().tn_packed_results_path, output_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    results_path = env_settings().results_path

    tn_dataset = get_dataset('trackingnet')

    for seq in tn_dataset:
        seq_name = seq.name

        if run_id is None:
            seq_results_path = '{}/{}/{}/{}.txt'.format(results_path, tracker_name, param_name, seq_name)
        else:
            seq_results_path = '{}/{}/{}_{:03d}/{}.txt'.format(results_path, tracker_name, param_name, run_id, seq_name)

        results = np.loadtxt(seq_results_path, dtype=np.float64)

        np.savetxt('{}/{}.txt'.format(output_path, seq_name), results, delimiter=',', fmt='%.2f')

    # Generate ZIP file
    shutil.make_archive(output_path, 'zip', output_path)

    # Remove raw text files
    shutil.rmtree(output_path)


def main():
    parser = argparse.ArgumentParser(description='Pack trackingnet results.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--out', type=str, default=None, help='Output name')

    args = parser.parse_args()

    pack_trackingnet_results(args.tracker_name, args.tracker_param, args.runid, args.out)


if __name__ == '__main__':
    main()