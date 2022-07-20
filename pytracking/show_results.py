import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from pytracking.evaluation import Tracker, get_dataset, trackerlist
from pytracking.evaluation import get_dataset


def show_result(tracker_name, tracker_param, run_id, dataset_name):
    trackers = [Tracker(tracker_name, tracker_param, run_id)]

    dataset = get_dataset(dataset_name)
    print_results(trackers, dataset, dataset_name, force_evaluation=True, merge_results=False, plot_types=('success', 'prec', 'norm_prec'))



def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')

    args = parser.parse_args()

    show_result(args.tracker_name, args.tracker_param, args.runid, args.dataset_name)


if __name__ == '__main__':
    main()