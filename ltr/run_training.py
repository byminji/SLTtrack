import os
import sys
import argparse
import importlib
import multiprocessing
import cv2 as cv
import torch.backends.cudnn
import random
import numpy as np

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)
import ltr.admin.settings as ws_settings


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def run_training(args):
    """Run a train scripts in train_settings.
    args:
        train_module: Name of module in the "train_settings/" folder.
        train_name: Name of the train settings file.
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """

    # This is needed to avoid strange crashes related to opencv
    cv.setNumThreads(0)

    if args.cudnn_benchmark:
        print("Using CuDNN Benchmark")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    else:
        print("Disabling CuDNN Benchmark")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print('Training:  {}  {}'.format(args.train_module, args.train_name))

    settings = ws_settings.Settings()
    settings.module_name = args.train_module
    settings.script_name = args.train_name
    settings.project_path = 'ltr/{}/{}'.format(args.train_module, args.train_name)

    expr_module = importlib.import_module('ltr.train_settings.{}.{}'.format(args.train_module, args.train_name))
    expr_func = getattr(expr_module, 'run')

    expr_func(settings)

def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('train_module', type=str, help='Name of module in the "train_settings/" folder.')
    parser.add_argument('train_name', type=str, help='Name of the train settings file.')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Set cudnn benchmark on (1) or off (0) (default is on).')

    args = parser.parse_args()

    run_training(args)

if __name__ == '__main__':
    seed_torch(12345)
    multiprocessing.set_start_method('spawn', force=True)
    main()
