import pickle
from utils.hparams import make_hparams
import argparse
from utils import constants
from modes.train import run_train
from modes.train_hcm import run_train_hcm
from modes.test import run_test
from modes.test_hcm import run_test_hcm
from modes.interactive import run_interactive

if __name__ == "__main__":

    hparams = make_hparams()

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # TRAINING MODE
    subparser = subparsers.add_parser("train_meqsum")
    subparser.set_defaults(callback=lambda args: run_train(hparams))
    hparams.populate_arguments(subparser)

    # TRAINING MODE
    subparser = subparsers.add_parser("train_hcm")
    subparser.set_defaults(callback=lambda args: run_train_hcm(hparams))
    hparams.populate_arguments(subparser)

    # TESTING MODE
    subparser = subparsers.add_parser("test_meqsum")
    subparser.set_defaults(callback=lambda args: run_test(hparams))
    hparams.populate_arguments(subparser)

    # TESTING MODE
    subparser = subparsers.add_parser("test_hcm")
    subparser.set_defaults(callback=lambda args: run_test_hcm(hparams))
    hparams.populate_arguments(subparser)

    # INTERACTIVE MODE
    subparser = subparsers.add_parser("interactive")
    subparser.set_defaults(callback=lambda args: run_interactive(hparams))
    hparams.populate_arguments(subparser)

    args = parser.parse_args()
    hparams.set_from_args(args)
    args.callback(args)
