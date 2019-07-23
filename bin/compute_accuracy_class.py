import os
import sys
from parse import parse
import pickle as pkl
from gm_hmm.src.utils import append_class, accuracy_fun, divide, parse_
from functools import partial

if __name__ == "__main__":
    usage = "bin/compute_accuracy_class.py 13feats/models/epoch2_class1.mdlc 13feats/data/train.13.pkl 13feats/data/test.13.pkl"
    if len(sys.argv) != 4 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage, file=sys.stderr)
        sys.exit(1)

    models_dir, epoch, iclass = parse("{}/epoch{:d}_class{:d}.mdlc", sys.argv[1])

    training_data_file = sys.argv[2]
    testing_data_file = sys.argv[3]

    mdl_file = os.path.join(models_dir, "epoch{}.mdl".format(epoch))

    # Load Model
    with open(mdl_file, "rb") as handle:
        mdl = pkl.load(handle)

    # Prepare for computation of results
    nclasses = len(mdl.hmms)

    # Builds an array of string containing the train and test data sets for each class
    # size: nclass x 2 (train, test)
    data_files = [append_class(training_data_file, iclass), append_class(testing_data_file, iclass)]
    #f = lambda x: divide(parse_(accuracy_fun(x, mdl=mdl)))
    f = lambda x: accuracy_fun(x, mdl=mdl)
    results = list(map(f, data_files))

    print("epoch: {} class: {} accc train: {} test: {}".format(epoch, iclass, results[0], results[1]), file=sys.stdout)
    sys.exit(0)