# A file to display the confusion matrix 
import sys
import os
from functools import partial
import pickle as pkl
from gm_hmm.src.genHMM_GLOW import GenHMMclassifier
import numpy as np
from parse import parse
from gm_hmm.src.utils import divide, acc_str, append_class, parse_, accuracy_fun
import torch
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt
from scipy.io import savemat

def print_results(mdl_file, data_files, results):
    epoch = parse("epoch{}.mdl", os.path.basename(mdl_file))[0]
    # Print class by class accuracy
    for res, data_f in zip(results, data_files):
        true_class = parse("{}_{}.pkl", os.path.basename(data_f[0]))[1]
        
        print("epoch:", epoch, "class:", true_class, mdl_file, data_f[0].astype("<U"), res[0], divide(parse_(res[0].astype("<U"))), sep='\t', file=sys.stdout)
        print("epoch:", epoch, "class:", true_class, mdl_file, data_f[1].astype("<U"), res[1], divide(parse_(res[1].astype("<U"))), sep='\t', file=sys.stdout)

    # Print total accuracy
    res = np.concatenate([np.array([parse_(r[0].astype("<U")), parse_(r[1].astype("<U"))]).T for r in results],axis=1)
    tr_res = res[:, ::2]
    te_res = res[:, 1::2]
    tr_res_str = str(tr_res[0].sum()/tr_res[1].sum())
    te_res_str = str(te_res[0].sum()/te_res[1].sum())
    print("epoch:", epoch, "Acc:", mdl_file, tr_res_str, te_res_str, sep='\t', file=sys.stdout)

if __name__ == "__main__":
    usage = "Usage: python bin/compute_accuracy_cfmatrix.py [mdl file] [ training and testing data .pkl files]\n" \
            "Example: python bin/compute_accuracy_cfmatrix.py models/epoch1.mdl data/train.39.pkl data/test.39.pkl" \

    if len(sys.argv) != 4 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage, file=sys.stdout)
        sys.exit(1)

    # Parse argument
    mdl_file = sys.argv[1]
    training_data_file = sys.argv[2]
    testing_data_file = sys.argv[3]

    # Load Model
    with open(mdl_file, "rb") as handle:
        mdl = pkl.load(handle)

    # Prepare for computation of results
    nclasses = len(mdl.hmms)
    totclasses = 39 # Initialise total number of classes
    cf_matrix = np.zeros((nclasses, nclasses))

    # Builds an array of string containing the train and test data sets for each class
    # size: nclass x 2 (train, test)
    #data_files = np.array([[append_class(training_data_file, iclass+1), append_class(testing_data_file, iclass+1)]
    #               for iclass in range(nclasses)])
    tr_data_files = np.array([[append_class(training_data_file, iclass+1)]
                   for iclass in range(nclasses)])
    te_data_files = np.array([[append_class(testing_data_file, iclass+1)]
                   for iclass in range(nclasses)])

    # Define a function for this particular HMMclassifier model
    #f = partial(accuracy_fun, mdl=mdl)
    #out = [[f(data_files[i, j]) for j in range(data_files.shape[1])] for i in range(data_files.shape[0])]
    #results = np.array(out)
    #print_results(mdl_file, data_files, results)
    
    # Obtain confusion matrix for test files
    for i in range(te_data_files.shape[0]):
        
        data_file = te_data_files[i]
        try:
            X = pkl.load(open(data_file, "rb"))
        except:
            return "0/1"
        # Get the length of all the sequences
        l = [xx.shape[0] for xx in X]
        # zero pad data for batch training

        true_class = parse("{}_{}.pkl", os.path.basename(data_file))[1]
        out_list = [mdl.forward(x_i[:,1:]) for x_i in X]
        out = np.array(out_list).transpose()

        # the out here should be the shape: data_size * nclasses
        class_hat = np.argmax(out, axis=0) + 1
        istrue = class_hat == int(true_class)
        print(data_file, "Done ...", "{}/{}".format(str(istrue.sum()), str(istrue.shape[0])))

        nclasses_arr = [int(c+1) for c in range(nclasses)]
        for c in nclasses_arr:
            istrue_c = class_hat == c
            cf_matrix[int(true_class) - 1, c-1] = istrue.sum()
        
    # Plotting the confusion matrix

    classmap_file = "./data/class_map.json"
    
    # load the classmap dict
    with open(classmap_file) as f:
        classmap = json.load(f)
    
    # Save the confusion matrix


    # Create labels for confusion matrix
    x_labels = [classmap[str(x1)] for x1 in range(cf_matrix.shape[1])]
    y_labels = [classmap[str(y1)] for y1 in range(cf_matrix.shape[0])]

    fig, ax = plt.subplots()
    plt.title(" Confusion Matrix for classification of {}/{} classes".format(nclasses, totclasses))
    cf = ax.imshow(A, aspect='auto', cmap='jet')
    ax.set_xticks([x1 for x1 in range(A.shape[1])])
    ax.set_yticks([x1 for x1 in range(A.shape[1])])
    ax.set_yticklabels(y_labels)
    ax.set_xticklabels(x_labels)
    fig.colorbar(cf, ax=ax)
    #plt.show()
    plt.savefig("./models/" + "cfmatrix.pdf")
    plt.close()
    sys.exit(0)