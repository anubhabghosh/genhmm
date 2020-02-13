import os
import sys
from parse import parse
import pickle as pkl
import json
import numpy as np
import time

from gm_hmm.src.ref_hmm import Gaussian_HMM, GMM_HMM, ConvgMonitor
#from src.ref_hmm import Gaussian_HMM, GMM_HMM, ConvgMonitor
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_random_state


if __name__ == "__main__":
    
    # This 'usage' string basically outlines how the given function 'train_class_gaus.py' should be called 
    # by the user. It is not a variable that is used later in the code. For Debug options, one should open
    # the 'launch.json' file and add the file name arguments under 'args'
    #usage = "python bin/train_class_gaus.py exp/gaus/39feats/data/train39.pkl exp/gaus/39feats/models/epoch1_class1.mdlc param.json"
    usage = "python bin/train_class_gaus.py exp/gaus/39feats/data/train.39.pkl exp/gaus/39feats/models/epoch1_class1.mdlc param.json"
    
    # If insufficient number of arguments are present, the system will exit
    if len(sys.argv) < 3 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage)
        sys.exit(1)

    # Parse the argument list and obtain the location of the input file (which are features stored in .pkl) 
    # and the location of the output model. NOTE: the Output model folder should be created beforehand
    train_inputfile = sys.argv[1]
    out_mdl = sys.argv[2]

    # Test for a third parameter
    # If no configuration file is present as 'param.json', the function chooses the default parameters in
    # 'default.json'
    try:
        param_file = sys.argv[3]
    except IndexError:
        param_file = "default.json"

    # Obtains the input file and epoch details for the particular class specified 
    # in the output model path
    epoch_str, iclass_str = parse('epoch{}_class{}.mdlc', os.path.basename(out_mdl))
    train_class_inputfile = train_inputfile.replace(".pkl", "_{}.pkl".format(iclass_str))

    # Load the data from the .pkl file
    # Each phoneme in the .pkl file is represented by a K-dimensional feature vector of MFCCs (K = 13 or 39)
    # Each vector 'x' is a collection of phonemes, aggregating all the phonemes makes the training data
    xtrain_ = pkl.load(open(train_class_inputfile, "rb")) # Loads the stream of bytes into the list
    xtrain = [x[:, 1:] for x in xtrain_] # Store the data first as a list of lists
    xtrain = np.concatenate(xtrain, axis=0) # Then concatenate all the lists together as a single numpy array
    #xtrain = xtrain[:100]
    # Get the length of all the sequences
    l = [x.shape[0] for x in xtrain_] # Get a list containing the sequence lengths of each sequence of phonemes
    
    # load the parameters
    with open(param_file) as f_in:
        options = json.load(f_in)

    # adaptive to set number of states as per the length of the sequence (l), NOTE: Have to understand why the mean length is divided by 2
    options["GMM"]["n_states"] = np.clip(int(np.floor(np.mean(l)/2)),
                                         options["GMM"]["n_states_min"],
                                         options["GMM"]["n_states_max"])

    #  Load or create model
    if epoch_str == '1':
        # init GaussianHMM model or GMM_HMM model by disable/comment one and enable another model. 
        # For GMM_HMM, we are now just maintaining diag type of covariance.
        mdl = GMM_HMM(n_components=options["Net"]["n_states"], \
                      n_mix=options["GMM"]["n_prob_components"], \
                      covariance_type="diag", tol=-np.inf, \
                      init_params="stwmc", params="stwmc", verbose=True)

        # mdl = Gaussian_HMM(n_components=options["Net"]["n_states"], \
        #                    covariance_type="full", tol=-np.inf, verbose=True)
        mdl.monitor_ = ConvgMonitor(mdl.tol, mdl.n_iter, mdl.verbose)

    else:
        # Load previous model (if model has already been created)
        mdl = pkl.load(open(out_mdl.replace("epoch" + epoch_str, "epoch" + str(int(epoch_str)-1)), "rb"))

    mdl.iepoch = epoch_str
    mdl.iclass = iclass_str
    
    print("epoch:{}\tclass:{}\t.".format(epoch_str, iclass_str), file=sys.stdout)
    
    # zero pad data for batch training
    # niter counts the number of em steps before saving a model checkpoint
    niter = options["Train"]["niter"]
    
    # add number of training data in model
    # mdl.number_training_data = len(xtrain)
    
    mdl.n_iter = niter # Add number of steps before saving model checkpoint

    # Model Training occurs here #
    mdl.fit(xtrain, lengths=l) # Calls the 'fit' function of the GMM class to run EM algorithm and find the parameters

    # Push back to cpu for compatibility when GPU unavailable.
    with open(out_mdl, "wb") as handle: # Prints out the final model with computed parameters in the .mdlc file in the specified location
        pkl.dump(mdl, handle)
    sys.exit(0)


