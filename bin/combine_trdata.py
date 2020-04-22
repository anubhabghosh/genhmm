# This function is used to combine together mutiple .pkl files for Multi-conditioned training
import numpy as np
import pickle as pkl
import os
import sys
import argparse

if __name__ == "__main__":
    
    usage = "Build one training set that would contain all noises and clean data as well.\n\"" \
        "Usage: python <location/filepath>/combine_trdata.py [n_features_used] [SNR levels] [noise types] [training data location]\n"\
        "Assuming all the training data are located in a proper folder called 'data' with a folder structure such as:\n"\
        "[training data location]/data/clean : contains clean training data stored as train.<num_features>.pkl\n"\
        "[training data location]/data/white : contains white noise corrupted training data stored as train.<num_features>.white.[SNR level]dB.pkl\n"\
        "[training data location]/data/babble : contains babble noise corrupted training data stored as train.<num_features>.[SNR level]dB.pkl, and so on\n"\
        "Expected to return a single .pkl file which contains a concatenation of all utterances called train.<num_features>_all.pkl\n"
    
    parser = argparse.ArgumentParser(description="Build one training set that would contain all noises and clean data as well.")
    parser.add_argument('-n_feats', metavar="<No. of features used in each training data file>", type=str)
    parser.add_argument('-snr_levels', metavar="<SNR levels used in dB, entered as a comma separated array without spaces>", type=str)
    parser.add_argument('-noise_types', metavar="<Noise types used (including option 'clean' for noise-free), entered as a comma separated array without spaces>", type=str)
    parser.add_argument('-tr_data_folderpath', metavar="<Folder location for the training data as per \"Usage\">", type=str)
    args = parser.parse_args()

    nfeats = args.n_feats # No. of features used
    SNR_levels = args.snr_levels.split(",") # No. of SNR levels (in dB) used
    noise_types = args.noise_types.split(",") # No. of different noise types used
    filepath = args.tr_data_folderpath # File location of the data
    
    if filepath[-1] != "/":
        filepath = filepath + "/"

    filenames = []
    for noise in noise_types:
        if noise == "clean":
            filenames.append(filepath + noise + "/" + "train." + nfeats + ".pkl")
        else:
            for snr in SNR_levels:
                filenames.append(filepath + noise + "/" + "train." + nfeats + "." + noise + "." + snr + "dB"+ ".pkl")
    
    #print(filenames)
    for file_ in filenames:
        assert(os.path.exists(file_) == True)
        print(file_)

    outfile = filepath + "MCT_Data_10dB_all_aliter/" + "train." + nfeats + "_all" + ".pkl"
    pickler_ = pkl.Pickler(open(outfile, "wb"))

    tr_DATA_all = []
    tr_keys_all = []
    tr_lengths_all = []
    tr_PHN_all = []
    for fname_dtrain in filenames:
        
        #tr_DATA, tr_keys, tr_lengths, tr_PHN = pkl.load(open(fname_dtrain, "rb"))
        unpickler = pkl.Unpickler(open(fname_dtrain, "rb"))
        tr_DATA, tr_keys, tr_lengths, tr_PHN = unpickler.load()
        tr_DATA_all += tr_DATA
        tr_keys_all += tr_keys
        tr_lengths_all += tr_lengths
        tr_PHN_all += tr_PHN
    
    pickler_.dump([tr_DATA_all, tr_keys_all, tr_lengths_all, tr_PHN_all])
    
    

    



