# A file to display the predictions by three separate models (.mdl files) on the same data
import sys
import os
from functools import partial
import pickle as pkl
from gm_hmm.src.genHMM_GLOW import GenHMMclassifier, load_model
import numpy as np
from parse import parse
from gm_hmm.src.utils import divide, acc_str, append_class, parse_, accuracy_fun, pad_data, TheDataset, get_freer_gpu
import torch
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt
from scipy.io import savemat
import json
import time

def set_model(mdl_file, model_type):

    if model_type == 'gaus':
        with open(mdl_file, "rb") as handle:
            mdl = pkl.load(handle)
        mdl.device = 'cpu'
        #f = lambda x: accuracy_fun(x, mdl=mdl)
    elif model_type == 'gen' or model_type == 'glow':
        mdl = load_model(mdl_file)
        if torch.cuda.is_available():
            if not options["Mul_gpu"]:
                # default case, only one gpu
                device = torch.device('cuda')
                mdl.device = device
                mdl.pushto(mdl.device)   
            else:
                for i in range(4):
                    try:
                        time.sleep(np.random.randint(10))
                        device = torch.device('cuda:{}'.format(int(get_freer_gpu()) ))
                        # print("Try to push to device: {}".format(device))
                        mdl.device = device
                        mdl.pushto(mdl.device)   
                        break
                    except:
                        # if push error (maybe memory overflow, try again)
                        # print("Push to device cuda:{} fail, try again ...")
                        continue
        else:
            mdl.device = 'cpu'
            mdl.pushto(mdl.device)

        # set model into eval mode
        mdl.eval()

    return mdl

def compute_voting_predictions(combined_mdls_hat, true_class=None):

    # Choosing the prediction that has maximum vote
    # NOTE: Needs to be fixed for the case when all three of them predict something differently
    # Simply choose the prediction that is selected by majority
    voted_mdl_class_hat = np.array(list(map(lambda x: np.argmax(np.bincount(x)), combined_mdls_hat)))
    return voted_mdl_class_hat
    
if __name__ == "__main__":
    
    usage = "Usage: python bin/compute_accuracy_voting.py [mdl file] [ training and testing data .pkl files separated by space]\n" \
            "Example: python bin/compute_accuracy_voting.py models/epoch1.mdl data/train.39.pkl data/test.39.pkl" \
            "NOTE: Relative paths for different kind of models required to be set within the function"\
            "NOTE: Assume default path from where this script is executed is in the glowHMM_* directory"

    if len(sys.argv) != 4 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage, file=sys.stdout)
        sys.exit(1)
    
    # Parse argument
    mdl_file = sys.argv[1]
    training_data_file = sys.argv[2]
    testing_data_file = sys.argv[3]

    normalize = True # Set row-wise normalizing flag to be true

    # Pathnames for different kind of models
    # Assume default path is in the glowHMM_clean directory

    gmm_mdl_path = "../../../gaus/39feats/gaussK20_clean_5classes/"
    nvp_mdl_path = "../genHMM_RealNVP_clean/"
    glow_mdl_path = "./"

    assert os.path.isfile(gmm_mdl_path + mdl_file) == True # GMM-HMM model file
    assert os.path.isfile(nvp_mdl_path + mdl_file) == True # NVP-HMM model file
    assert os.path.isfile(glow_mdl_path + mdl_file) == True # Glow-HMM model file

    gmm_mdl = os.path.join(gmm_mdl_path, mdl_file) # Getting the full file name for gmm
    nvp_mdl = os.path.join(nvp_mdl_path, mdl_file) # Getting the full file name or nvp
    glow_mdl = os.path.join(glow_mdl_path, mdl_file) # Getting the full file name for glow

    # Load the default set of parameters
    with open("default.json") as f_in:
        options = json.load(f_in)
    
    gmm_mdl_loaded = set_model(gmm_mdl, 'gaus')
    nvp_mdl_loaded = set_model(nvp_mdl, 'gen')
    glow_mdl_loaded = set_model(glow_mdl, 'glow')

    # Set the parameters for result computation
    nclasses = len(gmm_mdl_loaded.hmms)
    totclasses = 39
    batch_size_ = 128

    # Builds an array of string containing the train and test data sets for each class
    # size: nclass x 2 (train, test)
    te_data_files = np.array([append_class(testing_data_file, iclass+1)
                   for iclass in range(nclasses)])

    file1 = open("./log/metrics_class_all.log", "w+")

    for i in range(te_data_files.shape[0]):

        data_file = te_data_files[i] # Get the test data file

        # Load the data file
        try:
            X = pkl.load(open(data_file, "rb")) 
        except:
            print("File not found")
        
        #######################################################
        # Get the predicted class values for GMM-HMM
        #######################################################

        # Get the length of all the sequences
        l = [xx.shape[0] for xx in X]
        # zero pad data for batch training

        true_class = parse("{}_{}.pkl", os.path.basename(data_file))[1]
        gmm_mdl_out_list = [gmm_mdl_loaded.forward(x_i[:,1:]) for x_i in X]
        gmm_mdl_out = np.array(gmm_mdl_out_list).transpose()

        # the out here should be the shape: data_size * nclasses
        gmm_mdl_class_hat = np.argmax(gmm_mdl_out, axis=0) + 1
        istrue_gmm_mdl = gmm_mdl_class_hat == int(true_class)
        print("GMM-HMM -- ", data_file, "Done ...", "{}/{}".format(str(istrue_gmm_mdl.sum()), str(istrue_gmm_mdl.shape[0])))

        #######################################################
        # Get the predicted class values for NVP-HMM
        #######################################################

        # zero pad data for batch training
        max_len_ = max(l)
        x_padded = pad_data(X, max_len_)
        batchdata = DataLoader(dataset=TheDataset(x_padded,
                                                  lengths=l,
                                                  device=nvp_mdl_loaded.hmms[0].device),
                                                  batch_size=batch_size_, 
                                                  shuffle=True)
        
        nvp_mdl_out_list = [nvp_mdl_loaded.forward(x) for x in batchdata]
        nvp_mdl_out = torch.cat(nvp_mdl_out_list, dim=1)
        nvp_mdl_class_hat = torch.argmax(nvp_mdl_out, dim=0) + 1

        istrue_nvp_mdl = nvp_mdl_class_hat == int(true_class)
        print("NVP-HMM -- ", data_file, "Done ...", "{}/{}".format(str(istrue_nvp_mdl.sum().cpu().numpy()), str(istrue_nvp_mdl.shape[0])))

        #######################################################
        # Get the predicted class values for Glow-HMM
        #######################################################

        glow_mdl_out_list = [glow_mdl_loaded.forward(x) for x in batchdata]
        glow_mdl_out = torch.cat(glow_mdl_out_list, dim=1)
        glow_mdl_class_hat = torch.argmax(glow_mdl_out, dim=0) + 1

        istrue_glow_mdl = glow_mdl_class_hat == int(true_class)
        print("Glow-HMM -- ", data_file, "Done ...", "{}/{}".format(str(istrue_glow_mdl.sum().cpu().numpy()), str(istrue_glow_mdl.shape[0])))

        ##########################################################
        # Compute Additional Metrics to indicate:
        # 1. Percentage of corrects between two flow models
        # 2. Percentage of True positives among three models
        ##########################################################

        issimilar_nvp_glow = nvp_mdl_class_hat == glow_mdl_class_hat # Discover the number of same predictions
        issimilar_gmm_glow = gmm_mdl_class_hat == glow_mdl_class_hat # Discover the number of same predictions
        issimilar_gmm_nvp = nvp_mdl_class_hat == gmm_mdl_class_hat # Discover the number of same predictions
        
        combined_mdls_hat = np.concatenate((gmm_mdl_class_hat.reshape(-1, 1), nvp_mdl_class_hat.reshape(-1, 1), glow_mdl_class_hat.reshape(-1, 1)), axis=1)
        voted_mdl_class_hat = compute_voting_predictions(combined_mdls_hat)
        istrue_voted_mdl = voted_mdl_class_hat == int(true_class)
        
        file1.write("Class:{}\tAcc_GMM:{:.3f}\tAcc_NVP:{:.3f}\tAcc_Glow:{:.3f}\tAcc_Voted:{:.3f}\n".format(i+1, 
                                                                                                       istrue_gmm_mdl.sum()/istrue_gmm_mdl.shape[0],
                                                                                                       istrue_nvp_mdl.sum().cpu().numpy()/istrue_nvp_mdl.shape[0],
                                                                                                       istrue_glow_mdl.sum().cpu().numpy()/istrue_glow_mdl.shape[0],
                                                                                                       istrue_voted_mdl.sum().cpu().numpy()/istrue_voted_mdl.shape[0]))

        file1.write("Sim_NVP-Glow:{:.3f}\tSim_GMM-Glow:{:.3f}\tSim_GMM-NVP:{:.3f}\n".format(issimilar_nvp_glow.sum().cpu().numpy()/issimilar_nvp_glow.shape[0],
                                                                       issimilar_gmm_glow.sum().cpu().numpy()/issimilar_gmm_glow.shape[0],
                                                                       issimilar_gmm_nvp.sum().cpu().numpy()/issimilar_gmm_nvp.shape[0]))

    file1.close() # Closing the file after contents have been written
    sys.exit(0)