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
import pandas as pd
from scipy import stats

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
    #voted_mdl_class_hat = np.array(list(map(lambda x: np.argmax(np.bincount(x)), combined_mdls_hat)))
    modes, counts = stats.mode(combined_mdls_hat, axis=1)
    voted_mdl_class_hat = np.array([modes[i] if counts[i] > 1 else np.random.choice(combined_mdls_hat[i,:]) for i in range(combined_mdls_hat.shape[0])])
    #voted_mdl_class_hat = np.array([modes[i] if counts[i] > 1 else combined_mdls_hat[i,1] for i in range(combined_mdls_hat.shape[0])]) # Use NVP prediction in case there is no consensus

    Count3_true, Count3_false = check_pred_similarity(modes, counts, 3, true_class)
    Count2_true, Count2_false = check_pred_similarity(modes, counts, 2, true_class)
    Count1_true, Count1_false = check_pred_similarity(modes, counts, 1, true_class)

    Count_dict = {}
    Count_dict[1] = np.array([Count1_true, Count1_false]) / np.float(len(counts))
    Count_dict[2] = np.array([Count2_true, Count2_false]) / np.float(len(counts))
    Count_dict[3] = np.array([Count3_true, Count3_false]) / np.float(len(counts))

    return voted_mdl_class_hat, Count_dict

def compute_sample_complexity(datafiles):

    # Obtain lengths of the individual class data files and calculate the one with the highest number of samples
    lengths = []
    for i in range(len(datafiles)):
        datafile = pkl.load(open(datafiles[i], "rb"))
        lengths.append(datafile.shape[0])
    
    s_complexity = np.array(lengths) / max(lengths) 
    return s_complexity 

def check_pred_similarity(modes, counts, N, true_class):
    
    assert true_class is not None
    where_countN = np.where(counts == N)[0] # find indices where count is 2
    CountN_true = (modes[where_countN] == true_class).sum()
    CountN_false = len(modes[where_countN]) - CountN_true
    return CountN_true, CountN_false

def get_phoneme_type(phn, phonemes_type_dict):
    for item in phonemes_type_dict.items():
        if phn in item[1]:
            return item[0]

def create_phoneme_type_dict():

    # P - Plosives, F - Fricatives, N - Nasals, SV - Semi-Vowels, D - Dipthongs, C - Closures
    phonemes_type_dict = {'P':['b','d','g','p','t','k','jh','ch'],
                      'F':['s','sh','z','f','th','v','dh','hh'],
                      'N':['m','n','ng'],
                      'SV':['l','r','er','w','y'],
                      'V':['iy','ih','eh','ae','aa','ah','uh','uw'],
                      'D':['ey','aw','ay','oy','ow'],
                      'C':['sil','dx']}

    return phonemes_type_dict

def get_test_datafile(base_testing_data_file, noise_type, SNR_level):

    datafolder, filename = parse("{}/{}", base_testing_data_file)
    test_mode, n_feats, extension = parse("{}.{}.{}", filename)
    new_testing_data_file = datafolder + "/" + test_mode + "." + n_feats + \
        "." + noise_type + "." + SNR_level + "dB" + "." + extension
    return new_testing_data_file

if __name__ == "__main__":
    
    usage = "Usage: python bin/compute_accuracy_voting.py [mdl file] [ training and testing data .pkl files separated by space] [testing type as a string (test/train)]\n" \
            "Example: python bin/compute_accuracy_voting_noise.py models/epoch1.mdl data/train.39.pkl data/test.39.pkl [test/train] [{pink/white/babble/hfchannel}.{SNR value in dB}dB]\n" \
            "NOTE: Relative paths for different kind of models required to be set within the function\n" \
            "NOTE: Assume default path from where this script is executed is in the glowHMM_* directory\n" 
            
    if len(sys.argv) != 6 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage, file=sys.stdout)
        sys.exit(1)
    
    # Parse arguments
    mdl_file = sys.argv[1]
    training_data_file = sys.argv[2]
    base_testing_data_file = sys.argv[3]
    testing_type = sys.argv[4]
    noise_type = sys.argv[5]
    #SNR_level = sys.argv[6]
    noise_type, SNR_level = parse("{}.{}dB", sys.argv[5])

    testing_data_file = get_test_datafile(base_testing_data_file, noise_type, SNR_level)

    normalize = True # Set row-wise normalizing flag to be true

    # Pathnames for different kind of models
    # Assume default path is in the glowHMM_clean directory

    gmm_mdl_path = "../../../gaus/39feats/gmmHMM_clean/"
    nvp_mdl_path = "../nvpHMM_clean/"
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
    
    classmap_file = "./data/class_map.json" # As the same classmap is planned to used for all the classes-
    
    # load the classmap dict
    with open(classmap_file) as f:
        classmap = json.load(f)

    gmm_mdl_loaded = set_model(gmm_mdl, 'gaus')
    nvp_mdl_loaded = set_model(nvp_mdl, 'gen')
    glow_mdl_loaded = set_model(glow_mdl, 'glow')

    # Set the parameters for result computation
    nclasses = len(gmm_mdl_loaded.hmms)
    totclasses = 39
    batch_size_ = 128

    # Get the phoneme type mapping dictionary
    phonemes_type_dict = create_phoneme_type_dict()

    # Builds an array of string containing the train and test data sets for each class
    # size: nclass x 2 (train, test)
    te_data_files = np.array([append_class(testing_data_file, iclass+1)
                   for iclass in range(nclasses)])
    
    tr_data_files = np.array([append_class(training_data_file, iclass+1)
                   for iclass in range(nclasses)])

    tr_sample_complexity = compute_sample_complexity(tr_data_files) # vector containing values representing sample compleixty for training data
    #te_sample_complexity = compute_sample_complexity(te_data_files) # vector containing values representing sample complexity for test data

    file1 = open("./log/metrics_class_all_noisytest_{}.{}dB.log".format(noise_type, SNR_level), "w+") # Opening the file
    df_filename = "./log/metrics_class_all_noisytest_{}.{}dB.json".format(noise_type, SNR_level) # Define an excel file name for storing log results using dataframes
    correct_gmm = 0
    correct_nvp = 0
    correct_glow = 0
    correct_voted = 0
    total_samples = 0

    metrics = [] # Create a blank list to append lists for creating a dataframe
    metrics_columns = ['Phoneme', 'Type', 'C_train', 'Acc_GMM', 'Acc_NVP', 'Acc_Glow', 'Acc_Voting', 
                       'N_Agreed_3_True', 'N_Agreed_3_False', 'N_Agreed_2_True', 'N_Agreed_2_False', 
                       'N_Agreed_1_True', 'N_Agreed_1_False'] # Predefine the dataframe column headers 

    for i in range(te_data_files.shape[0]):

        te_data_file = te_data_files[i] # Get the test data file
        tr_data_file = tr_data_files[i] # Get the testing data file

        # Load the data file
        try:
            X_test = pkl.load(open(te_data_file, "rb")) 
            X_train = pkl.load(open(tr_data_file,"rb"))
        except:
            print("File not found")
        
        #######################################################
        # Get the predicted class values for GMM-HMM
        #######################################################

        # Get the sample complexity (as a ratio of no. of samples in the given training class file 
        # and the maximum number of samples in any given class file)
        C_train = tr_sample_complexity[i] 

        # Get the class phoneme for the given class number
        iclass_phn = classmap[str(i)]
        
        # Get the phoneme type
        iclass_phn_type = get_phoneme_type(iclass_phn, phonemes_type_dict)
        if iclass_phn_type == None:
            iclass_phn_type = '<UNK>' 

        # Choose the data file based on the training type
        if testing_type.lower() == "train":
            X = X_train
            data_file = tr_data_file
        elif testing_type.lower() == "test":
            X = X_test
            data_file = te_data_file
        else:
            print("Invalid testing argument !!!")
            sys.exit(1)

        # Get the length of all the sequences in the given class file
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
        issimilar_gmm_glow = gmm_mdl_class_hat == glow_mdl_class_hat.cpu().numpy() # Discover the number of same predictions
        issimilar_gmm_nvp = gmm_mdl_class_hat == nvp_mdl_class_hat.cpu().numpy() # Discover the number of same predictions
        
        combined_mdls_hat = np.concatenate((gmm_mdl_class_hat.reshape(-1, 1), nvp_mdl_class_hat.cpu().numpy().reshape(-1, 1), glow_mdl_class_hat.cpu().numpy().reshape(-1, 1)), axis=1)
        #combined_mdls_hat = np.concatenate((nvp_mdl_class_hat.cpu().numpy().reshape(-1, 1), glow_mdl_class_hat.cpu().numpy().reshape(-1, 1)), axis=1)
        
        # Compute the voting predictions
        voted_mdl_class_hat, Count_dict = compute_voting_predictions(combined_mdls_hat, int(true_class))
        istrue_voted_mdl = voted_mdl_class_hat == int(true_class)
        
        # Get additional statistics based on count
        counts_checked = list(Count_dict.keys())

        #print("NVP selections for True class:{} is :\n".format(int(true_class)))
        #print(nvp_mdl_class_hat)
        #print("Glow selections for True class:{} is :\n".format(int(true_class)))
        #print(glow_mdl_class_hat)
        #print("Voted selections for True class:{} is \n".format(int(true_class)))
        #print(voted_mdl_class_hat)

        print("Voted-HMM --", data_file, "Done ...", "{}/{}".format(str(istrue_voted_mdl.sum()), str(istrue_voted_mdl.shape[0])))
        
        file1.write("Class:{} ({})\n\n No. of samples: Train-{}, Test-{}\n".format(i+1, iclass_phn, X_train.shape[0], X_test.shape[0]))
        file1.write("Acc_GMM:{:.3f}\tAcc_NVP:{:.3f}\tAcc_Glow:{:.3f}\tAcc_Voted:{:.3f}\n".format(istrue_gmm_mdl.sum()/istrue_gmm_mdl.shape[0],
                                                                                                 istrue_nvp_mdl.sum().cpu().numpy()/istrue_nvp_mdl.shape[0],
                                                                                                 istrue_glow_mdl.sum().cpu().numpy()/istrue_glow_mdl.shape[0],
                                                                                                 istrue_voted_mdl.sum()/istrue_voted_mdl.shape[0]))

        file1.write("C_train:{:.3f}\tCount3_true:{:.3f}\tCount3_false:{:.3f}\tCount2_true:{:.3f}\tCount2_false:{:.3f}\tCount1_true:{:.3f}\tCount1_false:{:.3f}\n".format(C_train, Count_dict[3][0], Count_dict[3][1], Count_dict[2][0], Count_dict[2][1], Count_dict[1][0], Count_dict[1][1]))

        file1.write("Sim_NVP-Glow:{:.3f}\tSim_GMM-Glow:{:.3f}\tSim_GMM-NVP:{:.3f}\n\n".format(issimilar_nvp_glow.sum().cpu().numpy()/issimilar_nvp_glow.shape[0],
                                                                       issimilar_gmm_glow.sum()/issimilar_gmm_glow.shape[0],
                                                                       issimilar_gmm_nvp.sum()/issimilar_gmm_nvp.shape[0]))
        file1.write("------------------------------------------------\n")

        # Append the metrics to the list 'metrics' that is latern to be converted into a dataframe
        metric = [iclass_phn, iclass_phn_type, C_train, 
                  istrue_gmm_mdl.sum()/istrue_gmm_mdl.shape[0], 
                  istrue_nvp_mdl.sum().cpu().numpy()/istrue_nvp_mdl.shape[0], 
                  istrue_glow_mdl.sum().cpu().numpy()/istrue_glow_mdl.shape[0],
                  istrue_voted_mdl.sum()/istrue_voted_mdl.shape[0],
                  Count_dict[3][0], Count_dict[3][1], Count_dict[2][0], 
                  Count_dict[2][1], Count_dict[1][0], Count_dict[1][1]]

        metrics.append(metric)

        correct_gmm += istrue_gmm_mdl.sum()
        correct_nvp += istrue_nvp_mdl.sum().cpu().numpy()
        correct_glow += istrue_glow_mdl.sum().cpu().numpy()
        correct_voted += istrue_voted_mdl.sum()
        total_samples += istrue_gmm_mdl.shape[0]

    file1.write("------------------------------------------------\n")
    file1.write("------------------------------------------------\n")

    file1.write("GMM-HMM -- Total Acc {:d}/{:d} = {}\n".format(correct_gmm, total_samples, correct_gmm/total_samples))
    file1.write("NVP-HMM -- Total Acc {:d}/{:d} = {}\n".format(correct_nvp, total_samples, correct_nvp/total_samples))
    file1.write("Glow-HMM -- Total Acc {:d}/{:d} = {}\n".format(correct_glow, total_samples, correct_glow/total_samples))
    file1.write("Voted-HMM -- Total Acc {:d}/{:d} = {}\n".format(correct_voted, total_samples, correct_voted/total_samples))

    # Convert the collected lists into a dataframe for easy usage
    df_metrics = pd.DataFrame(metrics, columns=metrics_columns)
    print(df_metrics)
    df_metrics.to_json(df_filename, orient='split')

    file1.close() # Closing the file after contents have been written
    sys.exit(0)
