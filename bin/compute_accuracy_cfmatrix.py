# A file to display the confusion matrix 
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

def set_model(mdl_file, model_type):
    """ This function takes a given model file (with full pathname) as input and returns a 
    variable that contains the loaded model. The model is loaded into the CPU or the GPU depending
    on the type of the model. GMM-HMM model is loaded on the CPU, whereas Flow-based HMM models like
    NVP-HMM and Glow-HMM have to be loaded on the GPU
    ----
    Args:
    - mdl_file :  model file (.mdl extension) with full pathname
    - model_type : string that indicates the type of model: "gaus" - GMM, "gen" - NVP, "glow" -  Glow

    Returns:
    - mdl: Variable containing loaded model into the appropriate device

    """
    if model_type == 'gaus':
        # Loading the GMM-HMM model on the CPU
        with open(mdl_file, "rb") as handle:
            mdl = pkl.load(handle)
        mdl.device = 'cpu'
    
    elif model_type == 'gen' or model_type == 'glow':
        # In case the model is NVP-HMM ("gen") or Glow-HMM ("glow")
        mdl = load_model(mdl_file)
        if torch.cuda.is_available():
            if not options["Mul_gpu"]:
                # default case, only one gpu
                device = torch.device('cuda')
                mdl.device = device
                mdl.pushto(mdl.device)   
            else:
                # In case Mul_gpu option is set, the model is pushed to mutiple available GPU cores
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
            # In case no GPU device is available, the model is pushed to CPU
            mdl.device = 'cpu'
            mdl.pushto(mdl.device)

        # Set model into eval mode so that no parameters are learned during "Testing" phase
        mdl.eval()

    # Returns the loaded model file
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
    """This function gets the corresponding phoneme type
    from the dictionary of "PhonemeType : Phonemes"
    ---
    Args:
    - phn : Phoneme provided as a string
    - phonemes_type_dict: PhonemeType-Phoneme Python dictionary
    
    Returns:
    - 'string' indicating type of phoneme 
    """
    for item in phonemes_type_dict.items():
        if phn in item[1]:
            return item[0]

def create_phoneme_type_dict():
    """This function is used to create a dictionary of phoneme-type to phoneme mapping
    Taken from "Phoneme Recognition on the TIMIT Database"
    https://www.intechopen.com/books/speech-technologies/phoneme-recognition-on-the-timit-database 
    NOTE: The set used here is the folded set of 39 phonemes from a set of 61 phonemes
    """
    # P - Plosives, F - Fricatives, N - Nasals, SV - Semi-Vowels, D - Dipthongs, C - Closures
    phonemes_type_dict = {'P':['b','d','g','p','t','k','jh','ch'],
                      'F':['s','sh','z','f','th','v','dh','hh'],
                      'N':['m','n','ng'],
                      'SV':['l','r','er','w','y'],
                      'V':['iy','ih','eh','ae','aa','ah','uh','uw'],
                      'D':['ey','aw','ay','oy','ow'],
                      'C':['sil','dx']}

    return phonemes_type_dict

def compute_cfmetrics(cf_matrix, sample_complexity, classmap, phonemes_type_dict, model_name):
    """This function is used for computing essential metrics on a per class basis and also find 
    a weighted average of precision, recall and F1 scores
    """
    #####################################################################################
    # Compute Additional helpful metrics
    ####################################################################################

    cf_metrics = np.zeros((nclasses, 4))
    file1 = open("./log/cfmetrics_{}.log".format(model_name), "w+")
    #accuracy_avg = 0
    precision_avg = 0
    recall_avg = 0
    F1_avg = 0
    metrics = [] # Create a blank list to append lists for creating a dataframe
    metrics_columns = ['Phoneme', 'Type', 'Precision', 'Recall', 'F1_score'] # Predefine the dataframe column headers

    for i in range(cf_matrix.shape[0]):

        # Get the class phoneme for the given class number
        iclass_phn = classmap[str(i)]
        
        # Get the phoneme type
        iclass_phn_type = get_phoneme_type(iclass_phn, phonemes_type_dict)
        if iclass_phn_type == None:
            iclass_phn_type = '<UNK>'

        tp_iclass = cf_matrix[i,i]  # Get number of true positives
        fn_iclass = np.sum(cf_matrix[i,:]) - tp_iclass # Get number of false negatives
        fp_iclass = np.sum(cf_matrix[:,i]) - tp_iclass # Get number of false positives
        
        precision_iclass = tp_iclass / (tp_iclass + fp_iclass) # Get the precision
        recall_iclass = tp_iclass / (tp_iclass + fn_iclass) # Get the recall
        #acc_iclass = tp_iclass / np.sum(cf_matrix[i,:]) # Get the accuracy
        if precision_iclass != 0 or recall_iclass != 0:
            F1_iclass = 2 * precision_iclass * recall_iclass / (precision_iclass + recall_iclass)
        else:
            F1_iclass = 0 # In case of undefined values

        metrics.append([iclass_phn, iclass_phn_type, precision_iclass, recall_iclass, F1_iclass])
        #print("Class:{}\tAccuracy:{}\tPrecision:{}\tRecall:{}\tF1_score:{}".format(i+1, acc_iclass, precision_iclass, recall_iclass, F1_iclass)) 
        file1.write("Class:{}\t Phn:{} \t Phn_type:{} \t Precision:{:.3f}\t\tRecall:{:.3f}\tF1_score:{:.3f}\n".format(i+1, iclass_phn, 
                                                                                                                iclass_phn_type, 
                                                                                                                precision_iclass, 
                                                                                                                recall_iclass, F1_iclass))
        
        #accuracy_avg += sample_complexity[i] * acc_iclass
        precision_avg += sample_complexity[i] * precision_iclass
        recall_avg += sample_complexity[i] * recall_iclass
        F1_avg += sample_complexity[i] * F1_iclass

    file1.write("-----------------------------------------------------------------------------------------------------------------------------------\n")
    #file1.write("Weighted Accuracy for {}-hmm model:{:.3f} \n".format(model_name, accuracy_avg))
    file1.write("Weighted Precision for {}-hmm model:{:.3f} \n".format(model_name, precision_avg / np.sum(sample_complexity)))
    file1.write("Weighted Recall for {}-hmm model:{:.3f} \n".format(model_name, recall_avg / np.sum(sample_complexity)))
    file1.write("Weighted F1 for {}-hmm model:{:.3f} \n".format(model_name, F1_avg / np.sum(sample_complexity)))
    
    df_cfmetrics = pd.DataFrame(metrics, columns=metrics_columns)
    df_cfmetrics.to_json("./log/cfmetrics_{}.json".format(model_name), orient='split')
    file1.close() 

    #if normalize == True: # "True" Row-wise normalization
    #    cf_matrix = cf_matrix / cf_matrix.sum(axis=1, keepdims=True)

    #print("Confusion Matrix for Class :{} is {} ".format(true_class, cf_matrix))
    
    cf_dict = {}
    cf_dict['cfmatrix'] = cf_matrix
    cf_dict['normalize'] = False # Change this to True is normalization is indeed done
    #cf_dict['normalize'] = 'true_row_wise'
    cf_dict['numclasses'] = nclasses

    return cf_dict
    
if __name__ == "__main__":
    
    usage = "Usage: python bin/compute_accuracy_cfmatrix.py [mdl file] [ training and testing data .pkl files] [testing type denoted as a string (test/train)] \n" \
            "Example: python bin/compute_accuracy_cfmatrix.py models/epoch1.mdl data/train.39.pkl data/test.39.pkl [test/train]" \
            "NOTE: Relative paths for different kinds of models required to be set within the function \n" \
            "NOTE: For now the relative path is set w.r.t. the path to the execution of the script in the glowHMM_* directory\n" 

    # In case the number of arguments are not equal to the desired number, it will display 
    # usage of the function and how to call the function properly. 
    if len(sys.argv) != 5 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage, file=sys.stdout)
        sys.exit(1)

    # NOTE: An important assumption that is made here is that the name of the model file is the same 
    # for all the type of hmm models used, i.e. it is always assumed model file is named as models/epoch1.mdl

    # Parse arguments
    mdl_file = sys.argv[1] # Path for the model file
    training_data_file = sys.argv[2] # Path for the training data file
    testing_data_file = sys.argv[3] # Path for the testing data file
    dataset_type = sys.argv[4] # Type of dataset to be used for metric computation: Train/Test

    normalize = True # set row-wise normalizing flag to be true

    # Insert pathnames for different kind of models here
    # Assume the default path is in the 'glowHMM_clean' directory
    gmm_mdl_path = "../../../gaus/39feats/gmmHMM_clean/"
    nvp_mdl_path = "../nvpHMM_clean/"
    glow_mdl_path = "./"

    assert os.path.isfile(gmm_mdl_path + mdl_file) == True # GMM-HMM model file
    assert os.path.isfile(nvp_mdl_path + mdl_file) == True # NVP-HMM model file
    assert os.path.isfile(glow_mdl_path + mdl_file) == True # Glow-HMM model file

    gmm_mdl = os.path.join(gmm_mdl_path, mdl_file) # Getting the full file name for gmm
    nvp_mdl = os.path.join(nvp_mdl_path, mdl_file) # Getting the full file name or nvp
    glow_mdl = os.path.join(glow_mdl_path, mdl_file) # Getting the full file name for glow

    # Load the default set of parameters from the dictionary that will define the testing parameters
    with open("default.json") as f_in:
        options = json.load(f_in)
    
    # load the classmap dictionary which contains the TRUE labels i.e. Phonemes for every Class Number.
    classmap_file = "./data/class_map.json" # As the same classmap is planned to used for all the classes
    with open(classmap_file) as f:
        classmap = json.load(f)

    # Load Models 
    gmm_mdl_loaded = set_model(gmm_mdl, 'gaus')
    nvp_mdl_loaded = set_model(nvp_mdl, 'gen')
    glow_mdl_loaded = set_model(glow_mdl, 'glow')

    # Set the parameters for result computation
    nclasses = len(gmm_mdl_loaded.hmms)
    nclasses_arr = [int(c+1) for c in range(nclasses)] # List with class indexes used for computing Confusion matrices
    totclasses = 39
    batch_size_ = 128
    cf_matrix_gmm = np.zeros((nclasses, nclasses))
    cf_matrix_nvp = np.zeros((nclasses, nclasses))
    cf_matrix_glow = np.zeros((nclasses, nclasses))
    cf_matrix_voted = np.zeros((nclasses, nclasses))
    
    # Get the phoneme type mapping dictionary
    phonemes_type_dict = create_phoneme_type_dict()

    # Builds an array of string containing the train and test data sets for each class
    # size: nclass x 2 (train, test)
    #data_files = np.array([[append_class(training_data_file, iclass+1), append_class(testing_data_file, iclass+1)]
    #               for iclass in range(nclasses)])
    tr_data_files = np.array([append_class(training_data_file, iclass+1)
                   for iclass in range(nclasses)])
    te_data_files = np.array([append_class(testing_data_file, iclass+1)
                   for iclass in range(nclasses)])

    # Vector containing values representing sample compleixty for training data
    tr_sample_complexity = compute_sample_complexity(tr_data_files) 
    te_sample_complexity = compute_sample_complexity(te_data_files)

    # Define a function for this particular HMMclassifier model
    #f = partial(accuracy_fun, mdl=mdl)
    #out = [[f(data_files[i, j]) for j in range(data_files.shape[1])] for i in range(data_files.shape[0])]
    #results = np.array(out)
    #print_results(mdl_file, data_files, results)
    
    #file1 = open("./log/metrics_class_accuracy_test.log", "w+") # Opening the file
    #df_filename = "./log/metrics_class_accuracy_test.json" # Define an excel file name for storing log results using dataframes
    
    ###########################################################################
    # Obtain confusion matrix for test files
    ###########################################################################
    
    for i in range(te_data_files.shape[0]):
        
        te_data_file = te_data_files[i] # Get the test data file path
        tr_data_file = tr_data_files[i] # Get the training data file path

        # Load the appropriate data file
        try:
            #X_test = pkl.load(open(te_data_file, "rb")) 
            #X_train = pkl.load(open(tr_data_file,"rb"))
            if dataset_type.lower() == "test":
                X = pkl.load(open(te_data_file, "rb"))
                data_file = te_data_file
                sample_complexity = te_sample_complexity
            elif dataset_type.upper() == "train":
                X = pkl.load(open(tr_data_file, "rb"))
                data_file = tr_data_file
                sample_complexity = tr_sample_complexity
            else:
                print("Invalid testing argument!!!")
                sys.exit(1) # Error code for unsuccessful operation
        except:
            print("File not found")

        # Get the sample complexity (as a ratio of no. of samples in the given training class file 
        # and the maximum number of samples in any given class file)
        C_train = tr_sample_complexity[i] 

        # Get the class phoneme for the given class number
        iclass_phn = classmap[str(i)]
        
        # Get the phoneme type
        iclass_phn_type = get_phoneme_type(iclass_phn, phonemes_type_dict)
        if iclass_phn_type == None:
            iclass_phn_type = '<UNK>'
    
        #######################################################################
        # Get the predictions class values for GMM-HMM
        #######################################################################
        
        # Get the length of all the sequences
        l = [xx.shape[0] for xx in X]
        
        # Zero pad data for batch training
        true_class = parse("{}_{}.pkl", os.path.basename(data_file))[1]
        gmm_mdl_out_list = [gmm_mdl_loaded.forward(x_i[:,1:]) for x_i in X]
        gmm_mdl_out = np.array(gmm_mdl_out_list).transpose()
        # The out here should be the shape: data_size * nclasses
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

        #issimilar_nvp_glow = nvp_mdl_class_hat == glow_mdl_class_hat # Discover the number of same predictions
        #issimilar_gmm_glow = gmm_mdl_class_hat == glow_mdl_class_hat.cpu().numpy() # Discover the number of same predictions
        #issimilar_gmm_nvp = gmm_mdl_class_hat == nvp_mdl_class_hat.cpu().numpy() # Discover the number of same predictions
        
        combined_mdls_hat = np.concatenate((gmm_mdl_class_hat.reshape(-1, 1), nvp_mdl_class_hat.cpu().numpy().reshape(-1, 1), glow_mdl_class_hat.cpu().numpy().reshape(-1, 1)), axis=1)
        #combined_mdls_hat = np.concatenate((nvp_mdl_class_hat.cpu().numpy().reshape(-1, 1), glow_mdl_class_hat.cpu().numpy().reshape(-1, 1)), axis=1)
        
        # Compute the voting predictions
        voted_mdl_class_hat, Count_dict = compute_voting_predictions(combined_mdls_hat, int(true_class))
        istrue_voted_mdl = voted_mdl_class_hat == int(true_class)
        
        # Get additional statistics based on count
        # counts_checked = list(Count_dict.keys())

        print("Voted-HMM --", data_file, "Done ...", "{}/{}".format(str(istrue_voted_mdl.sum()), str(istrue_voted_mdl.shape[0])))
        
        #file1.write("Class:{} ({})\n\n No. of samples: Train-{}, Test-{}\n".format(i+1, iclass_phn, X_train.shape[0], X_test.shape[0]))
        #file1.write("Acc_GMM:{:.3f}\tAcc_NVP:{:.3f}\tAcc_Glow:{:.3f}\tAcc_Voted:{:.3f}\n".format(istrue_gmm_mdl.sum()/istrue_gmm_mdl.shape[0],
        #                                                                                         istrue_nvp_mdl.sum().cpu().numpy()/istrue_nvp_mdl.shape[0],
        #                                                                                         istrue_glow_mdl.sum().cpu().numpy()/istrue_glow_mdl.shape[0],
        #                                                                                         istrue_voted_mdl.sum()/istrue_voted_mdl.shape[0]))

        ######################################################################################
        # Confusion Matrix calculation for each of the models: GMM-HMM, NVP-HMM and Glow-HMM
        ######################################################################################
        
        for c in nclasses_arr:

            istrue_gmm_mdl_c = gmm_mdl_class_hat == c
            istrue_nvp_mdl_c = nvp_mdl_class_hat == c
            istrue_glow_mdl_c = glow_mdl_class_hat == c
            istrue_voted_mdl_c = voted_mdl_class_hat == c

            cf_matrix_gmm[int(true_class) - 1, c-1] = istrue_gmm_mdl_c.sum()
            cf_matrix_nvp[int(true_class) - 1, c-1] = istrue_nvp_mdl_c.sum()
            cf_matrix_glow[int(true_class) - 1, c-1] = istrue_glow_mdl_c.sum()
            cf_matrix_voted[int(true_class) - 1, c-1] = istrue_voted_mdl_c.sum()

    # Saving the confusion matrix
    gmm_cf_dict = compute_cfmetrics(cf_matrix_gmm, sample_complexity, classmap, phonemes_type_dict, "gmm")
    savemat("./log/gmm_cfmatrix.mat", gmm_cf_dict)

    nvp_cf_dict = compute_cfmetrics(cf_matrix_nvp, sample_complexity, classmap, phonemes_type_dict, "nvp")
    savemat("./log/nvp_cfmatrix.mat", nvp_cf_dict)

    glow_cf_dict = compute_cfmetrics(cf_matrix_glow, sample_complexity, classmap, phonemes_type_dict, "glow")
    savemat("./log/glow_cfmatrix.mat", glow_cf_dict)

    voted_cf_dict = compute_cfmetrics(cf_matrix_voted, sample_complexity, classmap, phonemes_type_dict, "voted")
    savemat("./log/voted_cfmatrix.mat", voted_cf_dict)

    # load the classmap dict
    #classmap_file = "./data/class_map.json"
    #with open(classmap_file) as f:
    #    classmap = json.load(f)
    # Create labels for confusion matrix
    #x_labels = [classmap[str(x1)] for x1 in range(cf_matrix.shape[1])]
    #y_labels = [classmap[str(y1)] for y1 in range(cf_matrix.shape[0])]

    #fig, ax = plt.subplots()
    #plt.title(" Confusion Matrix for {}/{} classes (Row:True,Col:Pred)".format(nclasses, totclasses))
    #cf = ax.imshow(cf_matrix, aspect='auto', cmap='jet')
    #ax.set_xticks([x1 for x1 in range(cf_matrix.shape[1])])
    #ax.set_yticks([x1 for x1 in range(cf_matrix.shape[1])])
    #ax.set_yticklabels(y_labels)
    #ax.set_xticklabels(x_labels)
    #ax.set_xlabel('Predicted classes')
    #ax.set_ylabel('True classes')
    #fig.colorbar(cf, ax=ax)
    #plt.show()
    #plt.savefig("./models/" + "cfmatrix_truenorm_full.pdf")
    #plt.close()
    
    sys.exit(0)
