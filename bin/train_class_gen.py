import os
import sys
from parse import parse
import pickle as pkl
from gm_hmm.src.genHMM import GenHMM, save_model, load_model, ConvgMonitor
#from gm_hmm.src.genHMM_GLOW import GenHMM, save_model, load_model, ConvgMonitor
from gm_hmm.src.utils import pad_data, TheDataset, get_freer_gpu
import torch
from torch.utils.data import DataLoader
import json
import numpy as np
import time

if __name__ == "__main__":
    usage = "python bin/train_class_gen.py data/train.39.pkl models/epoch1_class1.mdlc default.json"
    if len(sys.argv) < 3 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage)
        sys.exit(1)

    # Set random seeds for reproducibility
    # np.random.seed(2)
    # torch.manual_seed(2)
    
    # Parse
    train_inputfile = sys.argv[1]
    out_mdl = sys.argv[2]

    # Test for a third parameter
    try:
        param_file = sys.argv[3]
    except IndexError:
        param_file = "default.json"

    #network_type = "GLOW_Net"
    network_type = "Net"

    epoch_str, iclass_str = parse('epoch{}_class{}.mdlc',os.path.basename(out_mdl))
    train_class_inputfile = train_inputfile.replace(".pkl", "_{}.pkl".format(iclass_str))

    #  Load data
    
    xtrain = pkl.load(open(train_class_inputfile, "rb"))
    #xtrain = xtrain[:100]
    # Get the length of all the sequences
    l = [x.shape[0] for x in xtrain]

    # load the parameters
    with open(param_file) as f_in:
        options = json.load(f_in)

    # adoptive to set number of states
    #options["Net"]["n_states"] = np.clip(int(np.floor(np.mean(l)/2)),
    #                                     options["Train"]["n_states_min"],
    #                                     options["Train"]["n_states_max"])
    options[network_type]["n_states"] = np.clip(int(np.floor(np.mean(l)/2)),
                                         options["Train"]["n_states_min"],
                                         options["Train"]["n_states_max"])

    # Convergence monitoring parameters
    tol = 1e-2 # Convg. Monitor tolerance
    verbose = True # Verbose flag is True
    ncon_int = 4 # No. of consecutive iterations to be checked

    # niter counts the number of em steps before saving a model checkpoint
    niter = options["Train"]["niter"]
    
    if iclass_str == '39': # Special case for the <sil> class
        tol = 1e-2
        ncon_int = 2
    '''
    # NOTE: Poor fix for <sil> class and some other problematic classes 
    if iclass_str == '39':
        tol = 3e-2
        ncon_int = 2
    elif iclass_str == '6' or iclass_str == '16' or iclass_str == '26' or iclass_str == '30' or iclass_str == '38':
        tol = 2.5e-2
        ncon_int = 3
    '''
    # niter counts the number of em steps before saving a model checkpoint
    niter = options["Train"]["niter"]
    
    #  Load or create model
    if epoch_str == '1':
    #    mdl = GenHMM(**options["Net"])
        mdl = GenHMM(**options[network_type])

        # Inserting variable convergence control
        mdl.monitor_ = ConvgMonitor(tol, niter, verbose)

    else:
        # Load previous model
        mdl = load_model(out_mdl.replace("epoch" + epoch_str, "epoch" + str(int(epoch_str)-1)))

    mdl.iepoch = epoch_str
    mdl.iclass = iclass_str


    mdl.device = 'cpu'
    if torch.cuda.is_available():
        if not options["Mul_gpu"]:
            # default case, only one gpu
            device = torch.device('cuda')
            mdl.device = device
            mdl.pushto(mdl.device)   

        else:
            for i in range(4):
                try:
                    time.sleep(np.random.randint(20))
                    device = torch.device('cuda:{}'.format(int(get_freer_gpu()) ))
                    print("Try to push to device: {}".format(device))
                    mdl.device = device
                    mdl.pushto(mdl.device)   
                    break
                except:
                    # if push error (maybe memory overflow, try again)
                    print("Push to device cuda:{} fail, try again ...")
                    continue
    print("epoch:{}\tclass:{}\tPush model to {}. Done.".format(epoch_str,iclass_str, mdl.device), file=sys.stdout)
    
    # zero pad data for batch training
    max_len_ = max([x.shape[0] for x in xtrain])
    xtrain_padded = pad_data(xtrain, max_len_)

    traindata = DataLoader(dataset=TheDataset(xtrain_padded, lengths=l, device=mdl.device), batch_size=options["Train"]["batch_size"], shuffle=True)

    # niter counts the number of em steps before saving a model checkpoint
    # niter = options["Train"]["niter"]
    
    # add number of training data in model
    mdl.number_training_data = len(xtrain)
    
    # set model into train mode
    mdl.train()

    # Reset the convergence monitor
    if int(mdl.iepoch) == 1:
        mdl.monitor_._reset()
    
    #convg_count = 0
    #ncon_int = 3 # No. of consecutive iterations to be checked
    iter_arr = [] # Empty list to store iteration numbers to check for consecutive iterations
    iter_count = 0 # Counts the number of consecutive iterations
    iter_prev = 0 # Stores the value of the previous iteration index

    for iiter in range(niter):
        #mdl.fit(traindata)
        mdl.iter = iiter
        flag = mdl.fit(traindata)
        # if flag and iiter > (niter // 2):
        if flag == True and iter_prev == 0: # If convergence is satisfied in first condition itself
            print("Iteration:{}".format(iiter))
            iter_count += 1
            iter_arr.append(iiter)
            if iter_count == ncon_int:
                print("Exit and Convergence reached after {} iterations for relative change in NLL below :{}".format(iter_count, tol))
                break    

        elif flag == True and iter_prev == iiter - 1: # If convergence is satisfied
            print("Iteration:{}".format(iiter))                                                                        
            iter_count += 1 
            iter_arr.append(iiter)
            if iter_count == ncon_int:
                print("Consecutive iterations are:{}".format(iter_arr))
                print("Exit and Convergence reached after {} iterations for relative change in NLL below :{}".format(iter_count, tol))  
                break 
            
        else:
            print("Consecutive criteria failed, Buffer Reset !!")
            print("Buffer State:{}".format(iter_arr)) # Display the buffer state till that time
            iter_count = 0
            iter_arr = []

        iter_prev = iiter # Set iter_prev as the previous iteration index

    # Push back to cpu for compatibility when GPU unavailable.
    mdl.pushto('cpu')
    save_model(mdl, fname=out_mdl)
    sys.exit(0)


