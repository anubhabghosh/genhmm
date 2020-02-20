# GenerativeModel-HMM implementation
#
#
SHELL=/bin/bash
PYTHON=python
# Outline the various folders for the Data and Model storage
SRC=src
BIN=bin
DATA_tmp=data
MODELS_tmp=models

# Specify the number of classes that is to be classified 
# by the HMM : 10 or 39 or 61...If nothing is mentioned it takes 61 
ifndef totclasses
	totclasses=61
endif
# Specify the number of epochs required to run the algorithm
ifndef nepochs
	nepochs=10
endif
# Specify the number of classes, if nothing nclasses = 2 
ifndef nclasses
	nclasses=2
endif
# Specify the number of features used by the model (MFCC assumed)
ifndef nfeats
	nfeats=39
endif
# Specify the number of jobs used by the processor
ifndef j
	j=2
endif
# Specify the model name to be used : gaus (GMM-HMM) or genHMM (GenHMM)
ifndef model
	model=gaus
endif
# Specify the experiment name (use proper name to keep track later on)
ifndef exp_name
	exp_name=default
endif
# Some more directories defined, what is meant by ROBUST is not clear yet
ROBUST=robust
EXP=exp/$(model)
EXP_DIR=$(EXP)/$(nfeats)feats/$(exp_name)

MODELS=models
DATA=data
LOG=log

# 'init' here refers to a function, the function initialises the dependencies
init: MODELS=$(EXP_DIR)/models
init: LOG=$(EXP_DIR)/log

# Some intermediate variables initialised
MODELS_INTERM=$(shell echo $(MODELS)/epoch{1..$(nepochs)})
TEST_INTERM=$(shell echo {1..$(nepochs)})

# Defines the training data used with proper path
training_data=$(DATA)/train.$(nfeats).pkl

# If Clean data is used for testing, then 'noise' is not used
ifndef noise
	testing_data=$(DATA)/test.$(nfeats).pkl
else
	testing_data=$(DATA)/test.$(nfeats).$(noise).pkl
endif

# Defines the complete path for the model for every class and 
# for accuracy computation of every model
mdl_dep=$(shell echo $(MODELS)/%_class{1..$(nclasses)}.mdlc)
acc_dep=$(shell echo $(MODELS)/%_class{1..$(nclasses)}.accc)
rbst_dep=$(shell echo $(ROBUST)/epoch$(tepoch)_class{1..$(nclasses)}.accc)

###########################################
# This is what make will create : all #
###########################################
all: train

test:
	echo $(mdl_dep)
	echo $(acc_dep)

# Creates the model and log files using the dependencies provided earlier
init:
	mkdir -p $(MODELS) $(LOG)
	mkdir -p $(EXP_DIR)/data
	cp $(training_data) $(testing_data) $(EXP_DIR)/data
	ln -s $(realpath data)/phoneme_map_61_to_39.json $(EXP_DIR)/data/phoneme_map_61_to_39.json
	ln -s $(realpath bin) $(EXP_DIR)/bin
	ln -s $(realpath src) $(EXP_DIR)/src
	cp default.json $(EXP_DIR)
	sed -e 's/model=.*/model=$(model)/' -e 's/nfeats=.*/nfeats=${nfeats}/' -e 's/totclasses=.*/totclasses=$(totclasses)/' Makefile > $(EXP_DIR)/Makefile

# Prepares the data into training and testing sets
prepare_data: $(training_data) $(testing_data)
	$(PYTHON) $(BIN)/prepare_data.py "$(nclasses)/$(totclasses)" $^

# This is the function which does the main training, first checks whether the number of model files (empty)
# created are equal to the actual number of classes. If not removes the models existing, and creates models
# and acc files. 
train: prepare_data
	echo $(DATA) $(MODELS) $(LOG)
	echo $(MODELS_INTERM)
	for i in $(MODELS_INTERM); do \
		if [[ `echo $${i%.*}_class*.mdlc | wc -w` != $(nclasses) ]]; then rm -f $$i.{mdl,acc}; fi; \
		$(MAKE) -j $(j) -s $$i.mdl; \
	 	$(MAKE) -j $(j) -s $$i.acc; \
	 	sleep 2;\
	done
#	echo "Done" > $^

# This function is used to create the model file (.mdl) for the Gauss or Gen HMM depending on
# the aggregate_models.py function using all the individual class models (which have the extension '.mdlc')
$(MODELS)/%.mdl: $(mdl_dep)
	$(PYTHON) $(BIN)/aggregate_models.py $@

# This function is used to create the accuracy file (.acc) for the Gauss or Gen HMM model depending on
# the aggregate_accuracy.py function using all the individual class accuracies (which have the extension '.accc')
$(MODELS)/%.acc: $(acc_dep)
	$(PYTHON) $(BIN)/aggregate_accuracy.py $(training_data) $(testing_data) $^ > $@
	cat $@ >> $(LOG)/class_all.log

# This function is used to create the '.mdlc' (model file for every class) and has extension '.mdlc'
# Typically outputs the convergence related data provided by ConvergenceMonitor function and Experiment-related Metadata
$(MODELS)/%.mdlc:
	$(eval logfile=$(LOG)/`basename $@ | sed -e 's/^.*\(class\)/\1/g' -e 's/.mdlc/.log'/g`)
	echo `date` ":" $(PYTHON) $(BIN)/train_class_$(model).py $(training_data) $@ >> $(logfile)
	$(PYTHON) $(BIN)/train_class_$(model).py $(training_data) $@ >> $(logfile)

$(MODELS)/%.accc: $(MODELS)/%.mdlc
	$(PYTHON) $(BIN)/compute_accuracy_class.py $^ $(training_data) $(testing_data) >> $@

# testing part only
# test one checkpoint
test_one: 
	$(MAKE) -j $(j) -s $(ROBUST)/epoch$(tepoch).acc

# test multiple check points
test_all:
	echo $(TEST_INTERM)
	for i in $(TEST_INTERM); do \
		$(MAKE) -j $(j) -s $(ROBUST)/epoch$$i.acc tepoch=$$i; \
		sleep 2;\
	done

# NOTE: Did not understand the exact use of this function in Clean Data training
# Maybe Useful in case of Noisy Data
$(ROBUST)/epoch%.acc: $(rbst_dep)
	$(PYTHON) $(BIN)/aggregate_accuracy.py $(training_data) $(testing_data) $^ > $@
	cat $@ >> $(LOG)/class_all.log

$(ROBUST)/%.accc:
	@echo $(subst $(ROBUST),$(MODELS),$@)
# 	string replacement such compute_acccuracy_class can recognize
	$(PYTHON) $(BIN)/compute_accuracy_class.py $(subst .accc,.mdlc,$(subst $(ROBUST),$(MODELS),$@)) $(training_data) $(testing_data) >> $@

# This function is used to create the logfiles for the experiments
# Basically takes the end part of all log files including the main log file of class_all.log
# and the individual class log files and displays them on terminal
# Downside: tail -f makes it wait even if end of file is reached and 
# keeps waiting for data from the user
watch:
	tail -f $(LOG)/class*.log

# Use this to clean up / delete experimental results 
# Cleans up .mdl and .acc files for every epoch, individual
# .mdlc and .accc files and also the logs (both individual
# and total for every class)
clean:
#	rm -f $(DATA)/train*_*.pkl
#	rm -f $(DATA)/test*_*.pkl 
#	rm -f $(DATA)/class_map.json
	rm -f $(MODELS)/epoch*.{mdl,acc} 
	rm -f $(MODELS)/epoch*_class*.{mdlc,accc}
	rm -f $(LOG)/class*.log

# USE CAREFULLY: This will delete all data and also original
# phoneme mappings in class_map.json.
clean-data:
	rm -f $(DATA)/*_*.pkl $(DATA)/class_map.json


.SECONDARY: 

.PRECIOUS:
