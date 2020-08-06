#/bin/bash
SHELL=/bin/bash
# Project Root defines the path where the project is present
ifndef PROJECT_ROOT 
	PROJECT_ROOT=../../..
endif
ifndef KALDI_ROOT # If Kaldi Root path is not defined it will define the Kaldi Root path
	KALDI_ROOT=$(PROJECT_ROOT)/kaldi
endif
ifndef DATA_OUT # If the Data output path is not defined, it will define the Data Output path
	DATA_OUT=$(PROJECT_ROOT)/gm_hmm/data
endif
ifndef TIMIT_ROOT # If the TIMIT_root path is not defined, it will define the root path
	TIMIT_ROOT=$(DATA_OUT)/TIMIT
endif

ifndef ntype # ntype probably refers to the type of Noise
	ntype=clean
endif

ifndef snr # if the SNR value is not defined, it initialises the SNR with a default value
	snr=5
endif


#targets_= $(shell echo {train,test}.39.pkl)
#noise_targets_ = $(shell echo test.39.white.{12,17,22,30}dB.pkl)
noise_targets_ = $(shell echo test.39.white.5dB.pkl)

targets= $(addprefix $(DATA_OUT)/,$(targets_)) $(addprefix $(DATA_OUT)/,$(noise_targets_))

interm=.data
material=$(interm)/material
processed=$(interm)/processed

#interm_files= $(shell echo $(material)/{train,test}.lbl) $(shell echo $(material)/test.white.{12,17,22,30}dB.lbl)
interm_files= $(shell echo $(material)/{train,test}.lbl) $(shell echo $(material)/test.white.5dB.lbl)

all: noise
	
noise: feats
		

feats: $(targets)

%.pkl: %.scp
	python make_dataset.py $^ $(TIMIT_ROOT) $@

%.scp: $(interm_files)
	./extract_mfcc.sh $(KALDI_ROOT) $(TIMIT_ROOT) $(processed) $@

$(material)/%.lbl: %.noise
	python parsing.py $(TIMIT_ROOT) $*

%.noise:
	python add_noise_gen.py -timit $(TIMIT_ROOT) -opt $*
convert:
	./convert_wav.sh $(TIMIT_ROOT) $(KALDI_ROOT)

clean:
	rm -rf $(interm)
	rm -f $(DATA_OUT)/*.{scp,pkl}

.SECONDARY:
