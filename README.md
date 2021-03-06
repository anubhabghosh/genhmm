# Normalizing Flow based HMMs for Explainable Classification of Speech Phones

This repository contains the relevant code for the work: Normalizing Flow based HMMs for Explainable Classification of Speech Phones at 86.6\% Accuracy. 

Forked from the parent repository: https://github.com/FirstHandScientist/genhmm

### Creation of Virtual environment

Please rename the default directory "genhmm" into "gm_hmm" (current module importing depents on this directory name), e.g.
```bash
mv genhmm gm_hmm
```

Create a virtual environment with a python3 interpreter, in the newly created `gm_hmm/` directory.
```bash
$ cd gm_hmm
$ virtualenv -p python3.6 pyenv
$ cd ..
```

Add the parent directory of `gm_hmm/` to the path:
```bash
$ echo $PWD > gm_hmm/pyenv/lib/python3.6/site-packages/gm_hmm.pth
```

Install the dependencies:
```bash
$ cd gm_hmm
$ source pyenv/bin/activate
$ pip install -r requirements.txt
```
### Additional tools
You must install `GNU make`, on Ubuntu:
```bash
$ sudo apt install build-essential
$ make -v
GNU Make 4.1
Built for x86_64-pc-linux-gnu
...
```

## Getting Started

## Dataset preparation
See [README.md](src/timit-preprocessor/README.md) in `src/timit-preprocessor`

## Experimental setup
Start by creating the necessary experimental folders for using model "GenHMM" and data feature length of 39,  with:
```bash
$ make init model=gen nfeats=39 exp_name=genHMM
```
**NOTE:** The value assigned to the variable `model` decides which model is going to be run. It should be set to:
- `gmm` : GMM-HMM, or
- `gen` : NVP-HMM, or
- `glow` : Glow-HMM

Change directory to the created experiment directory:
```bash
$ cd exp/gen/39feats/genHMM
```
## Set training and model configurations
Appropriate training configurations can be set by modifying the parameters associated in the file *default.json*.
- For setting the values of training and evaluation parameters (under `"Train"`):
  - "niter" :  No. of training iterations
  - "batch_size": batch size used for training
  - "eval_batch_size": batch size used during testing / evaluation
  - "n_states_min": No. of minimum number of states in the HMM
  - "n_states_max": No. of maximum number of states in the HMM
- For setting model specific parameters, the relevant dictionary must be modified. For simulating Gaussian mixture model based HMM, change the parameters under `"GMM"`. Whereas for normalizing flow based mixture models, modify the parameters associated with `"Net"` for RealNVP based HMM (referred to as NVP-HMM) and `"GLOW_Net"` for Glow based HMM (referred to as Glow-HMM) 
- For running the NVP-HMM, edit the Makefile while being present in the *experiment folder* to set the value of the variable `model` to `gen`. For running GMM-HMM, `model=gmm` and similarly for Glow-HMM set `model=glow`.

### Running the training on clean data as well as evaluation on clean data

The number of epochs (`nepochs`) is here number of checkpoints. One checkpoint consist of multiple expectation maximization steps, which you can configure at *default.json*. To run the training of `genHMM` on 2 classes out of 39 classes and during 10 checkpoints, with two distributed jobs, run:
```
$ make j=2 nclasses=2 totclasses=39 nepochs=10 
```
Modify the `j` option to change the number of jobs for this experiment.

The logs appear in `log/class...`. you can follow the training with:
```bash
$ make watch
```
Before running the experiment, it is always very helpful to do a sanity check by using the 'dry-run' (`-n`) option of `make`:
```
make j=2 nclasses=2 totclasses=39 nepochs=10 -n
```
This will produce a detailed set of the commands that are going to be executed in an automated fashion. This helps to check whether the required number of classes are going to be trained or not, where the model files will be saved, where the test results will be saved, etc. 

### Running the training on clean data as well as evaluation on noise data
Considering testing on noise data at a particular SNR level (in dB), by adding an additional argument `noise=white.25dB`. The argument `white.25dB` translates as the use of the test data corrupted with *white* noise at 25dB SNR value. The number of epochs (`nepochs`) is here number of checkpoints. One checkpoint consist of multiple expectation maximization steps, which one can configure at *default.json*. To run the training of `genHMM` on 39 classes out of 39 classes and during 1 checkpoints, with two distributed jobs, run:
```
$ make j=2 noise=white.25dB nclasses=39 totclasses=39 nepochs=1
```
Modify the `j` option to change the number of jobs for this experiment.

Other varieties of arguments are also used for the experiments with the general format as <*noise_type*>.<*SNR*>dB, where *noise_type* can be "white" / "babble" / "pink" / "hfchannel" and *SNR* can be 10 / 15 / 20 / 25 (in dB scale)

### Running the training on clean + noisy data (10dB white noise in this case) as well as evaluation on noise data

For creating the combined dataset of clean + noisy data for training models like GMM-HMM and NVP-HMM, the pre-processing scripts must be run in `src/timit-preprocessor/` to first individually generate the clean training data for all classes (`train.39.pkl`), and then the noisy training data by adding white noise at 10dB (`train.39.white.10dB.pkl`). The two files can be combined into one training data file by using the script `bin/combine_trdata.py` (check the help by `python bin/combine_trdata.py -h`. This will yield a single training data file that can be used for noisy training. Training can be done as usual by a similar procedure as earlier (**NOTE**: It may be required to rename the combined training file appropriately to utilise the automated operations of the Makefile)

For testing on noise data at a particular SNR level (in dB), by adding an additional argument `noise=white.10dB`. The argument `white.10dB` translates as the use of the test data corrupted with *white* noise at 10dB SNR value. The number of epochs (`nepochs`) is here number of checkpoints. One checkpoint consist of multiple expectation maximization steps, which one can configure at *default.json*. To run the training of `genHMM` on 39 classes out of 39 classes and during 1 checkpoints, with two distributed jobs, run:
```
$ make j=2 noise=white.10dB nclasses=39 totclasses=39 nepochs=1
```
Modify the `j` option to change the number of jobs for this experiment.
Other values of SNR are also used for the experiments with the general format as white.<*SNR*>dB, where *SNR* can be 5dB, 10dB, 12dB, 15dB, etc. (in dB scale)

### Processed files
- Model files are created as in the format `epochX_classC.mdlc`, where `X` refers to the present checkpoint and `C` refers to the class under consideration. The aggregated model (`epochX.mdl`) which is mainly used for classification in the evaluation / testing stage (where `X` refers to the present checkpoint). Models as per specific generative model such as GMM / NVP / Glow can be run by modifying the appropriate `model` variable in the *Makefile* (after one has used `cd` to get to the specific experiment folder). 

- Test accuracies are generated and stored in `.accc` files in the format `epochX_classC.accc`, where `X` refers to the present checkpoint and `C` refers to the class under consideration. The weighted aggregate of accuracies (weighted average using number of samples in each class) is stored in the file `epochX.acc`, where `X` refers to the present checkpoint.

## Generate decision fusion results
The decision fusion results require three models trained on each of the 39 classes (we show results for 39 classes in the paper). This carried out by executing the `main()` associated with the file `bin/compute_accuracy_voting.py`. The required setup is that all three models have a folder setup (or similar) for aggregating model predictions on clean data:

```
gmmHMM_clean/
| - <other folders such as bin, data, src, etc.>
| - <other files>
| - models/
|   - <other files>
|   - epoch1.mdl
nvpHMM_clean/
| - <other folders such as bin, data, src, etc.>
| - <other files>
| - models/
|   - <other files>
|   - epoch1.mdl
glowHMM_clean/
| - <other folders such as bin, data, src, etc.>
| - <other files>
| - models/
|   - <other files>
|   - epoch1.mdl
```
So, the *aggregated model file* (models/epoch1.mdl) used for test set predictions is present in the same folder structure for each of the three models - GMM, NVP and Glow. The internal path to the `epoch1.mdl` for each of the models should be set (by the user) relative to that of the GlowHMM model (`glowHMM_clean/`) (i.e. executing the script (`bin/compute_accuracy_voting.py`) in the folder for `glowHMM_clean`. Also, inside the script, one has to set the name of the output file by using the variables `file1` (.log file) and `dr_filename` (.json file). The results are written down into two files: 
- One is a log file (textfile), that contains accuracy metrics computed on the test data for each of the three models, as well a couple of other metrics that are relevant to understanding the model performance.
- The other one is .json file that contains the same results that are print out in the log file but in the form of a Pandas dataframe so that better analysis can be done on them later on (**NOTE:** For post-analysis of data stored in .json files, Pandas (v1.0.4) needs to be installed)

### Usage (for clean data):

Example usage (for fusing predictions on clean test data for each of the three models - GMM, NVP, Glow):
```
python bin/compute_accuracy_voting.py models/epoch1.mdl data/train.39.pkl data/test.39.pkl test
```
*NOTE:* Before executing the script, the paths of the respective model files need to be set in the initial variables *gmm_mdl_path*, *nvp_mdl_path*, and *glow_mdl_path*. Usually the paths are set as relative model paths w.r.t. the location of the model file (models/epoch1.mdl) for the Glow-HMM file. In this example it has been assumed that we are concerned with fusing predictions for clean test data, i.e. free from varieties of noise. For seeing further instruction, before executing the script press `python bin/compute_accuracy_voting.py -h` or `python bin/compute_accuracy_voting.py --help`

### Usage (for noisy data):
Example usage (for fusing predictions on noisy test data for each of the three models - GMM, NVP, Glow):
```
python bin/compute_accuracy_voting_noise.py models/epoch1.mdl data/train.39.pkl data/test.39.pkl test [{pink/white/babble/hfchannel}.{SNR value in dB}dB]
```
For example for *white* noise, *SNR* value (in dB) is 25dB, on the test set, we have:
```
python bin/compute_accuracy_voting_noise.py models/epoch1.mdl data/train.39.pkl data/test.39.pkl test white.25dB
```
*NOTE:* Before executing the script, the paths of the respective model files need to be set in the initial variables *gmm_mdl_path*, *nvp_mdl_path*, and *glow_mdl_path*. Usually the paths are set as relative model paths w.r.t. the location of the model file (models/epoch1.mdl) for the Glow-HMM file. 
For seeing further instruction, before executing the script press `python bin/compute_accuracy_voting_noise.py -h` or `python bin/compute_accuracy_voting_noise.py --help`

## Additional metrics of interest
There is also a script for generating class wise as well as weighted metrics for precision, recall and F1-score. These weighted class-wise metrics are also referred to as *macro* class-wise metrics in some common literature. The script basically creates a confusion metrics and returns detailed class wise metrics in separate log files. Example usage (for generating class wise metrics and fusing predictions) for each of the three models (GMM, NVP, Glow) during the evaluation stage is as follows:
```
python bin/compute_accuracy_cfmatrix.py models/epoch1.mdl data/train.39.pkl data/test.39.pkl test
```
*NOTE:* Before executing the script, the paths of the respective model files need to be set in the initial variables *gmm_mdl_path*, *nvp_mdl_path*, and *glow_mdl_path*. Usually the paths are set as relative model paths w.r.t. the location of the model file (models/epoch1.mdl) for the Glow-HMM file. For seeing further instruction, before executing the script press `python bin/compute_accuracy_cfmatrix.py -h` or `python bin/compute_accuracy_cfmatrix.py --help`

## Examples of evaluation results:
- Examples of some evaluation results using scripts `bin/compute_accuracy_voting.py` is found in the folder: *com_machine/latest_results_Oct20_modifiedWN/* in the form of .json and .log file (for test data)

- Examples of some evaluation results using scripts `bin/compute_accuracy_cfmatrix.py` is found in the folder: */cfmetrics_results/* in the form of .json, .log and .mat files (for test data). The class-wise metrics for precision, recall, F1 are found in the form of detailed logs and json files for each of the type of models. Example of a saved file: cfmetrics_<model_name>.{log, json}, where *model_name* is one among *gmm / nvp / glow / voting*.  Example of a saved confusion matrix file: <model_name>_cfmatrix.mat, where *model_name* is one among *gmm / nvp / glow / voting*. 
### 

