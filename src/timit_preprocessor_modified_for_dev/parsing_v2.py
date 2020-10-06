# Modified and added comments 
# from the 'timit-preprocessor' from FirstHandScientist/genhmm/src 

# Import the necessary libraries
import os
import sys
from parse import parse
import argparse
import random
import numpy as np

def extension_formatter(ntype="clean", snr=None):
    """ This function returns a speech file extension based on the noise type
    and snr value (in case noise type is NOT "clean")
    """
    if ntype == "clean":
        sp_file_extension = ".WAV"
    else:
        # This particular extension style is used because it is the way 
        # the noise file addition fn. produces the output
        sp_file_extension = ".WAV.{}.{}dB".format(ntype, snr)
    
    return sp_file_extension

def extract_phone_seq(phone_labelfile_path):
    """ This function takes the full path of phone labelfile for a given sentence .WAV file
    and returns a list containing the labels for every phone spoken in the given sentence

    Phone sequences are stored in a .PHN file as:
    <BEGIN_SAMPLE_No.> <END_SAMPLE_No.> <Phonetic transcription>
    """
    phone_sequence = [] # Empty list for storing phone labels

    with open(phone_labelfile_path, "r+") as phnlbl:
        for line in phnlbl:
            # Obtain the phone label at the end of the line 
            # by splitting the line using spaces as delimiters and 
            # omitting the extra space at the end
            phone_sequence.append(line.split(' ')[-1].strip())

    return phone_sequence

def create_feature_scps_and_labels(args, target, target_folder, extension, phn_extension, num_dev_speakers = 50):
    """ This function creates the feature scp (.../[train/test/dev].wav.scp) files and 
    label files (.../[train/test/dev].lbl) given the arguments and target
    ----
    Args:
        - args ([object]): Argument object containing details passed at runtime 
        - target ([str]): Target string denoting whether to create "train / test / dev"
        - target_folder ([str]): Target folder string denoting whether to create "train / test / dev"
        - num_dev_speakers (int): Number of speakers in the validation set. By default 50
    Returns:
        - None
    """

    # Assign the directory for the features
    features = "{}/{}.wav.scp".format(INTERMEDIATE_PATH, args.folder)
    
    # Assign the directory for the labels
    labels = "{}/{}.lbl".format(INTERMEDIATE_PATH, args.folder)

    # Num_speakers per dialect region (there are 8 DRs)

    num_dev_speakers_per_dr_eq = num_dev_speakers // 8
    num_dev_speakers_per_dr_last = int(num_dev_speakers_per_dr_eq + (num_dev_speakers - num_dev_speakers_per_dr_eq * 8))
    num_dev_speakers_per_dr_array = [num_dev_speakers_per_dr_eq for _ in range(7)]
    num_dev_speakers_per_dr_array.append(num_dev_speakers_per_dr_last)
    random.shuffle(num_dev_speakers_per_dr_array)
    print(num_dev_speakers_per_dr_array)

    index_dr = 0

    # Parsing the feature file begins ...
    with open(features, "w+") as f, open(labels, "w+") as l:

        # Obtain the dialect region folder (DR1, DR2, DR3, ..., DR8) from the 
        # output of os.listdir(<train folder path>) or os.listdir(<test folder path>)
        for dialect_region in os.listdir(target_folder):
            
            num_dev_speakers_per_dr = num_dev_speakers_per_dr_array[index_dr]

            # Obtain the dialect region folder path
            dialect_region_path = os.path.join(target_folder, dialect_region)

            # Obtain the folder for the speaker by using os.listdir(<path to dialect region>)
            if target.lower() == "dev":
                
                # Dev set consists of 50 speakers chosen from the TEST SET
                speaker_ids_testanddev = os.listdir(dialect_region_path)
                speaker_ids_dev = [spk_id for spk_id in speaker_ids_testanddev if spk_id not in CORE_TEST_SPEAKERS]
                # First 50 speakers that are not in the core test set are chosen as dev-set speakers
                speaker_ids = list(np.random.choice(speaker_ids_dev, num_dev_speakers_per_dr, replace=False))
                print(speaker_ids)
                #speaker_ids = speaker_ids_dev[:num_dev_speakers] 
            
            elif target.lower() == "test":
                
                # In this case "test" refers to CORE_TEST_SET
                speaker_ids_testanddev = os.listdir(dialect_region_path)
                speaker_ids_test = [spk_id for spk_id in speaker_ids_testanddev if spk_id in CORE_TEST_SPEAKERS]
                speaker_ids = speaker_ids_test
            
            elif target.lower() == "train":
                speaker_ids = os.listdir(dialect_region_path)

            # Extract the speaker - based feature information hence forth
            for speaker_id in speaker_ids:
                
                speaker_id_path = os.path.join(dialect_region_path, speaker_id)
                
                # Creating a list containing the sentence Ids without the extensions (.wav / .phn / .wrd / .txt)
                # by traversing the os.listdir(speaker_id_path) and obtain the sentence IDs
                # which is of the form <SentenceType(SA/SI/SX)><sentence number>.<file extension> 
                # For example: SA1.PHN, SA1.WAV, etc.
                raw_sentence_ids = [sentence_id.split('.')[0] for sentence_id in os.listdir(speaker_id_path)]
                
                # Since the same sentence is spoken by multiple speakers, we obtain a unique list of 
                # sentence ids by using set() 
                sentence_ids = list(set(raw_sentence_ids))
                count = 0
                # Obtaining the sentence
                for sentence_id in sentence_ids:
                    # If the sentence ID exists and excludes the SA (phonetically dialect) sentences
                    if sentence_id and ("SA" not in sentence_id):   
                        count += 1
                        # Obtain the wavfile path
                        sentence_wavfile_path = os.path.join(speaker_id_path, sentence_id + extension)
                        # Write down the scp file for the features
                        f.write('{}-{}-{} {}\n'.format(dialect_region, speaker_id, sentence_id, os.path.abspath(sentence_wavfile_path)))
                        # Obtain the phonetic label file path
                        phone_labelfile_path = os.path.join(speaker_id_path, sentence_id + phn_extension)
                        # Obtain the phone sequence
                        phone_labels_seq = extract_phone_seq(phone_labelfile_path)
                        # Write down the lbl file for the phone labels
                        l.write('{}-{}-{} {}\n'.format(dialect_region, speaker_id, sentence_id, ','.join(phone_labels_seq)))
                #print(count)
            
            index_dr += 1

def main(args=None):
    """ This function takes the target folder and extracts the path locations for the features 
    and the actual labels from the .WAV and .PHN files respectively of the TIMIT Dataset
    ----
    Args:
    - args : name of the file in the format 
            "<target>.<noisetype>.<snrvalue>dB"
    Returns:
        None
    """
    # Usually target - {"train" or "test"}
    try:
        # In case the targe conatins noise
        target, noise_type, SNR = parse("{}.{}.{}dB", args.folder)
    except TypeError as e:
        # In case the target is not containing any noise, SNR = 0 dB
        target = args.folder
        noise_type = "clean"
        SNR = 0

    # Assign the extension for the target .WAV file
    extension = extension_formatter(ntype=noise_type, snr=SNR)

    # Assign the extension for the target .PHN file (which is .PHN as per docs)
    phn_extension = ".PHN"

    # Assign the full path to the target folder. The .upper() function is 
    # used as the subfolders are named as TEST / TRAIN. args.datapath stores 
    # relative path to the TIMIT dataset
    if target == "test":
        target_folder = os.path.join(args.datapath, target.upper()) 
    elif target == "dev":
        # As we create the dev set from the test set
        target_folder = os.path.join(args.datapath, "TEST")
    elif target == "train":
        target_folder = os.path.join(args.datapath, target.upper()) 

    # This will create the respective wav.scp and .lbl files
    create_feature_scps_and_labels(args, target, target_folder, extension, phn_extension, num_dev_speakers=50)

if __name__ == "__main__":

    # Style for calling this function
    parser = argparse.ArgumentParser(
        description="parsing .wav.scp files for advanced use\n"
                    "e.g. python3 parsing_v2.py ~/Workspace/data/timit train", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('datapath',
                        metavar="<timit relative path>", type=str)
    parser.add_argument('folder',
                        metavar="<train|test>", type=str)

    # Obtain the arguments to the parsing function viz. 
    # location of TIMIT dataset (relative path), target name (test / train / test.<noisetype>.<snr>dB)
    args = parser.parse_args()

    # Set the intermediate folder path
    INTERMEDIATE_PATH = '.data/material'

    # Core test set from timit/readme.doc (https://github.com/awni/speech/blob/master/examples/timit/preprocess.py)
    CORE_TEST_SPEAKERS_lower = ['mdab0', 'mwbt0', 'felc0', 'mtas1', 'mwew0', 'fpas0',
                     'mjmp0', 'mlnt0', 'fpkt0', 'mlll0', 'mtls0', 'fjlm0',
                     'mbpm0', 'mklt0', 'fnlp0', 'mcmj0', 'mjdh0', 'fmgd0',
                     'mgrt0', 'mnjm0', 'fdhc0', 'mjln0', 'mpam0', 'fmld0']

    CORE_TEST_SPEAKERS = [spk_id.upper() for spk_id in CORE_TEST_SPEAKERS_lower]

    # Create the path folder if it doesn't already exist
    if not os.path.exists(INTERMEDIATE_PATH):
        os.makedirs(INTERMEDIATE_PATH)

    # Calling the main function where the main parsing of dataset occurs
    main(args)


            




