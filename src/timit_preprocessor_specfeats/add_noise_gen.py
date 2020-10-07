from scipy.io import wavfile
import argparse
from glob import glob
from parse import parse
import os, sys
import numpy as np 
from multiprocessing.dummy import Pool
from functools import partial
import copy

def add_noise(n, ntype, sigma, Fs, noise_folder="../../data/NoiseDB/NoiseX_16kHz/"):
    
    """ Adds noise of a certain type and std.
    ------

    Args:

    n - No. of samples in the Signal data
    ntype - The type of Noise to be added
    sigma - The standard deviation of the noise required
    noise_folder - folder that contains the files from NoiseX92 Database
    
    Targets:

    noise - A noise snippet that is going to be added to the signal segment at a random location

    """

    try:

        if Fs % 1000 == 0:
            Fs_kHz = Fs // 1000
        
        noise_filename = os.path.join(noise_folder, "{}_{}kHz.wav".format(ntype, Fs_kHz))
        Fs_actual, loaded_noise = wavfile.read(noise_filename)

        try:
            assert(n < loaded_noise.shape[0])
            assert(Fs_actual == Fs)

        except AssertionError as e:
            
            print("Noise file:{} is too short or Sampling rate is inconsistent".format(noise_filename), file=sys.stderr)

        # Find a random location in the file to extract the noise and add it to the file
        istart = np.random.randint(loaded_noise.shape[0] - n) # generate a random location
        raw_noise = loaded_noise[istart: istart + n] # Extract a random part of the noise        

        # Add the particular SNR to the noise
        raw_noise_norm = mean_normalize(raw_noise) # Ensuring the noise signal has zero mean and unit variance
        noise = (raw_noise_norm * sigma) / raw_noise_norm.std() # Multiplying by this st.dev ensures the required noise power, hence required SNR

    except:

        print("{} noise file not found !!!!".format(noise_filename))
        noise = None
    
    return noise 

def mean_normalize(input_sig):
    
    """ This function checks if the given signal has zero mean and unit variance or not
        If Not, it converts the signal to a zero mean and unit variance before 
        other preprocessing operations
    ------
    
    Args:

    input_sig - input signal that needs to be checked

    Returns:

    input_sig_normalized - input signal that has been mean variance normalised

    """
    if np.mean(input_sig) != 0:
        # Normalization required
        input_sig_normalized = (input_sig - np.mean(input_sig, axis=0))
    
    else:
        # No normalization required
        input_sig_normalized = copy.deepcopy(input_sig)
    
    return input_sig_normalized 

def new_filename(file, ntype, snr):
    
    """ Append noise type and power at the end of wav filename.
    ------

    Args: 

    file - Name of the existing filename (.wav)
    ntype - Type of noise used for corruption
    snr - SNR value used for the experiments
    
    Returns:

    filename with the format "<NAME>.wav.<NOISETYPE>.<SNR>dB" as a string

    """
    return file.replace(".WAV", ".WAV.{}.{}dB".format(ntype, snr))


def corrupt_data(s, ntype, Fs, snr):

    """ Corrupt a signal with a particular noise.
    ------

    Args:

    s - signal data which is to be corrupted by noise
    ntype - noise type used for corruption
    snr - Signal to Noise ratio 

    Returns:

    sn - contains signal data that has been corrupted with the noise at random places

    NOTE: The signal and noise data should be individually processed by Mean Variance 
    Normalization beforehand
    
    """
    s_normalized = mean_normalize(s)
    s_std = np.std(s_normalized)
    #s_std = np.std(s)
    n_std = 10 ** (- snr / 20) * s_std
    n = add_noise(s_normalized.shape[0], ntype, n_std, Fs)
    sn = (s_normalized + n).astype(s.dtype)
    return sn


def corrupt_wav(file, ntype=None, snr=None):
    
    """ Read in  a wav file, corrupt it with noise and write to a new file.
    ------

    Args: 

    file - contains the name of the wavfile with complete path location
    ntype - Noise Type used for the corruption process
    snr - SNR value required by the experiments

    Returns:

    None

    The corrupted wavfiles are written in the same exact location as the 
    original wavfile, with a new name as per the type of noise and the data
    
    """
    rate, s = wavfile.read(file)
    sn = corrupt_data(s, ntype, rate, snr)
    wavfile.write(os.path.join(new_filename(file, ntype, snr)), rate, sn)
    return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Add a particular noise type to WAV files.")

    parser.add_argument('-timit', metavar="<Timit location>", type=str)
    parser.add_argument('-opt', metavar="<Signal to Noise Ratio (dB)>", type=str)
    parser.add_argument('-j', metavar="<Number of jobs (default: numcpu)>",
                        type=int, default=os.cpu_count())

    args = parser.parse_args()
    
    try:
        dset, ntype, snr = parse("{}.{}.{:d}dB", args.opt)
    except TypeError as e:
        print("No noise to be added with option: {}.\nExit.".format(args.opt),file=sys.stderr)
        sys.exit(0)

    #if os.path.isdir(os.path.join(args.timit, ntype + "_noise")):
    #    outfolder = os.path.join(args.timit, ntype + "_noise")
    #else:
    #    os.mkdir(os.path.join(args.timit, ntype + "_noise"))
    #    outfolder = os.path.join(args.timit, ntype + "_noise")

    if dset == "test":
        wavs = glob(os.path.join(args.timit, "TEST", "**" , "*.WAV"), recursive=True)
        f = partial(corrupt_wav, ntype=ntype, snr=snr)
        with Pool(args.j) as pool:
            pool.map(f, wavs)

    elif dset == "train":
        wavs = glob(os.path.join(args.timit, "TRAIN", "**" , "*.WAV"), recursive=True)
        f = partial(corrupt_wav, ntype=ntype, snr=snr)
        with Pool(args.j) as pool:
            pool.map(f, wavs)

    sys.exit(0)