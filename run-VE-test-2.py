"""
Perform speech separation on Data Set 2 (generated scenes using TASCAR) with FastICA.
"""
import os, shutil, numpy as np, soundfile as sf, subprocess
from scipy.io.wavfile import read, write
from sklearn.decomposition import PCA, FastICA
from pystoi import stoi
import pandas as pd
from os import listdir, mkdir
from os.path import exists
from time import perf_counter
import sys

from tascarpy import tascarpy

WRITE_TO_PATH = '.../tascar-data-set'

# pwd = '.../code/'
# CS_RENDER_PATH = pwd + 'audio/VEM-data/TASCAR/temp_cs/temp_render.wav'
# BG_RENDER_PATH = pwd + 'audio/VEM-data/TASCAR/temp_bg/temp_render.wav'
# TEST_RENDER_PATH = pwd + 'audio/VEM-data/TASCAR/temp_test/temp_render.wav'
# OUTPUT_FOLDER = pwd + 'audio/VEM-data/TASCAR/output_folder'
def _map_evaluate(clean_omit, background, X_L, X_res, fs, extended: bool = True):
    """
    Method that maps sound separations to either speech or background class, used in get_results().

    Parameters
    ---------
    clean_omit: The clean speech which has omitted the first OMIT_NUMBER samples.
    background: The background audio, of same size as clean_omit.
    X_L: The left input channel of the mixed sound before BSS. 
    X_res: The separated sources from FastICA, the two channels are to be determined which contain speech.
    fs: sampling rate needed for PyStoi. 
    extended: Boolean flag whether to use Extended STOI (ESTOI).
    
    Returns
    ---------
    stoi_res: STOI of separated speech.
    stoi_ref: STOI of reference input channel.
    backcorr_res: Pearson's correlation coefficient for separated background sound and ground truth.
    backcorr_ref: Pearson's correlation coefficient for input channel and ground truth.
    """
    
    def pcorrcoeff(truth, sample, scaling: bool = False):
        if scaling:
            truth = truth / np.max(np.abs(truth))
            sample = sample * np.max(np.abs(truth)) / np.max(np.abs(sample))

        a = truth - np.mean(truth)
        b = sample - np.mean(sample)
        # Noted that background sounds sometimes were flipped, i.e separated_background = -1 * background_truth.
        # So I decided to make the assumption that if correlation is negative after finding channel most likely to be speech, 
        # then we have the scenario above.
        return np.abs((np.sum( np.dot(a,b))) / (np.linalg.norm(a)*np.linalg.norm(b))) 
    
    # Get reference STOI from left channel
    stoi_ref = stoi(clean_omit, X_L, fs, extended)
    # Get reference abs of Pearson correlation coefficient
    backcorr_ref = pcorrcoeff(background, X_L)
    # Find channel in X_with highest STOI
    stoi_ch1 = max(stoi(clean_omit, X_res[:,0], fs, extended) , stoi(clean_omit, -X_res[:,0], fs, extended))
    stoi_ch2 = max(stoi(clean_omit, X_res[:,1], fs, extended) , stoi(clean_omit, -X_res[:,1], fs, extended))
    speech_channel_index = np.argmax([stoi_ch1, stoi_ch2])    
    if speech_channel_index == 0:
        stoi_res = stoi_ch1
        backcorr_res = pcorrcoeff(background, X_res[:, 1])
    else:
        stoi_res = stoi_ch2
        backcorr_res = pcorrcoeff(background, X_res[:, 0])
    
    return stoi_res, stoi_ref, backcorr_res, backcorr_ref


def get_results(CLEAN_SPEECH_PATH: str = '/clean-speech', scene: str = 'scene-1'):
    """
    Get results from performing speech separation of TASCAR-generated audio scenes.

    Parametres:
    -----------
    cs_folder : str
        The path to clean-speech folder.

    Saves:
    -----------
    Two pickles of results are written.
    """

    # Initialise TASCAR file rendering
    tsc = tascarpy()
    ica = FastICA(n_components=2,
                  fun = 'logcosh')
    
    # Fetch clean speech names
    speech_filenames = np.array(os.listdir(CLEAN_SPEECH_PATH))
    N = speech_filenames.shape[0]

    # Pre-allocate result vectors
    stois_res = np.zeros(N)
    stois_ref = np.zeros(N)
    intactness_res = np.zeros(N)
    intactness_ref = np.zeros(N)

    # Fetch background ground_truth which is constant while speeches differ
    bg, _ = tsc.tascar_render_ground_truth_bg()
    for i, speech_name in enumerate(speech_filenames):

        # Fetch current test samples
        cs, mix = tsc.get_test_sample(CLEAN_SPEECH_PATH + '/' + speech_name)

        print('\nworking on ', scene,' ', bg.dtype,' - ', speech_name, ' ', cs.dtype, '...')

        # Perform ICA on mixed audio scene
        X_res = ica.fit_transform(PCA().fit_transform(mix))

        # Write separation results as wavs to folder
        # write(WRITE_TO_PATH+'/reception/separated-50-60-Windy-birds/'+speech_name[:-4]+'-ch1.wav', rate = 48000, data = X_res[:,0] )
        # write(WRITE_TO_PATH+'/reception/separated-50-60-Windy-birds/'+speech_name[:-4]+'-ch2.wav', rate = 48000, data = X_res[:,1] )
        # Map result from ICA and evaluate accordingly
        container = _map_evaluate(cs.flatten(), bg[:cs.shape[0],0], mix[:,0], X_res, fs=48000, extended= True)
        stois_res[i] = container[0]
        stois_ref[i] = container[1]
        intactness_res[i] = container[2]
        intactness_ref[i] = container[3]

    return speech_filenames, scene, stois_res, stois_ref, intactness_res, intactness_ref

def build_df(speech_filenames, stois_res, stois_ref, intactness_res, intactness_ref):
    
    print('-----------------------', '\nBuilding pickle....')
    
    d = {
        'filenames' : speech_filenames,
        'Res. STOI SNR 0dB' : stois_res,  
        'Ref. STOI SNR 0dB' : stois_ref,
    }

    df = pd.DataFrame(data=d)
    
    d2 = {
        'filenames' : speech_filenames,
        'Res. BI SNR 0dB' : intactness_res,  
        'Ref. BI SNR 0dB' : intactness_ref,
    }
    
    df2 = pd.DataFrame(data=d2)

    return df, df2

def build_pickle(path: str, df, df2):
    print('-----------------------', '\nPickling STOI Data....')
    pd.to_pickle(df,path + '-STOI-results.pkl' )
    print('Data pickled!\n-----------------------')
    print('-----------------------', '\nPickling BI Data....')
    pd.to_pickle(df2,path + '-BI-results.pkl')
    print('Data pickled!\n-----------------------')

def main():
    """
    Perform Voice Extraction (VE) with FastICA and save results of TASCAR scenes
    """
    # Start the stopwatch / counter
    t1_start = perf_counter()     

    speech_filenames, scene, stois_res, stois_ref, intactness_res, intactness_ref = get_results()
    
    dir_name = WRITE_TO_PATH + '/' + scene
    mkdir(dir_name) if not exists(dir_name) else print('Putting results in ', dir_name)
    print('Putting results in ', dir_name, '...')
    data_frame, data_frame_2 = build_df(speech_filenames, stois_res, stois_ref, intactness_res, intactness_ref )
    build_pickle(dir_name + '/' + scene, data_frame, data_frame_2)
    
    # Stop the stopwatch / counter
    t1_stop = perf_counter() 
    print("Elapsed time (s): ",t1_stop-t1_start)

if __name__ == '__main__':
    main()