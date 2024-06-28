import os, shutil, numpy as np, soundfile as sf, subprocess
from pydub import AudioSegment

# Manually configure this. 
pwd = '.../code/'
CS_RENDER_PATH = pwd + '.../TASCAR/temp_cs/temp_render.wav'
BG_RENDER_PATH = pwd + '.../TASCAR/temp_bg/temp_render.wav'
TEST_RENDER_PATH = pwd + '.../TASCAR/temp_test/temp_render.wav'
OUTPUT_FOLDER = pwd + '.../TASCAR/output_folder'
TASCAR_SCENES_PATH = pwd + '.../TASCAR/scenes'

class tascarpy:
    """
    Quick Python class to generate data from TASCAR to be used in simulation. 
    
    The internal structure is setup to render temporary files of ground truth and the mix at an OUTPUT_FOLDER. 
    Parameters that configures the TASCAR scenes themselves need to be manually edited in the corresponding .tsc-file. 
    
    Figure 5 ('STOImethodology.png') in Master's Thesis motivates the internal structure of this class.

    --------
    EXAMPLE USAGE:
    See run-VE-test-2.py  
    """
    def __init__(self, pwd = pwd, CS_RENDER_PATH = CS_RENDER_PATH, BG_RENDER_PATH = BG_RENDER_PATH, TEST_RENDER_PATH= TEST_RENDER_PATH , OUTPUT_FOLDER = OUTPUT_FOLDER, TASCAR_SCENES_PATH = TASCAR_SCENES_PATH) -> None:
        self.pwd = pwd
        self.CS_RENDER_PATH = CS_RENDER_PATH
        self.BG_RENDER_PATH = BG_RENDER_PATH
        self.TEST_RENDER_PATH = TEST_RENDER_PATH
        self.OUTPUT_FOLDER = OUTPUT_FOLDER
        self.TASCAR_SCENES_PATH = TASCAR_SCENES_PATH
        

    def tascar_renderfile(self, wav_in_path='wav', wav_out_path='wav_tascar', tascar_scenes_path='TASCAR/scenes', moving_scene = -1, reset=False):
        """
        Essentially a set-up method.
        """
        if reset:
            if os.path.exists(wav_out_path):
                shutil.rmtree(wav_out_path)
            os.makedirs(wav_out_path)

        if moving_scene > 0: # specifics of my scenes, had to mkdir
            wav_path_long = 'wav_long/'
            if os.path.exists(wav_path_long):
                shutil.rmtree(wav_path_long)
            os.makedirs(wav_path_long)
            wav_files = os.listdir(wav_in_path)
            for i, file in enumerate(wav_files):
                input_file = os.path.abspath(wav_in_path + '/' + file)
                audio, sample_rate = sf.read(input_file)
                audio_length = len(audio)
                moving_scene_length = sample_rate * moving_scene
                audio_new = audio
                while audio_length + len(audio) < moving_scene_length:
                    audio_new = np.append(audio_new, audio)
                    audio_length = len(audio_new)
                sf.write(wav_path_long + file, audio_new, sample_rate)

        print('\n--- RENDERING WAV-FILES TO TASCAR SCENES ---')

        for scene in os.listdir(tascar_scenes_path):

            if os.path.isfile(tascar_scenes_path + '/' + scene):

                current_renderings = os.listdir(wav_out_path)
                if scene.replace('.tsc', '') not in current_renderings:

                    print('Rendering Scene: ' + scene)

                    tascar_file = os.path.abspath(tascar_scenes_path + '/' + scene)
                    scene_wav_path_out = wav_out_path + '/' + scene.replace('.tsc', '')
                    if os.path.exists(scene_wav_path_out):
                        shutil.rmtree(scene_wav_path_out)
                    os.makedirs(scene_wav_path_out)

                    wav_files = os.listdir(wav_in_path)
                    for i, file in enumerate(wav_files):
                        if moving_scene > 0:
                            input_file = os.path.abspath(wav_path_long + '/' + file)
                        else:
                            input_file = os.path.abspath(wav_in_path + '/' + file)
                        output_file = os.path.abspath(scene_wav_path_out + '/' + file)
                        print('[' + str(i + 1) + '/' + str(len(wav_files)) + '] - Rendering ' + file)
                        subprocess.run(['tascar_renderfile', '-i', input_file,'-o', output_file,'-f', str(moving_scene), tascar_file])

                    print('Done! Tascar scene wav-files has been saved to folder "' + scene_wav_path_out + '"')

    def put_new_temp_cs(self,path_to_wav: str):
        new_render = AudioSegment.from_wav(path_to_wav)
        target_dBFS = -30 
        delta = target_dBFS - new_render.dBFS
        current_speech_render, fs = self.pydub_to_np(new_render+delta)
        sf.write(CS_RENDER_PATH, current_speech_render, fs)
        return current_speech_render, fs

    def pydub_to_np(self, audio):
        """
        Converts pydub audio segmetn into np.float64 of shape [duration_in_seconds*sample_rate, channels],
        where each value is in range [-1.0, 1.0]. 
        Returns tuple (audio_np_array, sample_rate).
        """
        return np.array(audio.get_array_of_samples(), dtype=np.float64).reshape((-1, audio.channels)) / (
                1 << (8 * audio.sample_width - 1)), audio.frame_rate

    def tascar_render_ground_truth_cs(self,
            wav_in_path='audio/VEM-data/ground_truth/clean-speech/' + 'female_speech_09.wav', # example
            scene = 'reception'):
        
        # Generate ground_truth speech
        clean_speech, fs = self.put_new_temp_cs(wav_in_path) # put ground thuth
        subprocess.run([
            'tascar_renderfile', '-i', CS_RENDER_PATH,
            '-o', OUTPUT_FOLDER+'/cs.wav', 
            '-r', str(fs),
            '-u', str(clean_speech.shape[0]/48000),
            self.TASCAR_SCENES_PATH+'/'+scene+'_cs.tsc'
            ])
        return clean_speech, fs

    def tascar_render_ground_truth_bg(self,
            scene = 'reception'
            ): 
        
        # Generate ground truth background
        subprocess.run([
            'tascar_renderfile',
            '-o', OUTPUT_FOLDER+'/bg.wav', 
            '-r','48000',
            self.TASCAR_SCENES_PATH+'/'+scene+'_bg.tsc'
            ])
        return sf.read(self.OUTPUT_FOLDER+'/bg.wav')
        
    def tascar_render_mix(self,
            clean_speech,
            scene: str = 'reception', 
                        ):
        print('\n--- RENDERING MIX ---')
        # Generate test sample
        subprocess.run([
            'tascar_renderfile',
            '-o', OUTPUT_FOLDER+'/mix.wav', 
            '-r','48000',
            '-u', str(clean_speech.shape[0]/48000),
            self.TASCAR_SCENES_PATH +'/'+scene+'_test.tsc'
            ])
        return sf.read(OUTPUT_FOLDER+'/mix.wav')

    def get_test_sample(self,cs_sample_path: str):
        # Render ground truth and mix.
        print('\n--- RENDERING GROUND TRUTH ---')
        cs, fs = self.tascar_render_ground_truth_cs(cs_sample_path)
        mix, fs_mix = self.tascar_render_mix(cs)
        assert fs == fs_mix
        print('\n--- FINISHED RENDERING ---')
        return cs, mix

# if __name__ == '__main__':

# Render ground truth and mix.
# tsc = tascarpy(pwd,CS_RENDER_PATH, BG_RENDER_PATH, TEST_RENDER_PATH, OUTPUT_FOLDER, tascar_scenes_path= 'audio/VEM-data/TASCAR/scenes')
# tsc.get_test_sample('audio/VEM-data/ground_truth/clean-speech/' + 'female_speech_09.wav')
    
    
# Perform ICA and evaluate
#   from sklearn.decomposition import FastICA
#   ica = FastICA(n_components=2)
#   X_res = ica.fit_transform(mix)
#   container = _map_evaluate()

#   Open container and save results

#     """
#   tascar_renderfile -i .../TASCAR/temp_cs/temp_render.wav -o .../TASCAR/output_folder/female_speech_short.wav scene_test.tsc
#     """
