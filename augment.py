import librosa
import soundfile as sf
from audiomentations import Compose,AddGaussianNoise,PitchShift,HighPassFilter,PolarityInversion
augment=Compose([
    AddGaussianNoise(min_amplitude=0.001,max_amplitude=0.005,p=0.3),
    PitchShift(min_semitones=-8,max_semitones=8,p=0.6),
    PolarityInversion(p=0.9),
    HighPassFilter(p=0.5)
])
import os
# assign directory
directory = 'dataset/sad'

ac=89


# signal,sr=librosa.load("dataset/angry/angry 1.mp3")
# augmented_sig=augment(signal,sr)
# nameof_augfile="angry "+str(ac)
# ac=ac+1
# sf.write(nameof_augfile,augmented_sig,sr)
    
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    print(f)
    # checking if it is a file
    # if os.path.isfile(f):
    #     print(f)
    signal,sr=librosa.load(f)
    augmented_sig=augment(signal,sr)
    nameof_augfile="sad "+str(ac)+".mp3"
    ac=ac+1
    sf.write(nameof_augfile,augmented_sig,sr)
    if(ac==101):
         break



# signal,sr=librosa.load("dataset/angry/angry 7.mp3")
# sig=signal+signal
# sf.write("audio 7",sig,sr)
