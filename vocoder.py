import torch
import librosa      # Python library for analysing audio inputs 
import pickle
from synthesis import build_model
from synthesis import wavegen

spect_vc = pickle.load(open('results.pkl', 'rb'))
#device = torch.device("cuda")
#model = build_model().to(device)

device = torch.device("cpu")
model = build_model().to(device)
checkpoint = torch.load("checkpoints/Wavenet/checkpoint_step001000000_ema.pth")
model.load_state_dict(checkpoint["state_dict"])

for spect in spect_vc:
    name = spect[0]
    c = spect[1]
    print(name)
    waveform = wavegen(model, c=c)   
    librosa.output.write_wav(name+'.wav', waveform, sr=16000)