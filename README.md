# The AutoVC model to solve the voice conversion problem  
This is our implementation of the AutoVC model  
For more information, please checkout the original paper and GitHub repository
(links below)
***
# Desciptions 
Here, we adopt the AutoVC model to solve the many-to-many as well as the zero-shot voice conversion problem  
If you want to train the model, please checkout the **Instructions** below  
The purpose of each file in our repository will be described as follow  
## Directories
|Name                |Description                                                        |
|--------------------|-------------------------------------------------------------------|
|VTCK_Corpus_subset  |A subset of .flac audio files downloaded from the original dataset |
|wavs                |.wav audio files converted from the .flac audio files              |
|spmels_16khz        |Mel-spectrograms converted from .wav audio files                   |
## Files
|Name                |Description                                                        |
|--------------------|-------------------------------------------------------------------|
|Demo.ipynb          |Model's demo                                                       |
|Train.ipynb         |Model's training code                                              |
|data_loader.py      |Code to create custom Pytorch's dataloaders                        |
|flac_to_wav.py      |Convert .flac audio files to .wav audio files                      |
|hparams.py          |Hyperparameters for the Wavenet Vocoder                            |
|make_metadata.ipynb |Create metadata, which makes training become more efficient        |
|make_spectrogram.py |Convert .wav audio files to mel-spectrograms                       |
|model.py            |The AutoVC's architecture                                          |
|speaker_encoder.py  |The Speaker Encoder module's architecture                          |
|synthesis.py        |Generate .wav audio files using the Wavenet Vocoder network        |
***
# Instructions
If you want to train the model with your own dataset, please follow these instructions  
> **_NOTE:_**  The *.ipynb* files are created to run in Google Colab, if you want to run them locally, modify these files 

**Data preprocessing** 
- [ ] Change the directory in the *flac_to_wav.py* to your directory if your dataset comes in .flac format
- [ ] Run the *make_spectrogram.py*, the results will be saved in the spmels_16khz directory
- [ ] Run the *make_metadata.ipynb*, the results will be saved in the spmels_16khz directory under *train.pkl*  

**Training**
- [ ] Create a folder called "checkpoints"
- [ ] Create subfolders called "AutoVC", "AutoVC_custom_16khz", "Wavenet" "Speaker"
- [ ] Download the corresponding checkpoints, if you want to use them
- [ ] Run the *Train.ipynb*, the checkpoints will be saved in the "AutoVC_custom_16khz" folder  

**Demo**
- [ ] Modify and run the *Demo.ipynb* 
***
# References

**Checkpoints**
|AutoVC  |AutoVC_custom_16khz|Speaker |Wavenet |
|--------|-------------------|--------|--------|
|[link](https://drive.google.com/file/d/1dFQBhnsjIJdfkrQRXP48EgR56dypAA6h/view?usp=sharing)|[link](https://drive.google.com/file/d/1-wJokreO9282H2jZqUKhuhbo8vnEkLi7/view?usp=sharing)|[link](https://drive.google.com/file/d/1NUKlkvj8UERblp3XWksAeF7ORlQTjxAp/view?usp=sharing)|[link](https://drive.google.com/file/d/1A9IFQ-SBWwl2P1FVjODZVNIPUW7vA8S_/view?usp=sharing)|

**Original paper:**
[AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss](https://arxiv.org/abs/1905.05879)  

**Author's GitHub Repository:**
[link](https://github.com/auspicious3000/autovc.git)  





