"""
    convert the .flac audio file from the original dataset to .wav format to use in the model
"""

from pathlib import PurePath
from pydub import AudioSegment
import os


rootDir = "VCTK_corpus/DS_10283_3443/VCTK-Corpus-0.92/wav48_silence_trimmed"
dirName, subdirList, _ = next(os.walk(rootDir))


targetDir = "./wavs"
for subdir in sorted(subdirList):
    print('Processing: %s' % subdir)
    # Create new directory if the directory doesn't exist
    if not os.path.exists(os.path.join(targetDir, subdir)):
        os.makedirs(os.path.join(targetDir, subdir))

    _,_, fileList = next(os.walk(os.path.join(dirName,subdir)))

    for filename in sorted(fileList):
        path = os.path.join(dirName,subdir,filename)
        print("Now converting: " + path)
        file_path = PurePath(path)
        flac_tmp_audio_data = AudioSegment.from_file(file_path, file_path.suffix[1:])
        target_path = str(os.path.join(targetDir, subdir, filename[:-4])) + "wav"
        flac_tmp_audio_data.export(target_path, format="wav")


