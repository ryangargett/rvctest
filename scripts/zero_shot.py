import os
from TTS.api import TTS 

def get_files(path):
    
    files = []
    
    for file in os.listdir(path):
        if file.endswith(".wav"):
             files.append(file)
             
    return files

def main():

    tts = TTS("xtts", gpu=True)
    
    files = get_files("/home/ryan/projects/RVCTest/data")

    tts.tts_to_file(text = "This is a test input message.",
                    file_path = "output.wav",
                    speaker_wav = "/home/ryan/projects/RVCTest/data/amusement_1.wav",
                    language = "en")

if __name__ == "__main__":
    main()