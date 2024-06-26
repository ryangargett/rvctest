import os
from TTS.api import TTS 

def get_files(path):
    
    files = []
    
    for file in os.listdir(path):
        if file.endswith(".wav"):
             files.append(os.path.join(path, file))
             
    return files

def main():

    tts = TTS("xtts", gpu=True)
    
    files = get_files("/home/ryan/projects/RVCTest/data/segments")
    
    print(f"files: {files}")

    tts.tts_to_file(text = "This is a test input message. I am attempting to condition this model on a british female voice with a happy emotional context embedding. How did I do?",
                    file_path = "output.wav",
                    speaker_wav = ["/home/ryan/projects/RVCTest/data/Georgia - Happiness - Mild.post.wav", "/home/ryan/projects/RVCTest/data/Georgia - Happiness - Moderate - v2.post.wav", "/home/ryan/projects/RVCTest/data/Georgia - Happiness - Moderate.post.wav"],
                    language = "en",
                    split_sentences = False,
                    emotion = "excitement")

if __name__ == "__main__":
    main()