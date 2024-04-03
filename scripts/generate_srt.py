import os
from nltk import sent_tokenize
import whisperx
import pysrt
from pydub import AudioSegment
import json

WORDS_PER_MINUTE = 150
ASR_MODEL_ARCH = "large-v2"
DEF_BASE_NAME = "segments"

def process_audio(path, process_batch = False, buffer = 0):
    
    base_name = setup_training_dir(path, process_batch)
    
    if process_batch:
        audio_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".wav")]
        combined_transcription = {"segments": []}
        
        for audio_file in audio_files:
            sentences = transcribe_audio(audio_file)
            combined_transcription["segments"].extend(sentences["segments"])

            segment_audio(audio_file, sentences, base_name, buffer)
        
    else:
        
        for audio_file in os.listdir(path):
            if audio_file.endswith(".wav"):
                audio_path = os.path.join(path, audio_file)
                sentences = transcribe_audio(audio_path)
                segment_audio(audio_path, sentences, base_name, buffer)
      

def transcribe_audio(path):
    
    asr_model = whisperx.load_model(ASR_MODEL_ARCH, device = "cuda", compute_type = "float16")
    
    audio = whisperx.load_audio(path)
    transcription = asr_model.transcribe(audio, batch_size=16)
    
    alg_model, metadata = whisperx.load_align_model(language_code=transcription["language"], device="cuda")
    aligned_transcription = whisperx.align(transcription["segments"], alg_model, metadata, audio, device="cuda", return_char_alignments=False)
    
    return aligned_transcription
    
    
def setup_training_dir(path, process_batch):
    
    if process_batch is False:
    
        for file in os.listdir(path):
            
            if file.endswith(".wav"):
                
                base_name = os.path.splitext(os.path.basename(path))[0]
                os.makedirs(f"data/{base_name}", exist_ok=True)
                print(f"initialized training directory: data/{base_name}")
    
    else:
        base_name = DEF_BASE_NAME
        os.makedirs(f"data/{base_name}", exist_ok=True)
        print(f"initialized training directory: data/{base_name}")
        
    return base_name
    
    
def segment_audio(path, sentences, base_name, buffer):
    
    audio = AudioSegment.from_file(path)
    srt = pysrt.SubRipFile()
    
    for idx, sentence in enumerate(sentences["segments"]):
        
        start_time = sentence["start"] * 1000
        end_time = sentence["end"] * 1000 + buffer
        
        start = pysrt.SubRipTime(milliseconds=start_time)
        end = pysrt.SubRipTime(milliseconds=end_time)
        subtitle = pysrt.SubRipItem(index=idx, start=start, end=end, text=sentence["text"])
        srt.append(subtitle)
        segment = audio[start_time:end_time]
        
        segment.export(os.path.join(f"data/{base_name}", f"segment_{idx}.wav"), format="wav")
        
        start_time = end_time
    srt_name = f"data/{base_name}.srt"
    srt.save(srt_name, encoding = "utf-8")
    
    
def main():
    
    process_audio("data", process_batch = True)
    
if __name__ == "__main__":
    main()