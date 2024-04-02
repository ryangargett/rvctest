import os
from nltk import sent_tokenize
import whisper
import pysrt
from pydub import AudioSegment

WORDS_PER_MINUTE = 150
ASR_MODEL_ARCH = "base"

def process_audio(path):
    
    asr_model = whisper.load_model(ASR_MODEL_ARCH)
    transcription = asr_model.transcribe(path)
    
    sentences = sent_tokenize(transcription["text"])
    
    segment_audio(path, sentences)
    
    
def segment_audio(path, sentences):
    
    audio = AudioSegment.from_file(path)
    srt = pysrt.SubRipFile()
    
    start_time = 0
    
    for idx, sentence in enumerate(sentences):
        
        num_words = len(sentence.split())
        duration = num_words / (WORDS_PER_MINUTE / (60 * 1000))
        end_time = start_time + duration
        
        start = pysrt.SubRipTime(milliseconds=start_time)
        end = pysrt.SubRipTime(milliseconds=end_time)
        subtitle = pysrt.SubRipItem(index=idx, start=start, end=end, text=sentence)
        srt.append(subtitle)
        
        segment = audio[start_time:end_time]
        segment.export(os.path.join("data/segments", f"segment_{idx}.wav"), format="wav")
        
        start_time = end_time
        
    srt_name = f"{os.path.splitext(os.path.basename(path))[0]}.srt"
    srt.save(srt_name, encoding = "utf-8")
    
    
def main():
    process_audio("data/neutral.wav")
    
if __name__ == "__main__":
    main()