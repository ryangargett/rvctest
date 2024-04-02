import os
from nltk import sent_tokenize
import whisperx
import pysrt
from pydub import AudioSegment

WORDS_PER_MINUTE = 150
ASR_MODEL_ARCH = "large-v2"

def process_audio(path):
    
    asr_model = whisperx.load_model(ASR_MODEL_ARCH, device = "cuda", compute_type = "float16")
    
    audio = whisperx.load_audio(path)
    transcription = asr_model.transcribe(audio, batch_size=16)
    
    alg_model, metadata = whisperx.load_align_model(language_code=transcription["language"], device="cuda")
    aligned_transcription = whisperx.align(transcription["segments"], alg_model, metadata, audio, device="cuda", return_char_alignments=False)
    
    print(aligned_transcription["segments"])
    
    
    
    #segment_audio(path, sentences)
    
    
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