import os
from nltk import sent_tokenize
import whisperx
import pysrt
from pydub import AudioSegment
from pathlib import Path
import shutil
import inflect
import contractions

ASR_MODEL_ARCH = "large-v2"
DEF_BASE_NAME = "segments"


def process_audio(path, process_batch=False, buffer=100, clean_dir=False, generate_metadata=True):

    setup_training_dir(path, process_batch, clean_dir)

    start_idx = 0
    base_name = DEF_BASE_NAME

    for audio_file in sorted(os.listdir(path)):
        if audio_file.endswith(".wav"):
            audio_file = os.path.join(path, audio_file)
            print(f"processing {audio_file}")
            sentences = transcribe_audio(audio_file)

            if process_batch is False:
                base_name = _get_name(audio_file)

            segment_audio(audio_file, sentences, base_name, buffer, start_idx, generate_metadata)

            if process_batch is True:
                start_idx += len(sentences["segments"])


def transcribe_audio(path):

    asr_model = whisperx.load_model(ASR_MODEL_ARCH, device="cuda", compute_type="float16")

    audio = whisperx.load_audio(path)
    transcription = asr_model.transcribe(audio, batch_size=16)

    alg_model, metadata = whisperx.load_align_model(language_code=transcription["language"], device="cuda")
    aligned_transcription = whisperx.align(
        transcription["segments"], alg_model, metadata, audio, device="cuda", return_char_alignments=False)

    return aligned_transcription


def setup_training_dir(path, process_batch, clean_dir):

    training_dir = Path(path)

    if clean_dir:

        print(f"cleaning training directory...")

        for item in training_dir.iterdir():
            if item.is_file() and not item.name.endswith(".wav"):
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    if process_batch is False:

        for file in os.listdir(path):

            if file.endswith(".wav"):

                base_name = _get_name(file)
                os.makedirs(f"data/{base_name}", exist_ok=True)
                print(f"initialized training directory: data/{base_name}")

    else:
        base_name = DEF_BASE_NAME
        os.makedirs(f"data/{base_name}", exist_ok=True)
        print(f"initialized training directory: data/{base_name}")


def segment_audio(path, sentences, base_name, buffer, start_idx, generate_metadata):

    audio = AudioSegment.from_file(path)
    srt = pysrt.SubRipFile()

    metadata = []

    for idx, sentence in enumerate(sentences["segments"]):

        start_time = sentence["start"] * 1000
        end_time = sentence["end"] * 1000 + buffer

        start = pysrt.SubRipTime(milliseconds=start_time)
        end = pysrt.SubRipTime(milliseconds=end_time)
        subtitle = pysrt.SubRipItem(index=idx, start=start, end=end, text=sentence["text"])
        srt.append(subtitle)
        segment = audio[start_time:end_time]

        segment.export(os.path.join(f"data/{base_name}", f"segment_{idx + start_idx}.wav"), format="wav")

        if generate_metadata:

            segment_name = f"segment_{idx + start_idx}.wav"

            metadata.append(f"{segment_name}|{sentence['text']}|{_expand_transcriptions(sentence['text'])}")

        start_time = end_time
    srt_name = f"data/{_get_name(path)}.srt"
    # srt.save(srt_name, encoding = "utf-8")

    if generate_metadata:
        with open(f"data/metadata.txt", "a") as meta_file:
            for line in metadata:
                meta_file.write(line + "\n")


def _expand_transcriptions(sentence):

    expander = inflect.engine()
    words = sentence.split()
    for word in range(len(words)):
        if words[word].isdigit():
            words[word] = expander.number_to_words(words[word])

    expanded_numbers = " ".join(words)

    expanded_contractions = contractions.fix(expanded_numbers)

    return expanded_contractions


def _get_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def main():

    process_audio("data", process_batch=True, clean_dir=True, generate_metadata=True)


if __name__ == "__main__":
    main()
