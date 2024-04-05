import os
import pandas as pd

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from trainer import Trainer, TrainerArgs


def main():

    output_path = "/home/ryan/projects/RVCTest/data/segments/"

    dataset_config = BaseDatasetConfig(
        formatter="css10",
        meta_file_train="metadata.txt",
        path=output_path,
    )

    audio_config = VitsAudioConfig(
        sample_rate=22050, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
    )

    config = XttsConfig(
        audio=audio_config,
        run_name="xtts_ljspeech_ly",
        batch_size=16,
        eval_batch_size=16,
        batch_group_size=5,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        save_step=10,
        save_checkpoints=True,
        save_n_checkpoints=4,
        save_best_after=2000,
        text_cleaner="english_cleaners",
        use_phonemes=True,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=True,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        cudnn_benchmark=False,
    )

    processor = AudioProcessor.init_from_config(config)

    tokenizer, config = TTSTokenizer.init_from_config(config)

    train_data, val_data = load_tts_samples(
        dataset_config,
        eval_split=True
    )

    model = Vits(config, processor, tokenizer, speaker_manager=None)

    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_data,
        eval_samples=val_data
    )

    trainer.fit()


if __name__ == "__main__":
    main()
