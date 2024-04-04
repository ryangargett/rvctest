import os

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.utils.audio.processor import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.xtts import Xtts

from trainer import Trainer, TrainerArgs

def main():
    
    output_path = "/home/ryan/projects/RVCTest/output/"

    dataset_config = BaseDatasetConfig(
        formatter = "ljspeech",
        meta_file_train = "/home/ryan/projects/RVCTest/data/metadata.txt",
        path = "/home/ryan/projects/RVCTest/data/",
    )
    
    config = XttsConfig(
        batch_size=64,
        eval_batch_size=16,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        precompute_num_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner="phoneme_cleaners",
        use_phonemes=True,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        print_step=25,
        print_eval=False,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        min_text_len=0,
        max_text_len=500,
        min_audio_len=0,
        max_audio_len=500000,
)
    
    processor = AudioProcessor.init_from_config(config)
    
    tokenizer, cfg = TTSTokenizer.init_from_config(config)
    
    train_data, val_data = load_tts_samples(
        dataset_config,
        eval_split = True
    )
    
    model = Xtts(cfg, processor, tokenizer, speaker_manager = None)
    trainer = Trainer(
        TrainerArgs(),
        cfg,
        output_path,
        model=model,
        train_samples=train_data,
        eval_samples=val_data
    )
    
    trainer.fit()
    
    

if __name__ == "__main__":
    main()