import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

import time

# Add here the xtts_config path
CONFIG_PATH = "/home/ryan/projects/RVCTest/out/GPT_XTTS_v2.0_GEORGIA_FT-April-06-2024_11+50PM-d6edb46/config.json"
# Add here the vocab file that you have used to train the model
TOKENIZER_PATH = "/home/ryan/projects/RVCTest/out/XTTS_v2.0_original_model_files/vocab.json"
# Add here the checkpoint that you want to do inference with
XTTS_CHECKPOINT = "/home/ryan/projects/RVCTest/out/GPT_XTTS_v2.0_GEORGIA_FT-April-06-2024_11+50PM-d6edb46/best_model.pth"
# Add here the speaker reference
SPEAKER_REFERENCE = ["/home/ryan/projects/RVCTest/georgia_ds/Georgia - Pride - v2.post.wav"]

# output wav path
OUTPUT_WAV_PATH = "xtts-ft.wav"

print("Loading model...")
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=XTTS_CHECKPOINT, vocab_path=TOKENIZER_PATH, use_deepspeed=False)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=SPEAKER_REFERENCE)

print("Inference...")

start_time = time.perf_counter()

out = model.inference(
    " You know, our team really is something else. We've accomplished feats that others could only dream of!",
    "en",
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.6,  # Add custom parameters here
)

end_time = time.perf_counter()

print(f"Elapsed time: {end_time - start_time} seconds")
torchaudio.save(OUTPUT_WAV_PATH, torch.tensor(out["wav"]).unsqueeze(0), 24000)
