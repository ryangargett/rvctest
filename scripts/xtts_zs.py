##################################################
# File: xtts_zs.py                                #
# Project: AdaptiveLLM                            #
# Created Date: Sun Apr 07 2024                   #
# Author: Ryan Gargett                            #
# -----                                           #
#Last Modified: Thu Apr 11 2024                  #
#Modified By: Ryan Gargett                       #
##################################################


import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

import utils.config as cfg

import time


def _get_model_config(config_path):
    config = XttsConfig()
    config.load_json(config_path)

    return config


def load_model(model_path, config_path, tokenizer_path):

    print(f"Loading model config....")
    config = _get_model_config(config_path)

    print(f"Loading model....")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=model_path, vocab_path=tokenizer_path, use_deepspeed=False)
    model.cuda()

    return model


def generate_style(model, style):

    if style not in cfg.STYLE_REFERENCE:
        raise ValueError(f"Style '{style}' not found in reference lookup.")

    reference = cfg.STYLE_REFERENCE[style]

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=reference)
    return gpt_cond_latent, speaker_embedding


def inference(text, model, gpt_cond_latent, speaker_embedding, track_performance=False, output_path="output.wav"):

    start_time = time.perf_counter()

    out = model.inference(
        text,
        "en",
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.65
    )

    end_time = time.perf_counter()

    if track_performance:
        print(f"Elapsed time: {end_time - start_time} seconds")

    torchaudio.save(output_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)


def main():

    style_name = "happiness"

    model = load_model(cfg.XTTS_CHECKPOINT, cfg.CONFIG_PATH, cfg.TOKENIZER_PATH)
    gpt_cond_latent, speaker_embedding = generate_style(model, style_name)

    # model performs significantly better with emotionally aligned sentence context
    text = f"Hello! This is a test generation, attempting to capture {style_name} style."

    num_tests = 5

    for test in range(num_tests):
        print(f"Running test {test+1} of {num_tests}")
        inference(text, model, gpt_cond_latent, speaker_embedding,
                  track_performance=True, output_path=f"output_{test}.wav")


if __name__ == "__main__":
    main()
