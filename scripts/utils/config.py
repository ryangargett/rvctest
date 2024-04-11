##################################################
# File: config.py                                 #
# Project: AdaptiveLLM                            #
# Created Date: Thu Apr 11 2024                   #
# Author: Ryan Gargett                            #
# -----                                           #
#Last Modified: Thu Apr 11 2024                  #
#Modified By: Ryan Gargett                       #
##################################################

STYLE_REFERENCE = {
    "compassion": ["georgia_ds/compassion_1.wav", "georgia_ds/compassion_2.wav"],
    "concern": ["georgia_ds/concern_1.wav"],
    "curiosity": ["georgia_ds/curiosity_1.wav", "georgia_ds/curiosity_2.wav"],
    "empathy": ["georgia_ds/empathy_1.wav", "georgia_ds/empathy_2.wav", "georgia_ds/empathy_3.wav"],
    "excitement": ["georgia_ds/excite_1.wav", "georgia_ds/excite_2.wav"],
    "happiness": ["georgia_ds/happy_2.wav", "georgia_ds/happy_3.wav"],
    "neutral": ["georgia_ds/neutral_1.wav"],
    "pride": ["georgia_ds/pride_1.wav", "georgia_ds/pride_2.wav"],
    "supportive": ["georgia_ds/support_1.wav", "georgia_ds/support_2.wav"],
    "surprise": ["georgia_ds/surprise_1.wav", "georgia_ds/surprise_2.wav"]
}

# Model config
CONFIG_PATH = "out/GPT_XTTS_v2.0_GEORGIA_FT-April-06-2024_11+50PM-d6edb46/config.json"
TOKENIZER_PATH = "out/XTTS_v2.0_original_model_files/vocab.json"
XTTS_CHECKPOINT = "out/GPT_XTTS_v2.0_GEORGIA_FT-April-06-2024_11+50PM-d6edb46/best_model_2610.pth"
