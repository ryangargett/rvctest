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
    "amusement": ["georgia_ds/amuse_1.wav"],
    "compassion": ["compassion_1.wav", "compassion_2.wav"],
    "concern": ["concern_1.wav"],
    "curiosity": ["curiosity_1.wav", "curiosity_2.wav"],
    "empathy": ["empathy_1.wav", "empathy_2.wav", "empathy_3.wav"],
    "excitement": ["excite_1.wav", "excite_2.wav"],
    "happiness": ["georgia_ds/happy_2.wav", "georgia_ds/happy_3.wav"],
    "neutral": ["neutral_1.wav"],
    "pride": ["pride_1.wav", "pride_2.wav"],
    "supportive": ["support_1.wav", "support_2.wav"],
    "surprise": ["surprise_1.wav", "surprise_2.wav"]
}

# Model config
CONFIG_PATH = "out/GPT_XTTS_v2.0_GEORGIA_FT-April-06-2024_11+50PM-d6edb46/config.json"
TOKENIZER_PATH = "out/XTTS_v2.0_original_model_files/vocab.json"
XTTS_CHECKPOINT = "out/GPT_XTTS_v2.0_GEORGIA_FT-April-06-2024_11+50PM-d6edb46/best_model_2610.pth"
