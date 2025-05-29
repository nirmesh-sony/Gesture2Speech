import os
import sys
sys.path = ["/hdd/lokesh/models/dubwise/DubWise_sourcecode/multi_modified_TTS"] + sys.path
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

#main moe
CONFIG_PATH = "/hdd/lokesh/models/dubwise/DubWise_sourcecode/multi_modified_TTS/recipes/ljspeech/xtts_v2/run/training/GPT_XTTS_v2.0_PATS_FT-April-09-2025_12+17PM-c99e885c/config.json"
TOKENIZER_PATH = "/hdd/lokesh/models/dubwise/DubWise_sourcecode/multi_modified_TTS/recipes/ljspeech/xtts_v2/run/training/XTTS_v2.0_original_model_files/vocab.json"
XTTS_CHECKPOINT = "/hdd/lokesh/models/dubwise/DubWise_sourcecode/multi_modified_TTS/recipes/ljspeech/xtts_v2/run/training/GPT_XTTS_v2.0_PATS_FT-April-09-2025_12+17PM-c99e885c/best_model_62250.pth"

# CONFIG_PATH = "/hdd/lokesh/models/dubwise/DubWise_sourcecode/multi_modified_TTS/recipes/ljspeech/xtts_v2/run/training/GPT_XTTS_v2.0_PATS_FT-April-02-2025_04+54PM-c99e885c/config.json"
# TOKENIZER_PATH = "/hdd/lokesh/models/dubwise/DubWise_sourcecode/multi_modified_TTS/recipes/ljspeech/xtts_v2/run/training/XTTS_v2.0_original_model_files/vocab.json"
# XTTS_CHECKPOINT = "/hdd/lokesh/models/dubwise/DubWise_sourcecode/multi_modified_TTS/recipes/ljspeech/xtts_v2/run/training/GPT_XTTS_v2.0_PATS_FT-April-02-2025_04+54PM-c99e885c/best_model_49800.pth"

#vanilla
# CONFIG_PATH = "/hdd/lokesh/models/dubwise/DubWise_sourcecode/multi_modified_TTS/recipes/ljspeech/xtts_v2/run/training/GPT_XTTS_v2.0_PATS_FT-March-02-2025_12+06PM-c99e885c/config.json"
# TOKENIZER_PATH = "/hdd/lokesh/models/dubwise/DubWise_sourcecode/multi_modified_TTS/recipes/ljspeech/xtts_v2/run/training/XTTS_v2.0_original_model_files/vocab.json"
# XTTS_CHECKPOINT = "/hdd/lokesh/models/dubwise/DubWise_sourcecode/multi_modified_TTS/recipes/ljspeech/xtts_v2/run/training/GPT_XTTS_v2.0_PATS_FT-March-02-2025_12+06PM-c99e885c/best_model_153550.pth"

#moe base
# CONFIG_PATH = "/hdd/lokesh/models/dubwise/DubWise_sourcecode/multi_modified_TTS/recipes/ljspeech/xtts_v2/run/training/GPT_XTTS_v2.0_PATS_FT-April-01-2025_09+33AM-c99e885c/config.json"
# TOKENIZER_PATH = "/hdd/lokesh/models/dubwise/DubWise_sourcecode/multi_modified_TTS/recipes/ljspeech/xtts_v2/run/training/XTTS_v2.0_original_model_files/vocab.json"
# XTTS_CHECKPOINT = "/hdd/lokesh/models/dubwise/DubWise_sourcecode/multi_modified_TTS/recipes/ljspeech/xtts_v2/run/training/GPT_XTTS_v2.0_PATS_FT-April-01-2025_09+33AM-c99e885c/best_model_166000.pth"

#hmoe
# CONFIG_PATH = "/hdd/lokesh/models/dubwise/DubWise_sourcecode/multi_modified_TTS/recipes/ljspeech/xtts_v2/run/training/GPT_XTTS_v2.0_PATS_FT-April-25-2025_09+27AM-c99e885c/config.json"
# TOKENIZER_PATH = "/hdd/lokesh/models/dubwise/DubWise_sourcecode/multi_modified_TTS/recipes/ljspeech/xtts_v2/run/training/XTTS_v2.0_original_model_files/vocab.json"
# XTTS_CHECKPOINT = "/hdd/lokesh/models/dubwise/DubWise_sourcecode/multi_modified_TTS/recipes/ljspeech/xtts_v2/run/training/GPT_XTTS_v2.0_PATS_FT-April-25-2025_09+27AM-c99e885c/best_model_128650.pth"

print("Loading model...")
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=XTTS_CHECKPOINT, vocab_path=TOKENIZER_PATH, use_deepspeed=False)
model.cuda()

print("Computing speaker latents...")
# SPEAKER_REFERENCE = "/hdd5/neha/neha_s/hindi_data/wavs/pSpeaker5/Speaker5_Happy_2_457.wav"

# SPEAKER_REFERENCE = "/hdd/lokesh/models/dubwise/DubWise_sourcecode/multi_modified_TTS/recipes/ljspeech/xtts_v2/SOTA/GT/1.wav"
SPEAKER_REFERENCE = "/hdd/lokesh/models/dubwise/DubWise_sourcecode/multi_modified_TTS/recipes/ljspeech/xtts_v2/SOTA/GT/3.wav"

# SPEAKER_REFERENCE = "/hdd5/neha/neha_s/TTS/recipes/ljspeech/xtts_v2/HBKJ_samples_gen/jordar_ref_eng2_norm.wav"

motion_feat = '/hdd/lokesh/models/datasets/PATS/pats/data/motion_features1/chemistry/chemistry_67811.npy'
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=SPEAKER_REFERENCE, motion_feat=motion_feat)

# gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=SPEAKER_REFERENCE)

OUTPUT_WAV_PATH = "/hdd/lokesh/models/dubwise/DubWise_sourcecode/multi_modified_TTS/recipes/ljspeech/xtts_v2/github_inference_samples/SOTA/moe2/smoe3.wav"

print("Inference...")
out = model.inference(
# "So if you know few bond energies, you can calculate the entropy for lot of reactions.",
# "So you can just, instead of looking at 10 to the minus 10 you can say what's the.",
"So I can measure current flow in Ams in an electronic cell that forces galvanic cell.",
"en",
gpt_cond_latent,
speaker_embedding,
# diffusion_conditioning,
temperature=0.7, #.7, # Add custom parameters here
top_p=0.1,
# avfeat='/hdd5/neha/neha_s/lip2wav_chemistry_dataset/chem_release_version/videoAVHubert_feats/avfeats_renamed/{}.npy'.format("jGRhNqkYK18_051"),
posefeat = '/hdd/lokesh/models/datasets/PATS/pats/data/pose_features2/chemistry/chemistry_67811.npy',
# avfeat = '/hdd/lokesh/models/datasets/PATS/pats_experiments/resized_oliver_102265.npy',
)
torchaudio.save(OUTPUT_WAV_PATH, torch.tensor(out["wav"]).unsqueeze(0), 24000)
