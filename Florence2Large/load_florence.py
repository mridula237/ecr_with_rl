import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .configuration_florence2 import Florence2Config
from .modeling_florence2 import Florence2ForConditionalGeneration
import json
import torch
from transformers import AutoProcessor


LABLE2ID = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}

ID2LABEL = {v: k for k, v in LABLE2ID.items()}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_FILE = "Florence2large/config.json"
CHECKPOINT_PATH = "Florence2large/pytorch_model.bin"

pretrained_model_path="microsoft/Florence-2-base-ft"
model_revision='refs/pr/6'
new_token = '<EMOTION_DETECTION>'

def load_florence2_model(config_file: str = CONFIG_FILE, checkpoint_path: str = CHECKPOINT_PATH, device: str = DEVICE, include_audio: bool = False):
    # Load the configuration from the JSON file
    with open(config_file, "r") as f:
        config_dict = json.load(f)

    # Extract specific configs
    vision_config = config_dict.pop("vision_config")
    text_config = config_dict.pop("text_config")

    # Create the config with explicit parameters
    florence2_config = Florence2Config(
        vision_config=vision_config,
        text_config=text_config,
        **config_dict
    )

    florence2_config.add_cross_attention = True
    florence2_config.id2label = ID2LABEL
    florence2_config.label2id = LABLE2ID
    florence2_config.include_audio = include_audio
    # florence2_config.text_config.encoder_layers = 8 # default is 12
    # florence2_config.text_config.decoder_layers = 8 # default is 12
    # florence2_config.text_config.num_hidden_layers = 8 # default is 12 - setting this for double dipping
    # florence2_config.text_config.encoder_ffn_dim = 2048  # default is 4096
    # florence2_config.text_config.decoder_ffn_dim = 2048  # default is 4096
    # if include_audio:
    #     florence2_config.include_audio = True
    #     florence2_config.audio_config = config_dict.get("audio_config", {})

    florence2 = Florence2ForConditionalGeneration(config=florence2_config)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    florence2.load_state_dict(state_dict)

    processor = AutoProcessor.from_pretrained(
        pretrained_model_path,
        trust_remote_code=True,
        revision=model_revision,
    )
    
    tokenizer = processor.tokenizer

    if new_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': [new_token]})
        florence2.resize_token_embeddings(len(tokenizer))

    # ✅ Patch processor's task metadata
    if hasattr(processor, "tasks_answer_post_processing_type"):
        processor.tasks_answer_post_processing_type.update({
            new_token: 'pure_text'
        })

    if hasattr(processor, "task_prompts_with_input"):
        processor.task_prompts_with_input.update({
            new_token: "What is the emotion of the speaker that says {input}? Describe it in one word."
        })
    
    florence2.to(DEVICE)
    
    return florence2, processor