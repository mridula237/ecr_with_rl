
# test_rl_pipeline.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch.utils.data import DataLoader
from src.MELD import MELD, collate_fn
import functools
from torchvision.transforms import v2
from transformers import AutoProcessor, AutoModelForCausalLM

# Define class weights #from the MELD dataset github repo: https://github.com/declare-lab/MELD?tab=readme-ov-file#class-weights
CLASS_WEIGHTS = {
    "neutral": 4.0,
    "surprise": 15.0,
    "fear": 15.0,
    "sadness": 3.0,
    "joy": 1.0,
    "disgust": 6.0,
    "anger": 3.0,
}

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
transform = v2.RandomApply([
    v2.RandomHorizontalFlip(p=1.0),
    v2.RandomRotation(degrees=15),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
], p=0.5)

florence2 = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-base-ft",
    revision='refs/pr/6',
    trust_remote_code=True,
    output_hidden_states=True,
    return_dict=True,
)
florence2.to(DEVICE)


processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base-ft",
    trust_remote_code=True,
    revision='refs/pr/6',
)

print(f"florence2 config: {florence2.config}")

meld_collate_fn = functools.partial(collate_fn, processor=processor)


def test_inference():

    # test_data = MeldDataset(folder_path="/data/shared/meld/processed_videos_test",transform=transform)
    test_data = MELD(folder_path="/data/shared/meld/meldraw/train_processed",transform=transform)

    test_loader = DataLoader(
        test_data,
        batch_size=1,
        collate_fn=meld_collate_fn,
        shuffle=False
    )
    
    #check feature extractor
    batch = next(iter(test_loader))
    print(f"batch type: {type(batch)}")
    print(f"batch : {batch}")
    inputs, emotion, _ = batch
    print(f"attention mask shape: {inputs['attention_mask'].shape}, attention mask type: {inputs['attention_mask'].dtype}")
    unique_values = torch.unique(inputs['attention_mask'])
    print(f"Unique values in attention_mask: {unique_values}")
    
    inputs["decoder_input_ids"] = torch.zeros_like(inputs["input_ids"])
    # inputs["decoder_input_ids"] = decoder_input_ids=torch.zeros_like(inputs["input_ids"]).fill_(processor.tokenizer.pad_token_id)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    output = florence2(**inputs , return_dict=True)
    print(f"Florence2 output: {output}")
    print(f"Florence2 output keys: {output.keys()}")
    print(f"encoder last hidden state shape: {output.encoder_last_hidden_state.shape}")
    print(f"encoder last hidden[:,0] state shape: {output.encoder_last_hidden_state[:,0].shape}")

if __name__ == "__main__":
    test_inference()