import torch, torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio
from transformers import ASTFeatureExtractor, ASTForAudioClassification
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set the GPU device to use

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1.  Dataset & extractor
ds = load_dataset("ashraq/esc50", split="train").cast_column("audio", Audio(16_000))
extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

def prepare(batch):
    x = extractor(batch["audio"]["array"], sampling_rate=16_000,
                  return_tensors="pt")
    batch["input_values"] = x.input_values[0]
    batch["labels"] = batch["target"]
    return batch

ds = ds.map(prepare, remove_columns=ds.column_names)
loader = DataLoader(ds, batch_size=16, shuffle=True,
                    collate_fn=lambda b: {"input_values": torch.stack([i["input_values"] for i in b]),
                                          "labels": torch.tensor([i["labels"] for i in b])})
batch = next(iter(loader))
batch = {k: v.to(device) for k, v in batch.items()}
# 2.  Model & optimizer
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50).to(device)



out = model(**batch)

print(f"output keys {out.keys()}")
print(f"output: {out}")