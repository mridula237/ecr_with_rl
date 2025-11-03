import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from natsort import natsorted
from transformers import AutoProcessor, Wav2Vec2FeatureExtractor
import torchvision.transforms.v2 as T
import soundfile as sf

# ✅ Step 1: Define Image Transformations
transform = T.Compose([
    T.ToImage(),
    T.Resize((224, 224)),
    T.ToDtype(torch.float32),
])

# ✅ Step 2: Load Processor Once
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base-ft",
    trust_remote_code=True,
    revision='refs/pr/6'
)
wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-960h")

# ✅ Step 3: MELD Dataset (Utterance-Level Batching)
class MELD(Dataset):
    def __init__(self, folder_path, transform=None):
        self.utterances = []
        self.transform = transform

        # Collect all JSON files
        all_json_files = [
            os.path.join(root, file_name)
            for root, _, files in os.walk(folder_path)
            for file_name in files if file_name.endswith("_paired_data.json")
        ]
        all_json_files = natsorted(all_json_files)

        # Load utterances
        for file_path in all_json_files:
            with open(file_path, 'r') as f:
                utterances = json.load(f)
                self.utterances.extend(utterances)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utterance = self.utterances[idx]

        if utterance["emotion"] not in ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]:
            raise ValueError(f"Invalid emotion '{utterance['emotion']}' in utterance {idx}. Expected one of ['neutral', 'joy', 'sadness', 'anger', 'surprise', 'fear', 'disgust'].")

        image_path = os.path.normpath(utterance["image"]).replace("\\", "/")
        # image_path = os.path.join(os.path.dirname("/data/shared/meld/meldraw/"), image_path)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        audio_path = os.path.normpath(utterance["audio"]).replace("\\", "/")
        waveform, sampling_rate = sf.read(audio_path, dtype='float32', always_2d=True)
        mono = waveform.mean(axis=1)
        

        image = Image.open(image_path).convert("RGB")
        return image, utterance["text"], utterance["emotion"], mono


# ✅ Step 4: Optimized collate_fn
def collate_fn(batch, processor=processor, wav_feature_extractor=wav_feature_extractor):
    # Remove None values (for missing samples)
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        raise ValueError("Batch is empty after filtering. Ensure valid data is provided.")

    images, texts, emotions, audio = zip(*batch)
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, do_rescale=True)
    audio_inputs = wav_feature_extractor(audio, padding=True, return_tensors="pt", sampling_rate=16000)['input_values']


    return inputs, emotions, audio_inputs
