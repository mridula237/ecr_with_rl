from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Wav2Vec2 components (tokenizer not used unless transcript is involved)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-large-960h",
    output_hidden_states=True,
    output_attentions=True,
    return_dict=True,
    
)
model.eval()

# === Step 1: Load and trim/resample audio ===
audio_path = "/data/shared/meld/meldraw/train_processed/dia0_utt0/dia0_utt0.wav"
waveform, sampling_rate = sf.read(audio_path, dtype='float32', always_2d=True)

print("Original waveform shape:", waveform.shape)
print("Sample rate:", sampling_rate)

# Convert stereo to mono
if len(waveform.shape) == 2:
    print("Stereo detected. Converting to mono.")
    mono = waveform.mean(axis=1)

# Normalize and convert dtype
waveform = waveform.astype(np.float32).flatten()

# Truncate or pad to exactly 1024 samples
TARGET_LENGTH = 1024
if len(waveform) > TARGET_LENGTH:
    waveform = waveform[:TARGET_LENGTH]
else:
    waveform = np.pad(waveform, (0, TARGET_LENGTH - len(waveform)))

print(f"Final waveform shape: {waveform.shape}")

# === Step 2: Feature extraction ===
inputs = feature_extractor(mono, sampling_rate=16000, return_tensors="pt", )
print("inputs['input_values'] shape:", inputs["input_values"].shape)

outputs = model(**inputs)

# === Step 4: Extract outputs ===
last_hidden_state = outputs.last_hidden_state
extract_features = outputs.extract_features   
attentions = outputs.attentions

print("\n📊 Output Summary:")
print(f"last_hidden_state shape: {last_hidden_state.shape}")
print(f"extract_features shape: {extract_features.shape}")
print(f"Number of attention layers: {len(attentions)}")
print(f"Last attention layer shape: {attentions[-1].shape}")
print(f"Attentions: {attentions}")




import soundfile as sf
import numpy as np

def channels_redundant(path, thresh_corr=0.999, thresh_rms_db=-50):
    audio, sr = sf.read(path, always_2d=True, dtype='float32')
    if audio.shape[1] != 2:
        return False  # not stereo → nothing to test

    left, right = audio[:, 0], audio[:, 1]

    # 1) Pearson correlation
    corr = np.corrcoef(left, right)[0, 1]

    # 2) Energy of the difference signal
    diff = left - right
    rms_db = 20 * np.log10(np.sqrt(np.mean(diff**2)) + 1e-12)

    return corr >= thresh_corr or rms_db <= thresh_rms_db
