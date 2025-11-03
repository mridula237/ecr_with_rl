import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoProcessor, Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch
from src.lib import create_mlp

MODEL_PATH="microsoft/Florence-2-large-ft"

class Florence2WavClassifier(nn.Module):
    def __init__(self,
                 pretrained_model_path=MODEL_PATH,
                 model_revision='refs/pr/6',
                 wav_pretrained_model_path="facebook/wav2vec2-large-960h",
                 net_arch=[512, 256, 64],  # MLP architecture
                 activation_fn=nn.ReLU,
                 num_labels=7):
        super().__init__()

        self.num_labels = num_labels
        self.net_arch = net_arch
        self.activation_fn = activation_fn

        # Load Florence-2 base model
        self.florence2 = AutoModelForCausalLM.from_pretrained(
            pretrained_model_path,
            output_attentions=False,
            output_hidden_states=False,
            trust_remote_code=True,
            return_dict=True,
        )

        # Load the matching processor
        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_path,
            trust_remote_code=True,
        )
        
        self.wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav_pretrained_model_path)
        self.wav_model = Wav2Vec2Model.from_pretrained(wav_pretrained_model_path)
        self.wav_model.config.apply_spec_augment = False  # Disable spec augment for training stability because there are some utterances with very short audio clips

        # (4) optional LayerNorm after concatenation
        self.fuse_norm  = nn.LayerNorm(self.florence2.config.text_config.d_model * 2)  # 

        # Classification head
        # _mlp = create_mlp(self.florence2.config.text_config.d_model * 2, 
        #                 self.num_labels, 
        #                 self.net_arch, 
        #                 self.activation_fn, 
        #                 post_linear_modules=[lambda _: nn.Dropout(p=0.1), nn.LayerNorm])
        # self.classifier = nn.Sequential(*_mlp)
        self.classifier = nn.Sequential(
        nn.Linear(self.florence2.config.text_config.d_model * 2, 512),
        nn.Dropout(0.2),
        nn.LayerNorm(512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.Dropout(0.2),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.Dropout(0.2),
        nn.LayerNorm(64),
        nn.ReLU(),
        nn.Linear(64, num_labels)
        )

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, decoder_input_ids=None, labels=None, audio_inputs=None):

        # Florence-2 forward pass
        outputs = self.florence2(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=labels, 
            decoder_input_ids=torch.zeros_like(input_ids),  # Dummy decoder_input_ids required for causal LM
            output_hidden_states=True,
            return_dict=True
        )
        # both models output
        audio_vec = self.wav_model(audio_inputs).last_hidden_state.mean(dim=1)
        text_vec = outputs.encoder_last_hidden_state.mean(dim=1)


        # (3) combine audio and text features
        fused = torch.cat([audio_vec, text_vec], dim=-1)  # concatination = (B, 2048)
        # fused = 0.9 * text_vec + 0.1 * audio_vec  # linear combination = (B, 1024)

        # (4) global LayerNorm
        fused_features = self.fuse_norm(fused)      
        # (5) classification head  
        logits = self.classifier(fused_features)
        return {"logits": logits}



wav_pretrained_model_path="facebook/wav2vec2-large-960h" 
wav2vec_model = Wav2Vec2Model.from_pretrained(wav_pretrained_model_path,)
wav2vec_model