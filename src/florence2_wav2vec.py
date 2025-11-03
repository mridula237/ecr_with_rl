import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoProcessor, Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch
from src.lib import create_mlp

MODEL_PATH="microsoft/Florence-2-large-ft"
WAV_MODEL_PATH="facebook/wav2vec2-large-960h" 

class CrossModalityAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        attn_output, _ = self.attn(query, key, value, key_padding_mask=key_padding_mask)
        attn_output = self.dropout(attn_output)
        return attn_output
    
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.ffn(x)
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_fn=nn.ReLU):
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation_fn())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(0.1))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class Florence2WavAttnClassifier(nn.Module):
    def __init__(self,
                 pretrained_model_path=MODEL_PATH,
                 wav_pretrained_model_path=WAV_MODEL_PATH,
                 net_arch=[512, 256, 64],  # MLP architecture
                 activation_fn=nn.ReLU,
                 num_labels=7,
                 use_cross_attn=False):
        super().__init__()

        self.num_labels = num_labels
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.use_cross_attn = use_cross_attn

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

        self.cross_attn = CrossModalityAttention(
            embed_dim=self.florence2.config.text_config.d_model,
            num_heads=8,
            dropout=0.1
        )
        
        self.ffn = FeedForwardNetwork(
            input_dim=self.florence2.config.text_config.d_model,
            hidden_dim=1024,
            output_dim=self.florence2.config.text_config.d_model
        )
        
        if use_cross_attn:
            self.norm1 = nn.LayerNorm(self.florence2.config.text_config.d_model)
            self.norm2 = nn.LayerNorm(self.florence2.config.text_config.d_model)
        else:
            self.norm2 = nn.LayerNorm(self.florence2.config.text_config.d_model * 2)
        
        # Classification head
        self.mlp = MLP(
            input_dim=self.florence2.config.text_config.d_model,
            hidden_dims=self.net_arch,
            output_dim=self.num_labels,
            activation_fn=self.activation_fn
        )
        

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, decoder_input_ids=None, labels=None, audio_inputs=None):
        if audio_inputs is None:
            raise ValueError("audio_inputs must be provided for Florence2WavAttnClassifier")
        # Extract audio features
        audio_vec = self.wav_model(audio_inputs).last_hidden_state.mean(dim=1)  # Mean pooling over the sequence length
        
        # Florence-2 forward pass
        outputs = self.florence2(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            audio_features=audio_vec,  # Pass audio features
            labels=labels, 
            decoder_input_ids=torch.zeros_like(input_ids),  # Dummy decoder_input_ids required for causal LM
            output_hidden_states=True,
            return_dict=True
        )
        
        text_vec = outputs.encoder_last_hidden_state
        
        if self.use_cross_attn:
            attn_out = self.cross_attn(audio_vec, text_vec, text_vec)
            attn_out = self.norm1(attn_out + audio_vec)  # residual connection
            attn_out = self.ffn(attn_out)
            attn_out = self.norm2(attn_out + audio_vec)  # residual connection
            fused_features = attn_out.mean(dim=1)  # mean pooling over the sequence length
        else:
            audio_vec = audio_vec.mean(dim=1)
            text_vec = text_vec.mean(dim=1)
            fused = torch.cat([audio_vec, text_vec], dim=-1)  # concatination = (B, 2048)
            # fused = 0.9 * text_vec + 0.1 * audio_vec  # linear combination = (B, 1024)
            fused_features = self.norm2(fused)           
        
        # (5) classification head
        logits = self.mlp(fused_features)

        return {"logits": logits}



