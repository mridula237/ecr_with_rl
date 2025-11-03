import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from src.lib import create_mlp
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Florence2large.load_florence import load_florence2_model

MODEL_PATH="microsoft/Florence-2-large-ft"

class Florence2Classifier(nn.Module):
    def __init__(self,
                 pretrained_model_path=MODEL_PATH,
                 model_revision='refs/pr/6',
                 net_arch=[512, 256, 64],  # MLP architecture
                 activation_fn=nn.ReLU,
                 num_labels=7,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 ):
        super().__init__()
        
        self.num_labels = num_labels
        self.net_arch = net_arch
        self.activation_fn = activation_fn

        self.num_labels = num_labels

        # Load Florence-2 base model
        self.florence2 = AutoModelForCausalLM.from_pretrained(
            pretrained_model_path,
            output_attentions=False,
            output_hidden_states=False,
            trust_remote_code=True,
            return_dict=True,
            # revision=model_revision
        )

        # Load the matching processor
        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_path,
            trust_remote_code=True,
            # revision=model_revision
        )

        # Classification head
        # self.classifier = nn.Sequential(
        # nn.Linear(self.florence2.config.text_config.d_model, 512),
        # nn.Dropout(0.2),
        # nn.LayerNorm(512),
        # nn.ReLU(),
        # nn.Linear(512, 128),
        # nn.Dropout(0.2),
        # nn.LayerNorm(128),
        # nn.ReLU(),
        # nn.Linear(128, 64),
        # nn.Dropout(0.2),
        # nn.LayerNorm(64),
        # nn.ReLU(),
        # nn.Linear(64, num_labels)
        # )
        # self.florence2, self.processor = load_florence2_model(device = device)
    
        # Classification head
        _mlp = create_mlp(self.florence2.config.text_config.d_model, 
                        self.num_labels, 
                        self.net_arch, 
                        self.activation_fn, 
                        post_linear_modules=[lambda _: nn.Dropout(p=0.1), nn.LayerNorm])
        self.classifier = nn.Sequential(*_mlp)

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, decoder_input_ids=None, labels=None):
        
    
        
        outputs = self.florence2(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=labels, 
            decoder_input_ids=torch.zeros_like(input_ids),  # Dummy decoder_input_ids required for causal LM
            output_hidden_states=True,
            return_dict=True
        )

        # Use hidden state for classification
        fused_features = outputs.encoder_last_hidden_state.mean(dim=1)
        logits = self.classifier(fused_features)
        return {"logits": logits}

