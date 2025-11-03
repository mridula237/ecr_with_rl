import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.lib import create_mlp

mlp_ = create_mlp(
    input_dim=768,
    output_dim=7,
    net_arch=[512, 128, 64],
    activation_fn=torch.nn.ReLU,
    squash_output=False,
    with_bias=True,
    pre_linear_modules=[torch.nn.BatchNorm1d],
    post_linear_modules=[lambda _: torch.nn.Dropout(p=0.25), torch.nn.LayerNorm]
)

mlp_2 = create_mlp(
    input_dim=768,
    output_dim=7,
    net_arch=[512, 128, 64],
    activation_fn=torch.nn.ReLU,
    squash_output=False,
    with_bias=True,
    # pre_linear_modules=[torch.nn.BatchNorm1d],
    post_linear_modules=[lambda _: torch.nn.Dropout(p=0.25), torch.nn.LayerNorm]
)

mlp_3 = create_mlp(
    input_dim=768,
    output_dim=7,
    net_arch=[512, 128, 64],
    activation_fn=torch.nn.ReLU,
    squash_output=False,
    # with_bias=True,
    # pre_linear_modules=[torch.nn.BatchNorm1d],
    post_linear_modules=[lambda _: torch.nn.Dropout(p=0.25), torch.nn.LayerNorm]
)

_mlp = create_mlp(1024 * 2, 
                7, 
                [512, 256, 128, 64], 
                torch.nn.ReLU, 
                post_linear_modules=[lambda _: torch.nn.Dropout(p=0.1), torch.nn.LayerNorm])
classifier = torch.nn.Sequential(*_mlp)
        

print(f"full mlp: {mlp_} \n")
print(f"mlp no pre_linear modules: {mlp_2} \n")
print(f"mlp no with_bias: {mlp_3} \n")