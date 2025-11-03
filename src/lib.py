import torch
import torch.nn as nn
from typing import Optional

# Function to run the model on an example
def run_example(model, processor, task_prompt, text_input, image, device):
    prompt = task_prompt + text_input

    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    return parsed_answer


# Given label mapping
LABLE2ID = {
    "neutral": 0,
    "joy": 1,
    "sadness": 2,
    "anger": 3,
    "surprise": 4,
    "fear": 5,
    "disgust": 6,
}

def emotion_one_hot_encode(target_labels, label_map=LABLE2ID, device='cpu'):
    """
    Converts a list of target labels into a one-hot encoded tensor.

    Args:
        target_labels (list of str): List of target emotion labels.
        label_map (dict): Dictionary mapping labels to indices.
        device (str): The device to place the tensor on (default: 'cuda:0').

    Returns:
        torch.Tensor: One-hot encoded tensor of shape (batch_size, num_classes).
    """
    target_indices = []
    for label in target_labels:
        assert label in label_map, f"Invalid label detected: {label}"
        target_indices.append(label_map[label])

    target_indices = torch.tensor(target_indices, device=device)
    num_classes = len(label_map)

    assert (target_indices >= 0).all() and (target_indices < num_classes).all(), "Label index out of bounds!"

    one_hot_target = torch.nn.functional.one_hot(target_indices, num_classes=num_classes).float()

    return one_hot_target

def emotion_label_to_id(label_list, label_map=LABLE2ID): #TODO: need to be tested
    """
    Converts a list of emotion labels to their corresponding IDs.

    Args:
        label_list (list of str): List of emotion labels.
        label_map (dict): Dictionary mapping labels to indices.

    Returns:
        list of int: List of emotion IDs corresponding to the labels.
    """
    return [label_map[label] for label in label_list if label in label_map]


#coppied from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py
def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: list[int],
    activation_fn: type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
    with_bias: bool = True,
    pre_linear_modules: Optional[list[type[nn.Module]]] = None,
    post_linear_modules: Optional[list[type[nn.Module]]] = None,
) -> list[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim: Dimension of the output (last layer, for instance, the number of actions)
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :param with_bias: If set to False, the layers will not learn an additive bias
    :param pre_linear_modules: List of nn.Module to add before the linear layers.
        These modules should maintain the input tensor dimension (e.g. BatchNorm).
        The number of input features is passed to the module's constructor.
        Compared to post_linear_modules, they are used before the output layer (output_dim > 0).
    :param post_linear_modules: List of nn.Module to add after the linear layers
        (and before the activation function). These modules should maintain the input
        tensor dimension (e.g. Dropout, LayerNorm). They are not used after the
        output layer (output_dim > 0). The number of input features is passed to
        the module's constructor.
    :return: The list of layers of the neural network
    """

    pre_linear_modules = pre_linear_modules or []
    post_linear_modules = post_linear_modules or []

    modules = []
    if len(net_arch) > 0:
        # BatchNorm maintains input dim
        for module in pre_linear_modules:
            modules.append(module(input_dim))

        modules.append(nn.Linear(input_dim, net_arch[0], bias=with_bias))

        # LayerNorm, Dropout maintain output dim
        for module in post_linear_modules:
            modules.append(module(net_arch[0]))

        modules.append(activation_fn())

    for idx in range(len(net_arch) - 1):
        for module in pre_linear_modules:
            modules.append(module(net_arch[idx]))

        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))

        for module in post_linear_modules:
            modules.append(module(net_arch[idx + 1]))

        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        # Only add BatchNorm before output layer
        for module in pre_linear_modules:
            modules.append(module(last_layer_dim))

        modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))
    if squash_output:
        modules.append(nn.Tanh())
    return modules
