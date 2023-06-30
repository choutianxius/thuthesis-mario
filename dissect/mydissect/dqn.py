import torch
from torch import nn
from netdissect import nethook

import numpy as np
from torchvision import transforms as T
from matplotlib import pyplot as plt
from PIL import Image


# neural network
class MarioNet(nn.Module):
    """
    Mini CNN structure
    input
    -> (conv2d + relu) * 3
    -> flatten
    -> (dense + relu) * 2
    -> output
    """
    def __init__(self, output_dim):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, input):
        if isinstance(input, str):
            return self.forward(Image.open(input))
        if isinstance(input, Image.Image):
            return self.forward(T.ToTensor(input))

        if len(input.shape) == 3:
            return self.forward(input.unsqueeze(0))
        elif len(input.shape) == 4:
            return self.stack(stack_grayscale(grayscale_and_resize(input)))
        elif len(input.shape) == 5:# batch input
            return self.forward(input.squeeze(1))


# neural network
# input is already transformed (4*84*84 grayscale)
class MarioNet1(nn.Module):
    """
    Mini CNN structure
    input
    -> (conv2d + relu) * 3
    -> flatten
    -> (dense + relu) * 2
    -> output
    """
    def __init__(self, output_dim):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, input):
        if isinstance(input, str):
            return self.forward(Image.open(input))
        if isinstance(input, Image.Image):
            return self.forward(T.ToTensor(input))

        if len(input.shape) == 3:
            return self.forward(input.unsqueeze(0))
        elif len(input.shape) == 4:
            return self.stack(stack_grayscale(input))
        elif len(input.shape) == 5:# batch input
            return self.forward(input.squeeze(1))


def load_dqn(save_path, transform=True, random=False, device='cuda'):
    data = torch.load(save_path)
    state_dict = data.get('online_model')
    output_dim = data.get('output_dim')

    if transform: # transform input within the model
        model = MarioNet(output_dim=output_dim)
    else: # input is already transformed
        model = MarioNet1(output_dim=output_dim)
    if not random:
        model.load_state_dict(state_dict)
    model = nethook.InstrumentedModel(model).eval().to(device)
    return model


def stack_grayscale(grayscale):# [1, 84, 84] => [4, 84, 84]
    if len(grayscale.shape) == 4:
        return grayscale.expand(-1, 4, -1, -1)
    return grayscale.expand(4, -1, -1)


grayscale_and_resize = T.Compose([
    T.Grayscale(),
    T.Resize((84, 84), antialias = True),
])

transform_before_dqn = T.Compose([
    T.ToTensor(),
    T.Grayscale(),
    T.Resize((84, 84), antialias=True),
])


def visualize_transform_before_dqn(pil_img):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(pil_img)
    ax1.axis('off')
    ax1.set_title('Raw: 240 * 256 RGB')

    transformed = grayscale_and_resize((T.ToTensor())(pil_img))
    ax2.imshow(transformed[0], cmap="gray")
    ax2.axis('off')
    ax2.set_title('Transformed: Downsampled 84 * 84 Grayscale')
    plt.tight_layout()
    plt.show()
