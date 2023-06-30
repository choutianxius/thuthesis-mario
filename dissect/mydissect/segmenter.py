"""
Define the helper functions
"""
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
from torchvision import models as tvmodels
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class Segmodel:
    def __init__(self, save_path, transform=transforms.ToTensor(), device='cuda'):
        self.net = load_segnet(save_path).to(device) # input shape [k, C, H, W]
        self.transform = transform
        self.device=device
    def seg(self, input, numpy=False):
        if isinstance(input, str):
            return self.seg(Image.open(input), numpy)
        if isinstance(input, Image.Image):
            return self.seg(self.transform(input).to(self.device), numpy)
        if len(input.shape) == 3:
            return self.seg(input.unsqueeze(0), numpy)
        if len(input.shape) == 4:
            out = self.net(input)['out']
            segm = torch.argmax(out, dim=1)
            if numpy:
                segm = segm.detach().cpu().numpy()
            return segm
        if len(input.shape) == 5:
            assert input.shape[0] == 1
            return self.seg(input.squeeze(0), numpy)

    def __call__(self, input, numpy=False):
        return self.seg(input, numpy)


def load_segmodel(save_path, transform=transforms.ToTensor(), device='cuda'):
    return Segmodel(save_path, transform=transform, device=device)


def load_segnet(save_path):
    segmodel = tvmodels.segmentation.deeplabv3_resnet50(
        pretrained=True, progress=True)
    # Added a Sigmoid activation after the last convolution layer
    segmodel.classifier = DeepLabHead(2048, 6)

    # models/MarioSegmentationModel.pth
    segmodel.load_state_dict(torch.load(save_path))
    segmodel.eval()

    return segmodel


def decode_segmap(image, nc=21):
    ## Color palette for visualization of the 21 classes
    label_colors = np.array([(0, 0, 0),  # 0=background
    # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
    (0, 0,255), (127, 127, 0), (0, 255, 0), (255, 0, 0), (255, 255, 0),
    # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
    (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
    # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
    (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
    # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
  
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def test_segment(model, path, show_orig=True, save=False, save_path=None):
    img = Image.open(path)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    if show_orig: ax1.imshow(img); ax1.axis('off'); ax1.set_title('Raw Image')

    input_image = transform(img).unsqueeze(0).to(dev)

    out = net(input_image)['out'][0]

    segm = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    segm_rgb = decode_segmap(segm)
    ax2.imshow(segm_rgb)
    ax2.axis('off')
    ax2.set_title('Segmentation Mask')
    plt.tight_layout()
    if save:
        plt.save(save_path)
    else:
        plt.show()
