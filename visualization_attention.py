# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib
matplotlib.use('Agg')  # save to files only; avoids Qt on headless machines
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

from main import build_model


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


def to_heatmaps(attn, w_featmap, h_featmap, patch_size, threshold=None):
    """CLS-token attention -> per-head maps upsampled to image resolution.

    Returns (maps, th_masks); th_masks is None when no threshold is given.
    """
    nh = attn.shape[1]
    # we keep only the output patch attention
    attn = attn[0, :, 0, 1:].reshape(nh, -1)

    th_attn = None
    if threshold is not None:
        # the mass-based threshold assumes non-negative values; the differential
        # map has negatives, so only its positive part contributes
        mass = attn.clamp(min=0)
        val, idx = torch.sort(mass)
        val /= torch.sum(val, dim=1, keepdim=True).clamp(min=1e-12)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        th_attn = nn.functional.interpolate(
            th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    attn = attn.reshape(nh, w_featmap, h_featmap)
    attn = nn.functional.interpolate(
        attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    return attn, th_attn


def save_heatmaps(maps, output_dir, prefix, signed=False):
    """Write one png per head. `signed` maps get a diverging colormap centred on 0."""
    for j in range(maps.shape[0]):
        fname = os.path.join(output_dir, f"{prefix}-head{j}.png")
        if signed:
            lim = float(np.abs(maps[j]).max()) or 1.0
            plt.imsave(fname=fname, arr=maps[j], format='png', cmap='bwr', vmin=-lim, vmax=lim)
        else:
            plt.imsave(fname=fname, arr=maps[j], format='png')
        print(f"{fname} saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--mode', default='sa', type=str,
        help="Per-block attention mode: 'sa' | 'da' | list 'sa,da,...' | index form '1:da,4:da'")
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='output', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    model = build_model(arch=args.arch, patch_size=args.patch_size, mode=args.mode, num_classes=0)
    print(f"blocks: {model.modes}")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    # open image
    if args.image_path is None:
        # user has not specified any image - we use our own image
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, we take the first image in our paper.")
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
    elif os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)
    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // args.patch_size
    h_featmap = img.shape[-1] // args.patch_size

    attentions = model.get_last_selfattention(img.to(device))

    # a 'da' last block returns the two softmax maps and their difference,
    # a 'sa' one returns a single tensor
    if isinstance(attentions, dict):
        groups = [('attn1', attentions['attn1'], False),
                  ('attn2', attentions['attn2'], False),
                  ('diff', attentions['diff'], True)]
    else:
        groups = [('attn', attentions, False)]

    os.makedirs(args.output_dir, exist_ok=True)
    torchvision.utils.save_image(
        torchvision.utils.make_grid(img, normalize=True, scale_each=True),
        os.path.join(args.output_dir, "img.png"))
    image = skimage.io.imread(os.path.join(args.output_dir, "img.png")) if args.threshold is not None else None

    for name, attn, signed in groups:
        maps, th_attn = to_heatmaps(attn, w_featmap, h_featmap, args.patch_size, args.threshold)
        save_heatmaps(maps, args.output_dir, name, signed=signed)
        if th_attn is not None:
            if signed:
                print(f"note: '{name}' has negative values; the mask uses its positive part only.")
            for j in range(th_attn.shape[0]):
                display_instances(
                    image, th_attn[j],
                    fname=os.path.join(args.output_dir, f"mask_th{args.threshold}_{name}-head{j}.png"),
                    blur=False)
