# NegBioVAE core

This repository contains the count autoencoder, ordered birth–death bridge, energy correction, and a compact staged training script.

Requires Python and PyTorch.

## Training

Prepare separate training and validation PyTorch tensor files of shape `[N, C, H, W]`, with grayscale values in `[0, 1]` or RGB values in `[-1, 1]`. Supported dataset names are `MNIST`, `fmnist`, `CIFAR16`, and `CelebA64`. Run:

```bash
python train.py --dataset CIFAR16 --train-images /path/to/train.pt --validation-images /path/to/validation.pt --output /path/to/model.pt --device cpu
```

Use `python train.py --help` to adjust training budgets and the device. The script selects autoencoder and bridge weights using the validation split, then saves the final energy model after contrastive and policy training. The output file contains model weights and should be kept outside the repository.

## Inference

```python
import torch
from r01_model import build_r01_model

checkpoint = torch.load('/path/to/model.pt', map_location='cpu', weights_only=True)
model = build_r01_model(checkpoint['dataset'])
model.sampling_steps = checkpoint['sampling_steps']
model.load_state_dict(checkpoint['model'])
images = model.generate(n=2)
```

