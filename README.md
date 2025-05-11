# JAX LPIPS

LPIPS perceptual loss implementation for JAX.

Information on the metric is available in the [original repo](https://github.com/richzhang/PerceptualSimilarity).\
Pretrained network and LPIPS linear weights are available on [Hugging Face](https://huggingface.co/tux-type/jaxlpips).

## Installation
Install jaxlpips with:
```sh
pip install jaxlpips
```

## Why?
There are already some LPIPS versions for JAX.

This implementation provides:
- Alexnet and VGG support
- trimmed pretrained network parameters and calculations
- safetensors for all parameters

## Example
Supports `"alexnet"` and `"vgg16"` for `pretrained_network` to perform feature extraction.

Minimal example using dummy data:
```Python
import numpy as np
from jaxlpips import LPIPS

rng = np.random.default_rng(2781)

ref = rng.normal(size=(4, 64, 64, 3)).astype(np.float32)
tgt = rng.normal(size=(4, 64, 64, 3)).astype(np.float32)

lpips_loss_fn = LPIPS(pretrained_network="alexnet")

loss = lpips_loss_fn(ref, tgt)
```


## References
[1] Zhang, Richard, et al. "The unreasonable effectiveness of deep features as a perceptual metric." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
