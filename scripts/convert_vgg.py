from pathlib import Path
from typing import Callable
import itertools

import jax
import jax.numpy as jnp
import lpips
import numpy as np
import torch
from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict
from lpips import pretrained_networks as pn
from safetensors.flax import load_file, save_file
from torchvision.models import VGG, VGG16_Weights, vgg16

from scripts.vgg import VGG16Features

### Torch
torch_model = vgg16(VGG16_Weights.IMAGENET1K_V1)
# Convert to a format closer to JAX implementation
torch_state = {}
for param_name, param_value in torch_model.state_dict().items():
    if "features" not in param_name:
        continue
    _, jax_idx, param_type = param_name.split(".")
    jax_idx = int(jax_idx)
    param_type = "kernel" if param_type == "weight" else param_type
    param_dict = torch_state.get(jax_idx, dict())
    param_dict[param_type] = param_value
    torch_state[jax_idx] = param_dict

### Jax
# Abstract
jax_abs_model = nnx.eval_shape(lambda: VGG16Features(rngs=nnx.Rngs(0)))
graph_def, jax_state = nnx.split(jax_abs_model)
jax_state = jax_state.to_pure_dict()  # type: ignore


### LPIPS
pn_model = pn.vgg16()
pn_state = pn_model.state_dict()


# Torch to Jax
nested_pairs = list((i, j) for i in range(1, len(jax_state.keys()) + 1) for j in jax_state[f"conv{i}"]["conv"]["layers"].keys())
for (i, j), torch_weights in zip(nested_pairs, torch_state.values(), strict=True):
    jax_kernel = torch_weights["kernel"].detach().cpu().numpy().transpose((2, 3, 1, 0))
    jax_bias = torch_weights["bias"].detach().cpu().numpy()
    assert jax_state[f"conv{i}"]["conv"]["layers"][j]["kernel"].shape == jax_kernel.shape
    assert jax_state[f"conv{i}"]["conv"]["layers"][j]["bias"].shape == jax_bias.shape
    jax_state[f"conv{i}"]["conv"]["layers"][j]["kernel"] = jax_kernel
    jax_state[f"conv{i}"]["conv"]["layers"][j]["bias"] = jax_bias


jax_model = nnx.merge(graph_def, jax_state)

x_dummy = np.ones((4, 256, 256, 3), dtype=np.float32)

jax_output = jax_model(jnp.array(x_dummy))

torch_output = torch_model.features.forward(
    torch.from_numpy(x_dummy.transpose(0, 3, 1, 2))
)
torch_output = torch_output.detach().cpu().numpy().transpose(0, 2, 3, 1)

np.allclose(
    nnx.max_pool(jax_output[-1], window_shape=(2, 2), strides=(2, 2)),
    torch_output,
    atol=1e-03,
    rtol=0,
)

lpips_model = pn.vgg16()

lpips_output = lpips_model.forward(torch.from_numpy(x_dummy.transpose(0, 3, 1, 2)))
lpips_output = [
    output.detach().cpu().numpy().transpose(0, 2, 3, 1) for output in lpips_output
]

print(all([
    np.allclose(jax_o, lpips_o, atol=1e-02, rtol=0)
    for jax_o, lpips_o in zip(jax_output, lpips_output)
]))


print(jax_output[1][0][0][0][:10])
print(lpips_output[1][0][0][0][:10])


# Save/load model
def save_model(file_name: str | Path, model: nnx.Module | dict[str, jax.Array]):
    state_dict = model if isinstance(model, dict) else nnx.state(model).to_pure_dict()
    flat_dict = {}
    for k, v in flatten_dict(state_dict).items():
        new_k = ".".join((str(layer_name) for layer_name in k))
        flat_dict[new_k] = v
    # flat_dict: dict[str, jax.Array] = {
    #     ".".join(k): v for k, v in flatten_dict(state_dict).items()
    # }  # type: ignore
    save_file(tensors=flat_dict, filename=file_name)


def load_model(file_name: str, model: Callable[[], nnx.Module]) -> nnx.Module:
    graph_def = nnx.graphdef(nnx.eval_shape(model))
    flat_dict = load_file(file_name)
    flat_dict = {tuple(k.split(".")): v for k, v in flat_dict.items()}
    state_dict = unflatten_dict(flat_dict)
    model = nnx.merge(graph_def, state_dict)
    return model  # type: ignore


save_model("models/vgg16_features.safetensors", jax_state)

model: VGG16Features = load_model(
    "models/vgg16_features.safetensors", lambda: VGG16Features(rngs=nnx.Rngs(0))
)


reloaded_output = model(jnp.ones((4, 256, 256, 3)))

all([np.allclose(reloaded_output[i], jax_output[i]) for i in range(len(jax_output))])


