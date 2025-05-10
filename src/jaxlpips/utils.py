from typing import Callable, TypeVar

import jax
from flax import nnx
from flax.traverse_util import unflatten_dict
from huggingface_hub import hf_hub_download
from safetensors.flax import load_file

from jaxlpips.modules import AlexNetFeatures, VGG16Features

ModelType = TypeVar("ModelType", bound=nnx.Module)


def load_model(model_name: str) -> tuple[AlexNetFeatures | VGG16Features, list[jax.Array]]:
    match model_name:
        case "alexnet":
            model_class = AlexNetFeatures
        case "vgg16":
            model_class = VGG16Features
        case _:
            raise ValueError("Only 'alexnet' and 'vgg16' pretrained networks are supported.")

    model_weights_file = hf_hub_download(
        repo_id="tux-type/jaxlpips", filename=model_name + "_features.safetensors"
    )
    model_weights = load_file(model_weights_file)
    linear_weights_file = hf_hub_download(
        repo_id="tux-type/jaxlpips", filename=model_name + "_lpips.safetensors"
    )
    linear_weights = list(load_file(linear_weights_file).values())

    model = to_nnx(model_weights, lambda: model_class(rngs=nnx.Rngs(0)))
    return model, linear_weights


def to_nnx(flat_weights: dict[str, jax.Array], lazy_model: Callable[[], ModelType]) -> ModelType:
    graph_def = nnx.graphdef(nnx.eval_shape(lazy_model))
    flat_dict = {tuple(k.split(".")): v for k, v in flat_weights.items()}
    state_dict = unflatten_dict(flat_dict)
    model = nnx.merge(graph_def, state_dict)
    return model
