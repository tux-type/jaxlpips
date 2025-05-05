from pathlib import Path
from typing import Callable

import jax
from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict
from safetensors.flax import load_file, save_file


def save_model(file_name: str | Path, model: nnx.Module | dict[str, jax.Array]):
    state_dict = model if isinstance(model, dict) else nnx.state(model).to_pure_dict()
    flat_dict: dict[str, jax.Array] = {
        ".".join(k): v for k, v in flatten_dict(state_dict).items()
    }  # type: ignore
    save_file(tensors=flat_dict, filename=file_name)


def load_model(file_name: str, model: Callable[[], nnx.Module]) -> nnx.Module:
    graph_def = nnx.graphdef(nnx.eval_shape(model))
    flat_dict = load_file(file_name)
    flat_dict = {tuple(k.split(".")): v for k, v in flat_dict.items()}
    state_dict = unflatten_dict(flat_dict)
    model = nnx.merge(graph_def, state_dict)
    return model  # type: ignore
