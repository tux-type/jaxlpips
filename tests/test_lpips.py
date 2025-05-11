import lpips
import numpy as np
import pytest
import torch

from jaxlpips import LPIPS as JaxLPIPS


def to_torch_tensor(x: np.ndarray):
    out = x.transpose((0, 3, 1, 2))
    out = torch.tensor(out, requires_grad=False)
    return out


@pytest.fixture(scope="session")
def dummy_data():
    rng = np.random.default_rng(2781)

    ref = rng.normal(size=(4, 64, 64, 3)).astype(np.float32)
    tgt = rng.normal(size=(4, 64, 64, 3)).astype(np.float32)
    return ref, tgt


@pytest.fixture(scope="session")
def img_data():
    rng = np.random.default_rng(9127)
    ref = np.load("tests/data/test_imgs.npy")
    tgt = ref + rng.normal(0, 0.5, size=ref.shape).astype(np.float32)
    return ref, tgt


@pytest.mark.parametrize("pretrained_network", ["alex", "vgg"])
def test_lpips_dummy(dummy_data, pretrained_network):
    ref, tgt = dummy_data
    jax_loss_fn = JaxLPIPS(pretrained_network=pretrained_network)
    loss_fn = lpips.LPIPS(net=pretrained_network)

    jax_output = jax_loss_fn(ref, tgt)
    torch_output = loss_fn(to_torch_tensor(ref), to_torch_tensor(tgt)).detach().cpu().numpy()

    assert np.allclose(jax_output, torch_output, rtol=0, atol=1e-5)


@pytest.mark.parametrize("pretrained_network", ["alex", "vgg"])
def test_lpips_img(img_data, pretrained_network):
    ref, tgt = img_data
    jax_loss_fn = JaxLPIPS(pretrained_network=pretrained_network)
    loss_fn = lpips.LPIPS(net=pretrained_network)

    jax_output = jax_loss_fn(ref, tgt)
    torch_output = loss_fn(to_torch_tensor(ref), to_torch_tensor(tgt)).detach().cpu().numpy()

    assert np.allclose(jax_output, torch_output, rtol=0, atol=1e-5)
