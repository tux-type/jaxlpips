import jax
from flax import nnx


class AlexNetFeatures(nnx.Module):
    def __init__(self, rngs: nnx.Rngs) -> None:
        self.conv1 = nnx.Conv(3, 64, kernel_size=(11, 11), strides=4, padding=2, rngs=rngs)
        self.conv2 = nnx.Conv(64, 192, kernel_size=(5, 5), padding=2, rngs=rngs)
        self.conv3 = nnx.Conv(192, 384, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.conv4 = nnx.Conv(384, 256, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.conv5 = nnx.Conv(256, 256, kernel_size=(3, 3), padding=1, rngs=rngs)

    def __call__(self, x: jax.Array) -> list[jax.Array]:
        act1 = nnx.relu(self.conv1(x))
        act2 = nnx.relu(self.conv2(nnx.max_pool(act1, window_shape=(3, 3), strides=(2, 2))))
        act3 = nnx.relu(self.conv3(nnx.max_pool(act2, window_shape=(3, 3), strides=(2, 2))))
        act4 = nnx.relu(self.conv4(act3))
        act5 = nnx.relu(self.conv5(act4))

        return [act1, act2, act3, act4, act5]
