import jax
from flax import nnx


class ConvBlock(nnx.Module):
    def __init__(self, in_features: int, out_features: int, num_conv: int, rngs: nnx.Rngs):
        layers = []
        for _ in range(num_conv):
            conv = nnx.Conv(in_features, out_features, kernel_size=(3, 3), padding=1, rngs=rngs)
            layers.append(conv)
            layers.append(nnx.relu)
            in_features = out_features
        self.conv = nnx.Sequential(*layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.conv(x)


class VGG16Features(nnx.Module):
    def __init__(self, rngs: nnx.Rngs) -> None:
        self.conv1 = ConvBlock(3, 64, num_conv=2, rngs=rngs)
        self.conv2 = ConvBlock(64, 128, num_conv=2, rngs=rngs)
        self.conv3 = ConvBlock(128, 256, num_conv=3, rngs=rngs)
        self.conv4 = ConvBlock(256, 512, num_conv=3, rngs=rngs)
        self.conv5 = ConvBlock(512, 512, num_conv=3, rngs=rngs)

    def __call__(self, x: jax.Array) -> list[jax.Array]:
        act1 = self.conv1(x)
        act2 = self.conv2(nnx.max_pool(act1, window_shape=(2, 2), strides=(2, 2)))
        act3 = self.conv3(nnx.max_pool(act2, window_shape=(2, 2), strides=(2, 2)))
        act4 = self.conv4(nnx.max_pool(act3, window_shape=(2, 2), strides=(2, 2)))
        act5 = self.conv5(nnx.max_pool(act4, window_shape=(2, 2), strides=(2, 2)))

        return [act1, act2, act3, act4, act5]
