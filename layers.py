import mlx.core as mx


def flatten(x):
    return x.reshape(x.shape[0], -1)


def upsample_nearest(x, scale: int = 2):
    B, H, W, C = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, scale, W, scale, C))
    x = x.reshape(B, H * scale, W * scale, C)
    return x


def global_avg_pool2d(x):
    return x.mean((1, 2), keepdims=True)


def avg_pool2d(x, stride=2):
    B, W, H, C = x.shape
    x = x.reshape(B, W // stride, stride, H // stride, stride, C).mean((2, 4))
    return x


def max_pool2d(x, stride=2):
    B, W, H, C = x.shape
    x = x.reshape(B, W // stride, stride, H // stride, stride, C).max((2, 4))
    return x