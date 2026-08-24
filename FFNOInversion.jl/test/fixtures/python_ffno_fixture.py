import numpy as np

factory_calls = 0


class FixtureBackend:
    def __init__(self, scale=2.0):
        self.scale = float(scale)

    def predict(self, features, z_scale, dx, dy):
        features = np.asarray(features)
        z_scale = np.asarray(z_scale)
        assert features.shape[1:] == z_scale.shape
        assert float(dx) > 0 and float(dy) > 0
        temperature = features[0]
        return np.stack((temperature, self.scale * temperature), axis=-1).astype(np.float32)


def create_backend(scale=2.0):
    global factory_calls
    factory_calls += 1
    return FixtureBackend(scale=scale)
