import numpy as np


class Parameters:
    def __init__(self, **shapes):
        self.shapes = shapes
        self.index_map = {}
        counter = 0
        for k, v in shapes.items():
            n_params = np.prod(v)
            idx = np.arange(n_params) + counter
            if not v == ():
                idx = idx.reshape(v)
            self.index_map[k] = idx.astype(int)
            counter += n_params
        self.params = list(self.index_map.keys())

    def flatten(self, **params):
        return np.concatenate([np.asarray(v).flatten() for k, v in params.items()])

    def unflatten(self, params):
        return {k: params[idx] for k, idx in self.index_map.items()}
