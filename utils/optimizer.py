class NoamOpt(object):
    "Optim wrapper that implements rate."

    def __init__(self, config, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = config.warmup
        self.factor = config.factor
        self.model_size = config.d_model

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))



