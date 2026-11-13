import torch
from torch import nn

from .ae import ReconstructionAE
from .counts import BirthDeathBridge, CountEnergy, metropolis


class UnifiedR01(nn.Module):
    def __init__(self, config, channels=3, size=16, sampling_steps=64, mh_steps=4, reference_rate=5.):
        super().__init__()
        if config['states'] != 5 or config['soft']:
            raise ValueError('This release is the five-state hard-count R01 model.')
        self.channels, self.size = channels, size
        self.ae = ReconstructionAE(config, channels=channels, size=size)
        self.bridge = BirthDeathBridge(reference_rate)
        self.energy = CountEnergy()
        self.sampling_steps, self.mh_steps = sampling_steps, mh_steps

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.channels, self.size, self.size):
            raise ValueError(f'Expected shape [B,{self.channels},{self.size},{self.size}].')
        return self.ae(x)[0]

    @torch.no_grad()
    def reconstruct(self, x):
        return self(x)

    @torch.no_grad()
    def encode(self, x):
        z, _ = self.ae.encode(x)
        return z.flatten(1)

    def decode(self, z):
        return self.ae.decode(z.reshape(-1, 16, 4, 4).float())

    def prior(self, n):
        logp = self.ae.prior_log_probs().expand(n, -1, -1, -1, -1)
        return self.ae.draw(logp, tau=1.).flatten(1)

    @torch.no_grad()
    def generate(self, n=64, *, batch_size=64, mode='full'):
        if not isinstance(n, int) or n <= 0 or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('n and batch_size must be positive integers.')
        if mode not in ('full', 'prior'):
            raise ValueError("mode must be 'full' or 'prior'.")
        device = next(self.parameters()).device
        batches = []
        for start in range(0, n, batch_size):
            count = min(batch_size, n-start)
            if mode == 'prior':
                z = self.prior(count)
            else:
                proposals = self.bridge.sample(self.prior((self.mh_steps+1)*count), self.sampling_steps)
                proposals = proposals.reshape(self.mh_steps+1, count, 256)
                z, _, _ = metropolis(self.energy, proposals)
            batches.append(self.decode(z))
        return torch.cat(batches)
