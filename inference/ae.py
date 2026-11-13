import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .counts import Residual, SpatialPool


class ResidualDown(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        assert 4 * cin % cout == 0
        self.cout = cout
        self.learned = nn.Conv2d(cin, cout, 3, stride=2, padding=1)

    def forward(self, x):
        
        x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2))
        a = F.pixel_unshuffle(x, 2)
        shortcut = a.reshape(len(x), self.cout, -1, *a.shape[-2:]).mean(2)
        return (shortcut + self.learned(x)) / math.sqrt(2)


class ResidualUp(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        assert 4 * cout % cin == 0
        self.copies = 4 * cout // cin
        self.learned = nn.Conv2d(cin, 4 * cout, 3, padding=1)

    def forward(self, x):
        shortcut = F.pixel_shuffle(x.repeat_interleave(self.copies, dim=1), 2)
        return (shortcut + F.pixel_shuffle(self.learned(x), 2)) / math.sqrt(2)


class ReconstructionAE(nn.Module):
    def __init__(self, config, channels=3, size=16):
        super().__init__()
        self.config = dict(config)
        self.channels, self.size, self.states = channels, size, config['states']
        self.local = nn.Sequential(nn.Conv2d(channels, 32, 3, padding=1), Residual(32),
                                   ResidualDown(32, 64), Residual(64),
                                   ResidualDown(64, 128), Residual(128),
                                   SpatialPool(), nn.Conv2d(128, 64, 1))
        layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0., activation='gelu',
                                            batch_first=True, norm_first=True)
        self.semantic = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.position = nn.Parameter(torch.randn(1, 16, 64) * .02)
        self.local_logits = nn.Conv2d(64, 16 * self.states, 1)
        self.gate = nn.Conv2d(129, 64, 1)
        self.posterior = nn.Conv2d(64, (8 if config['residual'] else 16) * self.states, 1)
        if config['adaptive']:
            self.temperature = nn.Conv2d(64, 16, 1)
        if config['residual']:
            self.coarse_lift = nn.Conv2d(8, 64, 1)
            self.fine_posterior = nn.Sequential(Residual(64), nn.Conv2d(64, 8 * self.states, 1))
        self.log_r_prior = nn.Parameter(torch.zeros(1, 16, 4, 4))
        self.logit_p_prior = nn.Parameter(torch.zeros(1, 16, 4, 4))
        width = 192 if config['wide'] else 128
        self.decoder_start = nn.Sequential(nn.Conv2d(16, width, 3, padding=1), Residual(width))
        blocks = [ResidualUp(width, width // 2), Residual(width // 2),
                  ResidualUp(width // 2, width // 4), Residual(width // 4)]
        resolution = 16
        while resolution < size:
            blocks += [ResidualUp(width // 4, width // 4), Residual(width // 4)]
            resolution *= 2
        self.decoder_up = nn.Sequential(*blocks)
        self.decoder_out = nn.Conv2d(width // 4, channels, 3, padding=1)

    def prior_log_probs(self):
        k = torch.arange(self.states, device=self.log_r_prior.device)
        r = self.log_r_prior.clamp(-4, 5).exp()[..., None] + 1e-6
        p = self.logit_p_prior.sigmoid().clamp(1e-5, 1-1e-5)[..., None]
        logp = ((k+r).lgamma() - r.lgamma() - (k+1).lgamma()
                + r*p.log() + k*torch.log1p(-p))
        return logp.log_softmax(-1)

    def probs(self, raw, logprior, temperature=None):
        q = raw.reshape(len(raw), -1, self.states, 4, 4).permute(0, 1, 3, 4, 2)
        if temperature is not None:
            q = q / temperature[..., None]
        return (q + logprior).log_softmax(-1)

    def draw(self, logq, tau):
        k = torch.arange(self.states, device=logq.device, dtype=logq.dtype)
        if self.config['soft']:
            return (logq.exp() * k).sum(-1)
        
        
        weights = F.gumbel_softmax(logq, tau=tau, hard=True, dim=-1)
        return (weights * k).sum(-1)

    def normalized(self, z):
        return z * (2 / (self.states-1)) - 1

    def encode(self, x, tau=1.):
        prior = self.prior_log_probs()
        local = self.local(x)
        tokens = local.flatten(2).transpose(1, 2)
        with sdpa_kernel(SDPBackend.MATH):
            semantic = self.semantic(tokens + self.position).transpose(1, 2).reshape(-1, 64, 4, 4)
        local_logq = self.probs(self.local_logits(local), prior)
        k = torch.arange(self.states, device=x.device, dtype=x.dtype)
        q = local_logq.exp()
        variance = ((q*k.square()).sum(-1) - (q*k).sum(-1).square()).clamp_min(0)
        uncertainty = variance.mean(1, keepdim=True) / ((self.states-1)**2/4)
        gate = self.gate(torch.cat([local, semantic, uncertainty.clamp(0, 1)], 1)).sigmoid()
        fused = local + gate*semantic
        temperature = .25 + 1.75*self.temperature(fused).sigmoid() if self.config['adaptive'] else None
        if self.config['residual']:
            t0 = None if temperature is None else temperature[:, :8]
            t1 = None if temperature is None else temperature[:, 8:]
            logq0 = self.probs(self.posterior(fused), prior[:, :8], t0)
            z0 = self.draw(logq0, tau)
            detail = fused - self.coarse_lift(self.normalized(z0))
            logq1 = self.probs(self.fine_posterior(detail), prior[:, 8:], t1)
            z1 = self.draw(logq1, tau)
            logq, z = torch.cat([logq0, logq1], 1), torch.cat([z0, z1], 1)
        else:
            logq = self.probs(self.posterior(fused), prior, temperature)
            z = self.draw(logq, tau)
        
        
        
        kl = (logq.exp() * (logq-prior)).sum(-1).mean()
        entropy = -(logq.exp()*logq).sum(-1).mean()
        return z, dict(kl=kl, entropy=entropy, logq=logq, uncertainty=uncertainty.mean())

    def decode(self, z):
        assert z.shape[1:] == (16, 4, 4)
        h = self.decoder_up(self.decoder_start(self.normalized(z)))
        
        if h.shape[-2:] != (self.size, self.size):
            h = F.interpolate(h, size=(self.size, self.size), mode='bilinear', align_corners=False)
        y = self.decoder_out(h)
        return y.sigmoid() if self.channels == 1 else y.tanh()

    def forward(self, x, tau=1.):
        z, info = self.encode(x, tau=tau)
        return self.decode(z), z, info

    def prefix_reconstruction(self, z, channels):
        mask = torch.arange(16, device=z.device)[None, :, None, None] < channels
        
        neutral = torch.full_like(z, (self.states-1)/2)
        return self.decode(torch.where(mask, z, neutral))
