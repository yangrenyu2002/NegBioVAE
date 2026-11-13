import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(8, channels), nn.SiLU(),
                                 nn.Conv2d(channels, channels, 3, padding=1),
                                 nn.GroupNorm(8, channels), nn.SiLU(),
                                 nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return (x + self.net(x)) / math.sqrt(2)


class SpatialPool(nn.Module):
    def forward(self, x):
        h, w = x.shape[-2:]
        rows = []
        for i in range(4):
            cells = []
            for j in range(4):
                area = x[..., i*h//4:math.ceil((i+1)*h/4), j*w//4:math.ceil((j+1)*w/4)]
                cells.append(area.mean((-2,-1)))
            rows.append(torch.stack(cells,-1))
        return torch.stack(rows,-2)


class TimeBlock(nn.Module):
    def __init__(self, width=64, blocks=3, output=80):
        super().__init__()
        self.input = nn.Conv2d(80, width, 1)
        self.time = nn.Sequential(nn.Linear(17, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.Sequential(*[Residual(width) for _ in range(blocks)])
        layer = nn.TransformerEncoderLayer(width, 4, width*2, dropout=0., activation="gelu",
                                            batch_first=True, norm_first=True)
        self.attention = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.output = nn.Conv2d(width, output, 1)

    def forward(self, z, t):
        onehot = F.one_hot(z.long().reshape(-1, 16, 4, 4), 5).permute(0, 1, 4, 2, 3).flatten(1, 2).float()
        f = torch.arange(1, 9, device=z.device).float()[None] * math.pi
        time = torch.cat([t[:, None], (t[:, None]*f).sin(), (t[:, None]*f).cos()], 1)
        h = self.blocks(self.input(onehot) + self.time(time)[:, :, None, None])
        with sdpa_kernel(SDPBackend.MATH):
            h = self.attention(h.flatten(2).transpose(1, 2)).transpose(1, 2).reshape_as(h)
        return self.output(h).reshape(-1, 16, 5, 4, 4).permute(0, 1, 3, 4, 2).reshape(-1, 256, 5)


class BirthDeathBridge(nn.Module):
    def __init__(self, reference_rate=5.):
        super().__init__()
        self.net = TimeBlock()
        q = torch.zeros(5, 5)
        for k in range(5):
            for j in [k-1, k+1]:
                if 0 <= j < 5:
                    q[k, j] = reference_rate
            q[k, k] = -q[k].sum()
        eigenvalues, eigenvectors = torch.linalg.eigh(q.double())
        self.register_buffer("eigenvalues", eigenvalues.float())
        self.register_buffer("eigenvectors", eigenvectors.float())
        self.reference_rate = reference_rate

    def kernel(self, times):
        u = self.eigenvectors
        k = (u[None] * (times[:, None] * self.eigenvalues[None]).exp()[:, None, :]) @ u.T
        k = k.clamp_min(1e-10)
        return k / k.sum(-1, keepdim=True)

    @torch.no_grad()
    def couple(self, source, target):
        
        
        log_kernel = self.kernel(torch.ones(1, device=source.device))[0].log()
        a = F.one_hot(source.long(), 5).float().flatten(1)
        b = (-log_kernel[:, target.long()]).permute(1, 2, 0).reshape(len(target), -1)
        log_plan = -(a @ b.T)
        u = torch.zeros(len(source), device=source.device)
        v = torch.zeros(len(target), device=source.device)
        for _ in range(40):
            u = -torch.logsumexp(log_plan + v[None], dim=1)
            v = -torch.logsumexp(log_plan + u[:, None], dim=0)
        plan = (log_plan + u[:, None] + v[None]).softmax(-1)
        selected = torch.multinomial(plan, 1).squeeze(1)
        return target[selected]

    def training_loss(self, source, target):
        target = self.couple(source, target)
        t = torch.rand(len(source), device=source.device) * .98 + .01
        left, right = self.kernel(t), self.kernel(1-t)
        ids = torch.arange(len(source), device=source.device)[:, None]
        bridge = left[ids, source.long()] * right[ids, target.long()]
        bridge = bridge / bridge.sum(-1, keepdim=True)
        zt = torch.multinomial(bridge.reshape(-1, 5), 1).reshape_as(source)
        return F.cross_entropy(self.net(zt, t).reshape(-1, 5), target.long().flatten())

    @torch.no_grad()
    def sample(self, source, steps=64):
        z = source.long().clone()
        ids = torch.arange(len(z), device=z.device)[:, None]
        dt = 1 / steps
        for step in range(steps):
            t = torch.full((len(z),), step / steps, device=z.device)
            endpoint = self.net(z, t).softmax(-1)
            remaining = self.kernel((1-t).clamp_min(dt))
            denominator = remaining[ids, z].clamp_min(1e-10)
            plus, minus = (z+1).clamp_max(4), (z-1).clamp_min(0)
            up = self.reference_rate * (endpoint * remaining[ids, plus] / denominator).sum(-1)
            down = self.reference_rate * (endpoint * remaining[ids, minus] / denominator).sum(-1)
            up = up * (z < 4)
            down = down * (z > 0)
            total = (up + down).clamp_min(1e-10)
            jump = torch.rand_like(total) < -torch.expm1(-total*dt)
            direction = torch.where(torch.rand_like(total) < up/total, 1, -1)
            z = z + jump * direction
        return z


class CountEnergy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(16, 64, 3, padding=1), Residual(64), Residual(64),
                                 nn.Flatten(), nn.Linear(1024, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, z):
        return self.net(z.float().reshape(-1, 16, 4, 4) / 4).flatten()


def metropolis(energy, proposals, policy_gradient=False):
    z = proposals[0]
    current = energy(z)
    score = torch.zeros(len(z), device=z.device)
    accepted = torch.zeros((), device=z.device)
    for candidate in proposals[1:]:
        other = energy(candidate)
        log_accept = (current - other).clamp_max(0)
        accept = torch.rand_like(log_accept) < log_accept.exp()
        if policy_gradient:
            log_reject = torch.log1p(-log_accept.exp().clamp_max(1-1e-6))
            score = score + torch.where(accept, log_accept, log_reject)
        z = torch.where(accept[:, None], candidate, z)
        current = torch.where(accept, other, current)
        accepted = accepted + accept.float().mean()
    return z, score, accepted / (len(proposals)-1)

