import argparse
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from inference.counts import CountEnergy, metropolis
from r01_model import build_r01_model


def frequency_loss(x, y):
    def bands(image):
        a = image[..., ::2, ::2]
        b = image[..., ::2, 1::2]
        c = image[..., 1::2, ::2]
        d = image[..., 1::2, 1::2]
        return torch.cat(((a-b+c-d)/2, (a+b-c-d)/2, (a-b-c+d)/2), dim=1)

    return (bands(x) - bands(y)).abs().mean()


def image_loader(path, batch_size, shuffle):
    images = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or not images.is_floating_point():
        raise ValueError('Image data must be a floating-point tensor with shape [N,C,H,W].')
    if len(images) < 2:
        raise ValueError('Image data must contain at least two images.')
    return DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=shuffle)


def train_autoencoder(model, train, validation, device, epochs):
    ae = model.ae
    optimizer = torch.optim.AdamW(ae.parameters(), lr=1e-3, weight_decay=1e-4)
    best = float('inf')
    best_state = None
    for epoch in range(1, epochs + 1):
        ae.train()
        lr = 1e-3 * (.05 + .95 * (1 + math.cos(math.pi * (epoch - 1) / epochs)) / 2)
        optimizer.param_groups[0]['lr'] = lr
        tau = (0.35 ** min((epoch - 1) / 120, 1))
        beta = 3e-4 * min(epoch / 30, 1)
        for (batch,) in train:
            x = batch.to(device)
            reconstruction, _, info = ae(x, tau=tau)
            loss = F.mse_loss(reconstruction, x) + beta * info['kl'] + .005 * frequency_loss(reconstruction, x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        ae.eval()
        error = 0.
        with torch.no_grad():
            for (batch,) in validation:
                x = batch.to(device)
                error += F.mse_loss(ae(x)[0], x, reduction='sum').item()
        error /= len(validation.dataset) * validation.dataset.tensors[0][0].numel()
        print(f'autoencoder epoch={epoch} validation_mse={error:.6f}', flush=True)
        if error < best:
            best = error
            best_state = {k: v.detach().cpu().clone() for k, v in ae.state_dict().items()}
    ae.load_state_dict(best_state)
    ae.requires_grad_(False).eval()


@torch.no_grad()
def encode_data(model, loader, device):
    return torch.cat([model.encode(batch.to(device)).cpu() for (batch,) in loader])


def train_bridge(model, train_codes, validation_codes, device, steps, batch_size):
    bridge = model.bridge
    optimizer = torch.optim.Adam(bridge.parameters(), lr=3e-4)
    best = float('inf')
    best_state = None
    for step in range(steps):
        bridge.train()
        indices = torch.randint(len(train_codes), (min(batch_size, len(train_codes)),))
        target = train_codes[indices].to(device)
        with torch.no_grad():
            source = model.prior(len(target))
        loss = bridge.training_loss(source, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 250 == 0 or step + 1 == steps:
            bridge.eval()
            total = 0.
            with torch.no_grad():
                for target in validation_codes.split(batch_size):
                    target = target.to(device)
                    total += bridge.training_loss(model.prior(len(target)), target).item() * len(target)
            metric = total / len(validation_codes)
            print(f'bridge step={step + 1} validation_ce={metric:.6f}', flush=True)
            if metric < best:
                best = metric
                best_state = {k: v.detach().cpu().clone() for k, v in bridge.state_dict().items()}
    bridge.load_state_dict(best_state)
    bridge.requires_grad_(False).eval()


@torch.no_grad()
def proposal_pool(model, size, batch_size):
    return torch.cat([model.bridge.sample(model.prior(min(batch_size, size - start)), model.sampling_steps).cpu()
                      for start in range(0, size, batch_size)])


def train_energy(model, train_codes, pool, device, warmup, refinement, batch_size):
    energy = model.energy.train()
    critic = CountEnergy().to(device).train()
    optimizer = torch.optim.Adam(energy.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
    for step in range(warmup + refinement):
        count = min(batch_size, len(train_codes))
        real = train_codes[torch.randint(len(train_codes), (count,))].to(device)
        fake = pool[torch.randint(len(pool), (count,))].to(device)
        optimizer.zero_grad(set_to_none=True)
        if step < warmup:
            real_energy, fake_energy = energy(real), energy(fake)
            loss = F.softplus(real_energy).mean() + F.softplus(-fake_energy).mean()
            loss = loss + .001 * (real_energy.square().mean() + fake_energy.square().mean())
        else:
            indices = torch.randint(len(pool), ((model.mh_steps + 1) * count,))
            proposals = pool[indices].to(device).reshape(model.mh_steps + 1, count, 256)
            retained, log_policy, _ = metropolis(energy, proposals, policy_gradient=True)
            critic_optimizer.zero_grad(set_to_none=True)
            real_score, fake_score = critic(real), critic(retained.detach())
            critic_loss = F.softplus(-real_score).mean() + F.softplus(fake_score).mean()
            critic_loss = critic_loss + .001 * (real_score.square().mean() + fake_score.square().mean())
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.)
            critic_optimizer.step()
            with torch.no_grad():
                reward = critic(retained).sigmoid()
                advantage = (reward - reward.mean()) / reward.std(unbiased=False).clamp_min(.05)
            real_energy, fake_energy = energy(real), energy(fake)
            policy = -(advantage * log_policy).mean()
            contrastive = F.softplus(real_energy).mean() + F.softplus(-fake_energy).mean()
            loss = policy + .1 * contrastive + .001 * (real_energy.square().mean() + fake_energy.square().mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(energy.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 250 == 0 or step + 1 == warmup + refinement:
            print(f'energy step={step + 1} loss={loss.item():.6f}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['MNIST', 'fmnist', 'CIFAR16', 'CelebA64'], required=True)
    parser.add_argument('--train-images', type=Path, required=True)
    parser.add_argument('--validation-images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--ae-epochs', type=int, default=200)
    parser.add_argument('--bridge-steps', type=int, default=4000)
    parser.add_argument('--sampling-steps', type=int, default=64)
    parser.add_argument('--energy-warmup', type=int, default=1000)
    parser.add_argument('--energy-refinement', type=int, default=1000)
    parser.add_argument('--pool-size', type=int, default=32768)
    args = parser.parse_args()
    if min(args.batch_size, args.ae_epochs, args.bridge_steps, args.sampling_steps, args.pool_size) < 1 or min(args.energy_warmup, args.energy_refinement) < 0:
        parser.error('Batch size, epochs, bridge steps, sampling steps and pool size must be positive.')
    if args.output.exists():
        parser.error('Output path already exists.')
    device = torch.device(args.device)
    model = build_r01_model(args.dataset, device)
    model.sampling_steps = args.sampling_steps
    train = image_loader(args.train_images, args.batch_size, True)
    validation = image_loader(args.validation_images, args.batch_size, False)
    expected = (model.channels, model.size, model.size)
    if tuple(train.dataset.tensors[0].shape[1:]) != expected or tuple(validation.dataset.tensors[0].shape[1:]) != expected:
        parser.error(f'Expected image shape [N,{expected[0]},{expected[1]},{expected[2]}].')
    train_autoencoder(model, train, validation, device, args.ae_epochs)
    train_codes = encode_data(model, train, device)
    validation_codes = encode_data(model, validation, device)
    train_bridge(model, train_codes, validation_codes, device, args.bridge_steps, args.batch_size)
    pool = proposal_pool(model, args.pool_size, args.batch_size)
    train_energy(model, train_codes, pool, device, args.energy_warmup, args.energy_refinement, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'dataset': args.dataset, 'sampling_steps': args.sampling_steps, 'model': model.state_dict()}, args.output)


if __name__ == '__main__':
    main()
