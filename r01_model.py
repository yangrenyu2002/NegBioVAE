from inference import UnifiedR01


def build_r01_model(dataset="CIFAR16", device="cpu"):
    datasets = {
        "mnist": (1, 28),
        "fmnist": (1, 28),
        "fashionmnist": (1, 28),
        "fashion-mnist": (1, 28),
        "cifar16": (3, 16),
        "celeba64": (3, 64),
    }
    channels, size = datasets[dataset.lower()]
    config = {
        "states": 5,
        "soft": False,
        "adaptive": False,
        "residual": False,
        "wide": False,
    }
    return UnifiedR01(config, channels=channels, size=size).to(device).eval()
