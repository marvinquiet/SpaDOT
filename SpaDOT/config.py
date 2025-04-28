from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class SpaDOTConfig:
    # General training parameters
    maxiter: int = 100
    ot_epoch: int = 50
    kmeans_epoch: int = 1
    batch_size: int = 512
    z_dim: int = 20
    n_clusters: int = 10
    # Learning rate and model architecture
    lr: float = 3e-4
    encoder_layers: List[int] = field(default_factory=lambda: [256, 64])
    decoder_layers: List[int] = field(default_factory=lambda: [64, 256])

    # SVGP-related parameters
    kernel_type: str = "Gaussian"
    inducing_point_nums: int = 1200
    lambda1: float = 0.1
    beta1: float = 1.0
    # GAT-related parameters
    lambda2: float = 0.0
    beta2: float = 1e-4
    # Alignment and loss weights
    omiga1: float = 0.1
    omiga2: float = 0.1
    omiga3: float = 1.0

    # File paths and directories
    model_file: str = "model.pt"
    final_latent_file: str = "final_latent.txt"
    result_dir: str = "./results"

    # Additional settings
    kernel_scale: List[float] = field(default_factory=lambda: [0.1])
    timepoints: List[str] = field(default_factory=list)

    # OT configuration
    ot_config: Dict[str, float] = field(default_factory=lambda: {
        "growth_iters": 3,
        "ot_epochs": 10,
        "epsilon": 0.05,
        "lambda1": 0.1,
        "lambda2": 5,
        "epsilon0": 1,
        "tau": 1000,
        "scaling_iter": 3000,
        "inner_iter_max": 50,
        "tolerance": 1e-8,
        "max_iter": 1e7,
        "batch_size": 5,
        "extra_iter": 1000,
        "numItermax": 1000000,
        "use_Py": False,
        "use_C": True,
        "profiling": False,
        "method": "waddington"
    })