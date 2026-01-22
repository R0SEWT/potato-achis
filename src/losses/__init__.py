# Loss functions
# MMD, Domain adversarial, Classification losses

from .domain_adversarial_loss import DomainAdversarialLoss
from .mmd_loss import MMDLoss

__all__ = ["MMDLoss", "DomainAdversarialLoss"]
