# Loss functions
# MMD, Domain adversarial, Classification losses

from .mmd_loss import MMDLoss
from .domain_adversarial_loss import DomainAdversarialLoss

__all__ = ["MMDLoss", "DomainAdversarialLoss"]
