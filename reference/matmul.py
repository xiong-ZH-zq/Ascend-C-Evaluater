import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs matrix multiplication with bias (C = A * B + bias)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication with bias.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).
            bias: Bias tensor of shape (N,).

        Returns:
            Output tensor of shape (M, N).
        """
        return torch.matmul(A, B) + bias

M = 1024
K = 256
N = 640

def get_inputs():
    A = torch.randn(M, K, dtype=torch.float16)
    B = torch.randn(K, N, dtype=torch.float16)
    bias = torch.randn(N, dtype=torch.float32)
    return [A, B, bias]

def get_init_inputs():
    return []  # No special initialization inputs needed