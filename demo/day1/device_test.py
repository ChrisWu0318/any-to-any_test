import torch

if torch.backends.mps.is_available():
    print("MPS backend available!")
    device = torch.device("mps")
else:
    print("MPS backend NOT available, using CPU.")
    device = torch.device("cpu")