import torch

from dilated_attention_pytorch.long_net import LongNet

device = torch.device("cuda")
dtype = torch.float32

# NOTE: Showing all default values, which are described in the paper.
net = LongNet(
    d_model=768,
    nhead=12,
    num_encoder_layers=12,
    num_decoder_layers=12,
    dim_feedforward=3072,
    segment_lengths=[2048, 4096, 8192, 16384, 32768],
    dilation_rates=[1, 2, 4, 6, 12],
    dropout=0.0,
    activation="relu",
    layer_norm_eps=1e-5,
    device=device,
    dtype=dtype,
)
# shape: (batch_size, seq_len, d_model)
x = torch.randn(1, 32768, 768, device=device, dtype=dtype)
with torch.no_grad():
    y = net.forward(x, is_causal=True)  # default: is_causal=True
print(y.shape)
print(y)
# torch.Size([1, 32768, 768])
