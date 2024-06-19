import torch
from dilated_attention_pytorch.long_net import LongNetTS

# @Author: Junchi Ma
# @Description: the longnet for time series example

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
dtype = torch.float16

# Initialize the model
model = LongNetTS(
    num_features=7,  # Number of features in the time series data
    d_model=128,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=512,
    segment_lengths=[10, 20, 30, 40, 50],  # Example segment lengths
    dilation_rates=[1, 2, 3, 4, 5],  # Example dilation rates
    dropout=0.1,
    activation="relu",
    layer_norm_eps=1e-5,
    pred_len=10,  # Number of prediction length(steps)
    device=device,
    dtype=dtype,
).to(device)

# Generate dummy input data (batch_size=32, seq_len=50, num_features=7)
input_data = torch.randn(32, 50, 7, device=device, dtype=dtype)

# Perform a forward pass
with torch.no_grad():
    output = model(input_data, is_causal=True)

# Print the output shape
print(output.shape)  # Expected output shape: (32, 10, 1)
