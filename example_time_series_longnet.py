import torch
from dilated_attention_pytorch.long_net import LongNetTS

# @Author: Junchi Ma
# @Description: the longnet for time series example

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
dtype = torch.float32


# Initialize the model
model = LongNetTS(
    num_features=7,  # Number of features in the time series data
    d_model=128,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=512,
    segment_lengths=[6, 12, 24],  # Example segment lengths
    dilation_rates=[1, 2, 3],  # Example dilation rates
    dropout=0.1,
    activation="relu",
    layer_norm_eps=1e-5,
    pred_len=96,  # Number of prediction length(steps)
    device=device,
    dtype=dtype,
).to(device)

# Generate dummy input data (batch_size=32, seq_len=96, num_features=7)
batch_size, seq_len, num_features = 32, 96, 7
input_data = torch.randn(batch_size, seq_len, num_features, device=device, dtype=dtype)

# Generate dummy decoder input data (batch_size=32, seq_len=144, num_features=7)
# Assuming label_len is 48 for this example
label_len = 48
dec_inp = torch.zeros(batch_size, label_len + seq_len, num_features, device=device, dtype=dtype)
dec_inp[:, :label_len, :] = torch.randn(batch_size, label_len, num_features, device=device, dtype=dtype)

# Perform a forward pass
with torch.no_grad():
    output = model(input_data, dec_inp[:, -seq_len:, :], is_causal=True)

# Print the output shape
print(output.shape)  # Expected output shape: (32, 96, 7)
print(output)
