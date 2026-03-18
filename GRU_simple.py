import torch
import torch.nn as nn

input_feature = 8
hidden_units = 64
batch_size = 32
seq_length = 10

gru_model = nn.GRU(input_size= input_feature, hidden_size= hidden_units,
                   batch_first= True)

sample_input = torch.randn(batch_size, seq_length, input_feature)

output_seq_pt, final_hidden_state_pt = gru_model(sample_input)
"""
    output_seq_pt (Cuộn băng ghi hình): Chứa toàn bộ các trạng thái ẩn ở mọi bước thời gian. Kích thước: (batch_size, seq_len, hidden_size).

    final_hidden_state_pt (Ảnh chụp chốt sổ): Chứa trạng thái ẩn của bước cuối cùng (hoặc chốt sổ của tất cả các tầng nếu xếp chồng nhiều lớp kiểu có thể có nhiều lớp GRU). 
        Kích thước: (num_layers * num_directions, batch_size, hidden_size).
"""
# Output sequence shape: (batch_size, seq_length, hidden_size)
print(f"PyTorch Input shape: {sample_input.shape}")
print(f"PyTorch Output sequence shape: {output_seq_pt.shape}")

# Final hidden state shape: (num_layers * num_directions, batch_size, hidden_size)
# For a single-layer, unidirectional GRU, num_layers=1, num_directions=1.
print(f"PyTorch Final hidden state shape: {final_hidden_state_pt.shape}")

# Extracting the output of the very last time step for each sequence in the batch
last_step_output_pt = output_seq_pt[:, -1, :]
print(f"PyTorch Last time step output shape: {last_step_output_pt.shape}")
# Shape: (batch_size, hidden_size)