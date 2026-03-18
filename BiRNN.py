import torch
import torch.nn as nn

input_size = 10
hidden_size = 64
num_layers = 1

bi_gru__layer = nn.GRU(input_size= input_size,
                       hidden_size= hidden_size,
                       num_layers= num_layers,
                       bidirectional= True,
                       batch_first= True)

batch_size = 32
seq_len = 20

input_tensor = torch.randn(batch_size, seq_len, input_size)
output, hn = bi_gru__layer(input_tensor)

print("Output shape:", output.shape) # ở đâu nó ghép nối hai chiều ngược xuôi vs nhau
print("Final hidden state shape:", hn.shape) # ở hn nó sắp xếp riêng hai chiều ngược xuôi với nhau

# The 'output' tensor contains the concatenated forward and backward hidden states
# at each time step.
# output[batch, t, :hidden_size] is the forward state at time t
# output[batch, t, hidden_size:] is the backward state at time t

# The 'hn' tensor contains the final hidden states for each layer and direction.
# For a single layer BiGRU:
# hn[0, :, :] is the final forward hidden state h_T->
# hn[1, :, :] is the final backward hidden state h_1<- (from the start of the reversed sequence)