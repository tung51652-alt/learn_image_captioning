import torch
import torch.nn as nn

"""
    tạo ra một lớp LSTM
"""

input_size = 8 # kích thước vector đầu vào mỗi nhịp thời gian xt
hidden_size = 64 # kích thước trạng thái ẩn ht và Ct
seq_len = 10
batch_size = 32

lstm_layer = nn.LSTM(input_size= input_size, 
                     hidden_size = hidden_size,
                     batch_first= True) # cái này giải thích ở RNN đơn giản rồi

"""
    dropout: Thêm lớp Dropout để chống học vẹt (overfitting) 
            nếu bạn dùng nhiều tầng LSTM.

    bidirectional: Nếu đặt True, mô hình sẽ đọc câu từ trái sang phải, 
                    rồi lại đọc từ phải sang trái (LSTM hai chiều).
"""
input_data = torch.randn(batch_size, seq_len, input_size)

output, (h_n, c_n) = lstm_layer(input_data)
"""
    output: Chứa toàn bộ các trạng thái ẩn (h_t) ở tất cả các bước thời gian. Hình dáng của nó là (batch_size, seq_len, hidden_size). 
        Lấy ví dụ trên, kích thước sẽ là (32, 15, 64). Nó là một cuộn băng ghi hình lại toàn bộ quá trình suy nghĩ của mô hình qua từng từ một.
        (cái này nó sẽ chứa kiểu [h1, h2, ... , hn])
    h_n (Hidden state cuối cùng): Chỉ chứa "trạng thái ẩn" ở bước thời gian cuối cùng (khi đã đọc xong từ cuối cùng của câu). 
        Hình dáng: (num_layers, batch_size, hidden_size). (cái này giống như kiểu trạng thái của vị trí cuối cùng của thằng đầu vào)
    c_n (Cell state cuối cùng): Chỉ chứa "cuốn sổ tay" ở bước thời gian cuối cùng. Hình dáng giống hệt h_n.

    tải sao lại có cả output và hn
    nếu sinh cả chuỗi hoặc phân loại => chỉ cần hn
    nếu sinh từng từ 1 => cần output vì nó lưu từng cái một 
"""

print("Input shape:", input_data.shape)
print("Output shape (all timesteps):", output.shape) # (batch, seq_len, hidden_size)
print("Final hidden state shape (h_n):", h_n.shape) # (num_layers*num_directions, batch, hidden_size)
print("Final cell state shape (c_n):", c_n.shape) # (num_layers*num_directions, batch, hidden_size)

# To get only the last time step's output from the 'output' tensor:
last_step_output = output[:, -1, :]
print("Last time step output shape:", last_step_output.shape) # (batch, hidden_size)
