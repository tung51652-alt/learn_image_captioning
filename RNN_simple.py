import torch 
import torch.nn as nn


# define RNN layer
"""
    cái này mới chỉ chạy để hiểu đầu ra và đầu vào của RNN cũng như Tensor trong pytorch
"""
# input_size: số lượng đặc trưng x với mỗi bước lặp
# hidden_size: số lượng đặc trưng của trạng thái ẩn h
# batch_first = True (để đưa dữ liệu ở 3D tensor (batch_size, sequence_length, features))

rnn_layer = nn.RNN(input_size= 10, hidden_size= 20, batch_first= True)

batch_sizes = 5
sequence_length = 15
input_features = 10

# tạo ra giá trị ngẫu nhiên đê kiểm tra đầu ra và đầu vào của RNN
# đầu vào đoạn này tạo ra ko gian Tensor để nhét thử vô RNN
"""
    nhét 5 câu caption (batch_size), có độ dài max là 15 (sequence_length) tất cả đều phải bằng nhau (nhờ padding)
    số chiều của vector feature đó là 10 chiều nghĩa là ở mỗi chữ trong câu sẽ có 10 con số chỉ đăng trưng ở mỗi chữ ngay tại đó 
    (sau này vào thực sẽ sửa cho đúng)
"""
dummy_input = torch.randn(batch_sizes, sequence_length, input_features)

"""
    tạo ra một trạng thái ban đầu giả 
    tại sao lại có sự khác nhau về kích thức giữa input và state này
    vì RNN của pytorch
    - khi batch_first = true nghĩa là đầu vào và đầu ra thì batch sẽ ở đầu (output cũng phải thế)
    (để sau này xử lý song song cho 5 câu cùng lúc chăng hoặc lần lượt nhưng theo tôi hiểu về batch khi train thì là song song có thể ở RNN sẽ khác)

    - với initial_hidden_state cái batch_first không ảnh hưởng nên theo đúng chuẩn pytorch là (num_layers, batch_size, hidden_size) 
    (mỗi lần chạy nó nhúp cả dòng ký ức luôn không cần chạy từng chữ )


    ví dụ dễ hiểu khi xử lý nhé nó sẽ nhúp t = 0 của cả 5 câu trc rồi nó nhúp đúng 5 dòng state đó nên nó sẽ để là như thế
"""
initial_hidden_state = torch.randn(1, batch_sizes, 20)

output, final_hidden_state = rnn_layer(dummy_input, initial_hidden_state) # output này chính là ht của cả đầu vào


print("Input shape:", dummy_input.shape)         # torch.Size([5, 15, 10])
print("Output shape:", output.shape)             # torch.Size([5, 15, 20])
print("Final hidden state shape:", final_hidden_state.shape) # torch.Size([1, 5, 20])


