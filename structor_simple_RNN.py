import torch
import torch.nn as nn


"""
    xây dựng một RNN đơn giản 
    sẽ có 3 lớp (đã ghi chi tiết ở vở)
    embedding, RNN (simple), Dense

"""
vocab_size = 10000
embedding_dim = 16
rnn_units = 32
num_classes = 2
input_features = 5

# ví dụ khi có embedding layer
class SimpleRNNClassifier(nn.Module):
    # bắt buộc phải kế thừa nn.Module để nó lưu trữ lại các ma trận trọng số khi ta chạy hàm init
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_classes):
        super().__init__()

        # tạo lớp embedding
        # tạo ra quấn từ điến với 10000 từ vựng (vocab_size) và mỗi từ có vector dài 16 số
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # tạo lớp cho model RNN
        # đầu vào là embedding_dim, sức chứa trạng thái là rnn_units
        self.rnn = nn.RNN(embedding_dim, rnn_units, batch_first= True)

        # tạo lớp Dense
        # đây là lớp fully-connected do bài toán này ta quy định 
        # đầu vào là trạng thái sau khi đã xử lý cả từ rồi ép về còn 2 chiều 
        # đầu ra của Liear là các con só nguyên có âm, dương và không
        self.fc = nn.Linear(rnn_units, num_classes)

    def forward(self, x):

        """
            hàm này nhận đầu vào là x rồi đưa ra rồi phân loại xem nó sẽ là gì?
        """

        # nhét đầu vào vô embedding để nó tính ra dạng (batch_size, seq_len, 16)
        embedded = self.embedding(x)

        # từ đầu vào như thế (có cái batch_size ở đầu nên lúc cái rnn mới đến batch_first = True)
        # đưa cái embedded đó vào rnn để nó đưa ra output và trạng thái sau khi phân tích được hết
        output, hidden = self.rnn(embedded)

        # cái hidden khi đi ra thì size nó là 1, batch_size, 32
        # mà lớp Linear ko đọc khối 3D nên phải bỏ đi cái 1 đầu tiên vì vậy cần squeeze(0)
        final_hidden = hidden.squeeze(0) # -> đầu ra là (batch_size, 32)


        out = self.fc(final_hidden) # output sẽ là (batch_Size, 2) 
        # ở đây mới chỉ có số thôi nhé
        """
            để ra đáp án phân loại cần thêm nn.CrossEntropyLoss()
            khi này là hàm vừa tính loss và vừa tính softmax trong này
            why? đây là tối ưu của pytorch để không bị tràn bộ nhớ khi tính toán
        """

        return out

# ví dụ khi không có embedding layer
class SimpleRNNRegressor(nn.Module):
    """
        đây là bài toán mà vốn dữ liệu đầu vào của nó đã là một con số có ý nghĩa sẵn rồi
        và có dạng (batch_size, seq_len, input_feat) nên sẽ cắm thẳng vô RNN luôn

        bài toán hồi quy là kiểu dự đoán từ hoặc chữ gõ tiếp theo ý
    """
    def __init__(self, input_features, rnn_units):
        super().__init__()

        self.rnn = nn.RNN(input_features, rnn_units, batch_first= True)
        self.fc = nn.Linear(rnn_units, 1)
    
    def forward(self, x):

        output, hidden = self.rnn(x)

        final__hidden = hidden.squeeze(0)

        out = self.fc(final__hidden)

        return out

model_text_pt = SimpleRNNClassifier(vocab_size, embedding_dim, rnn_units, num_classes)
model_numeric_pt = SimpleRNNRegressor(input_features, rnn_units)

# Print the model structure
print("PyTorch Text Model Structure:")
print(model_text_pt)

print("\nPyTorch Numeric Model Structure:")
print(model_numeric_pt)