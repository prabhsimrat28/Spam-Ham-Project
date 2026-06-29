import torch.nn as nn
VOCAB_SIZE=8531

class mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(VOCAB_SIZE,100)
        self.lstm=nn.LSTM(input_size=100,hidden_size=128,batch_first=True)
        self.dropout=nn.Dropout(0.5)
        self.fc1=nn.Linear(128,2)

    def forward(self,x):
        x=self.embedding(x)
        output, (h_n, c_n)=self.lstm(x)
        final_state=self.dropout(h_n[-1])
        final_state=self.fc1(final_state)
        return final_state       