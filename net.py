import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
三层MLP
'''
class MyMlp(nn.Module):
    def __init__(self, size=2) -> None:
        super(MyMlp, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(size, 64),nn.ReLU(),
            nn.Linear(64, 16),nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x):
        return self.layer(x)

'''
使用残差网络
'''
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1), nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1), nn.ReLU()
        )

    def forward(self, x):
        residual = x
        out = self.layer(x)
        out = out + residual
        out = F.relu(out)
        return out
class MyRes(nn.Module):
    def __init__(self, size=2, num_blocks=3) -> None:
        super(MyRes, self).__init__()
        self.first_layer = nn.Conv1d(1, 64, kernel_size=1)
        self.blocks = nn.Sequential(
            *[ResBlock(64, 64) for _ in range(num_blocks)]
        )
        self.last_layer = nn.Sequential(
            nn.Conv1d(64, 1, kernel_size=1),
            nn.Linear(size, 1)
        ) 
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        out = self.first_layer(x.unsqueeze(1))
        out = self.blocks(out)
        out = self.last_layer(out)
        return out.squeeze(1)
    
'''
使用RNN
'''
class MyRNN(nn.Module):
    def __init__(self, hidden_size=16) -> None:
        super(MyRNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=1) # 1D convolution layer
        self.rnn1 = nn.RNN(input_size=8, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.rnn2 = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.layer = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # add channel dimension for 1D convolution
                            # (batch_size, 1, 2)
        out = self.conv(x)  # (batch_size, 8, 2)
        out = out.permute(0, 2, 1)  # (batch_size, 2, 8)
        h1 = torch.zeros(1, x.size(0), self.rnn1.hidden_size).to(x.device)
        out, hn1 = self.rnn1(out, h1) # (batch_size, 2, 16)
        h2 = torch.zeros(1, x.size(0), self.rnn2.hidden_size).to(x.device)
        out, hn2 = self.rnn2(out, h2) # (batch_size, 2, 16)
        out = self.layer(out[:, -1, :]) # (batch_size, 1)
        return out

'''
使用自注意力机制
'''
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size=16, num_heads=4) -> None:
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        self.q_linear = nn. Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        
        self.ln1 = nn.BatchNorm1d(num_features=self.hidden_size)

        self.dropout = nn.Dropout(p=0.5)
        self.out_linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.ln2 = nn.BatchNorm1d(num_features=self.hidden_size)
        
    def forward(self, x): 
        batch_size = x.shape[0]
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_size)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_size)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_size)
        
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32))  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = self.dropout(attention_weights)
        
        out = torch.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len, head_size)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)  # (batch_size, seq_len, hidden_size)
        out = self.ln1(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = out + x
        res = out
        out = self.out_linear(out)  # (batch_size, seq_len, hidden_size)
        out = self.ln2(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = out + res
        return out

class MyAttNet(nn.Module):
    def __init__(self,size=2, seq_len=64 , hidden_size=8, num_heads=4) -> None:
        super(MyAttNet, self).__init__()
        self.input = nn.Linear(size, seq_len)
        self.conv = nn.Conv1d(in_channels=1, out_channels=hidden_size, kernel_size=1) # 1D convolution layer
        self.multi_head_attention1 = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.multi_head_attention2 = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.multi_head_attention3 = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.layer = nn.Sequential(
            nn.Linear(hidden_size*seq_len, 1)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x):
        out = self.input(x)
        out = out.unsqueeze(1)  # (batch_size, 1, seq_len)
        out = self.conv(out)  # (batch_size, hidden_size, seq_len)
        out = out.permute(0, 2, 1)  # (batch_size, seq_len, hidden_size)
        out = self.multi_head_attention1(out)  # (batch_size, seq_len, hidden_size)
        out = self.multi_head_attention2(out)  # (batch_size, seq_len, hidden_size)
        out = self.multi_head_attention3(out)  # (batch_size, seq_len, hidden_size)
        out = out.reshape(out.size(0), -1)  # (batch_size, seq_len * hidden_size)
        out = self.layer(out)  # (batch_size, 1)
        return out

'''
使用镜像网络
'''
class MirrorEncoder(nn.Module):
    def __init__(self,size=2, seq_len=64) -> None:
        super(MirrorEncoder, self).__init__()
        self.size = size

        self.input = nn.Sequential(
            nn.Linear(1, seq_len), nn.ReLU(),
            nn.Linear(seq_len, seq_len) 
        )
        self.multi_head_attention = MultiHeadAttention(hidden_size=1, num_heads=1)

    def forward(self, x):
        out = torch.zeros_like(x[:, 0].view(-1, 1)) # (batch_size, seq_len)
        for i in range(self.size):
            sub_out = self.input(x[:, i].view(-1, 1))
            out = out + sub_out
        return out

class MyMirrorNet(nn.Module):
    def __init__(self,size=2, seq_len=64 , hidden_size=8, num_heads=4) -> None:
        super(MyMirrorNet, self).__init__()
        self.size = size

        self.encoder = MirrorEncoder(size=size, seq_len= seq_len)
        self.conv = nn.Conv1d(in_channels=1, out_channels=hidden_size, kernel_size=1) # 1D convolution layer
        self.multi_head_attention1 = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.multi_head_attention2 = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.multi_head_attention3 = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.layer = nn.Sequential(
            nn.Linear(hidden_size*seq_len, 1)
        )
        
    def forward(self, x):
        out = self.encoder(x)
        out = out.unsqueeze(1)  # (batch_size, 1, seq_len)
        out = self.conv(out)  # (batch_size, hidden_size, seq_len)
        out = out.permute(0, 2, 1)  # (batch_size, seq_len, hidden_size)
        out = self.multi_head_attention1(out)  # (batch_size, seq_len, hidden_size)
        out = self.multi_head_attention2(out)  # (batch_size, seq_len, hidden_size)
        out = self.multi_head_attention3(out)  # (batch_size, seq_len, hidden_size)
        out = out.reshape(out.size(0), -1)  # (batch_size, seq_len * hidden_size)
        out = self.layer(out)  # (batch_size, 1)
        return out

'''
使用旋转不变性网络
'''
def rotate(data, beta):
    x = data[:, 0]  # (batch_size,)
    y = data[:, 1]  # (batch_size,)
    angles = torch.atan2(y, x)  # 计算原始点的角度
    radii = torch.sqrt(x**2 + y**2)  # 计算原始点的长度
    angles = angles + beta
    a = radii * torch.cos(angles)  # 计算旋转后的点的x坐标
    b = radii * torch.sin(angles)  # 计算旋转后的点的y坐标
    combined_array = torch.stack([a, b], dim=1)
    return combined_array

class RotateEncoder(nn.Module):
    def __init__(self, input_size=2, seq_len=64, beta=np.pi/3) -> None:
        super(RotateEncoder, self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.beta = beta
        self.input = nn.Sequential(
            nn.Linear(input_size, seq_len), nn.ReLU(),
            nn.Linear(seq_len, seq_len) 
        )

    def forward(self, x):
        out = torch.zeros((len(x), self.seq_len)).to(device)
        for _ in range(int(2*np.pi/self.beta)):
            out = out + self.input(x)
            x = rotate(x, beta= self.beta)
        return out

class MyRotateNet(nn.Module):
    def __init__(self, input_size=2, seq_len=64, beta=np.pi/3, hidden_size=8, num_heads=4) -> None:
        super(MyRotateNet, self).__init__()
        self.input_size = input_size

        self.encoder = RotateEncoder(input_size=input_size, seq_len=seq_len, beta=beta)
        self.conv = nn.Conv1d(in_channels=1, out_channels=hidden_size, kernel_size=1) # 1D convolution layer
        self.multi_head_attention1 = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.multi_head_attention2 = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.multi_head_attention3 = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.layer = nn.Sequential(
            nn.Linear(hidden_size*seq_len, 1)
        )
        
    def forward(self, x):
        out = self.encoder(x)
        out = out.unsqueeze(1)  # (batch_size, 1, seq_len)
        out = self.conv(out)  # (batch_size, hidden_size, seq_len)
        out = out.permute(0, 2, 1)  # (batch_size, seq_len, hidden_size)
        
        out = self.multi_head_attention1(out)  # (batch_size, seq_len, hidden_size)
        out = self.multi_head_attention2(out)  # (batch_size, seq_len, hidden_size)
        out = self.multi_head_attention3(out)  # (batch_size, seq_len, hidden_size)
        
        out = out.reshape(out.size(0), -1)  # (batch_size, seq_len * hidden_size)
        out = self.layer(out)  # (batch_size, 1)
        return out

'''旋转不变性网络-直接旋转点的位置至0~beta后进行训练/预测'''
class MyRotateNet2(nn.Module):
    def __init__(self, input_size=2, seq_len=64, beta=np.pi/3, hidden_size=8, num_heads=4) -> None:
        super(MyRotateNet2, self).__init__()
        self.input_size = input_size
        self.beta = beta
        self.input = nn.Sequential(
            nn.Linear(input_size, seq_len), nn.ReLU(),
            nn.Linear(seq_len, seq_len) 
        )
        self.conv = nn.Conv1d(in_channels=1, out_channels=hidden_size, kernel_size=1) # 1D convolution layer
        self.multi_head_attention1 = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.multi_head_attention2 = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.multi_head_attention3 = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.layer = nn.Sequential(
            nn.Linear(hidden_size*seq_len, 1)
        )

    # 旋转batch内所有的点到0~beta角度内
    def transform(self, data):
        x = data[:, 0]  # (batch_size,)
        y = data[:, 1]  # (batch_size,)
        angles = torch.atan2(y, x)  # 计算原始点的角度
        radii = torch.sqrt(x**2 + y**2)  # 计算原始点的长度
        while torch.any( angles<=2*torch.pi ):
            angles = torch.where(angles<=2*torch.pi, angles+self.beta, angles)
        
        a = radii * torch.cos(angles)  # 计算旋转后的点的x坐标
        b = radii * torch.sin(angles)  # 计算旋转后的点的y坐标
        combined_array = torch.stack([a, b], dim=1)
        return combined_array

    def forward(self, x):
        x = self.transform(data=x)
        out = self.input(x)
        out = out.unsqueeze(1)  # (batch_size, 1, seq_len)
        out = self.conv(out)  # (batch_size, hidden_size, seq_len)
        out = out.permute(0, 2, 1)  # (batch_size, seq_len, hidden_size)
        
        out = self.multi_head_attention1(out)  # (batch_size, seq_len, hidden_size)
        out = self.multi_head_attention2(out)  # (batch_size, seq_len, hidden_size)
        out = self.multi_head_attention3(out)  # (batch_size, seq_len, hidden_size)
        
        out = out.reshape(out.size(0), -1)  # (batch_size, seq_len * hidden_size)
        out = self.layer(out)  # (batch_size, 1)
        return out

if __name__ == '__main__':
    device = 'cpu'
    
    x = torch.randn(8,2,dtype=torch.float32)
    net = MyRotateNet2(beta=np.pi/3)
    print("Input  Shape:", x.shape)
    print("Output Shape:", net(x).shape)


    # net = MyRotateNet2(beta=np.pi/2).eval()
    # x1 = torch.tensor([1,1], dtype=torch.float32).view(1, 2)
    # x2 = torch.tensor([-1,1], dtype=torch.float32).view(1, 2)
    # x3 = torch.tensor([1,-1], dtype=torch.float32).view(1, 2)
    # x4 = torch.tensor([-1,-1], dtype=torch.float32).view(1, 2)
    # print(net(x1), net(x2), net(x3), net(x4))