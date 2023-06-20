import torch
from net import *
from data import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
import os
weight_path = "./param/func.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
beta = np.pi/3
batch_size = 64

warmup_epoch = 500
epoch = 1500
initial_lr = 0.05  # 初始学习率
target_lr = 0.05  # 目标学习率

lr1 = 0.01
lr2 = 0.005
lr3 = 0.001

if __name__ == '__main__':
    net = MyRotateNet2(beta=beta).to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight!')
    else:
        print('not successful load weight!')

    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    lossF = torch.nn.MSELoss()

    data_loader = DataLoader(MyDataset(width=80, height=80, beta=beta), batch_size=batch_size, shuffle=True)
    epoch_losses = []

    # 定义学习率调度器
    warmup_lr_scheduler = LambdaLR(opt, lr_lambda=lambda epoch: initial_lr + (target_lr - initial_lr)*epoch / warmup_epoch if epoch < warmup_epoch else target_lr)
    lr_scheduler = LambdaLR(opt, lr_lambda=lambda epoch: lr1 if epoch <= 400 else lr2 if epoch<=700 else lr3)
    # warm-up
    for index in range(warmup_epoch):
        total_loss = 0.0
        for i, (xy, z) in enumerate(data_loader):
            xy, z = xy.to(device), z.to(device)
            _z = net(xy)
            loss = lossF(_z, z)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        epoch_losses.append(avg_loss)
        print(f"warm_up Epoch {index+1} loss: {avg_loss:.4f}")
        warmup_lr_scheduler.step()

    print("warm-up step finished")

    # train
    for index in range(epoch):
        total_loss = 0.0
        for i, (xy, z) in enumerate(data_loader):
            xy, z = xy.to(device), z.to(device)
            _z = net(xy)
            loss = lossF(_z, z)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {index+1} loss: {avg_loss:.4f}")
        lr_scheduler.step()


    # 绘制loss曲线图
    plt.plot(epoch_losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("./param/loss_curve.png")  # 保存loss曲线图


    torch.save(net.state_dict(), weight_path)