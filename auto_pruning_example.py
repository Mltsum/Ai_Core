from collections import OrderedDict

import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from mnncompress.pytorch.SNIP_level_pruner import SNIPLevelPruner
from mnncompress.pytorch.SIMD_OC_pruner import SIMDOCPruner
Pruner = SIMDOCPruner

"""
模型稀疏之后，可以进一步使用PyTorch训练量化工具进行量化，得到稀疏量化模型。

"""

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


# print(example_targets)
# print(example_data.shape)

# fig = plt.figure()
# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.title("Ground Truth: {}".format(example_targets[i]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

"""
继承 nn.Module
在 __init()__中定义网络的层
重写(override)父类的抽象方法forward()
"""
class Net(nn.Module):
    '''

    自定义的CNN网络，3个卷积层，包含batch norm。2个pool,
    3个全连接层，包含Dropout
    输入：28x28x1s
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.feature = nn.Sequential(
            OrderedDict(
                [
                    # 28x28x1
                    ('conv1', nn.Conv2d(in_channels=1,
                                        out_channels=32,
                                        kernel_size=5,
                                        stride=1,
                                        padding=2)),

                    ('relu1', nn.ReLU()),
                    ('bn1', nn.BatchNorm2d(num_features=32)),

                    # 28x28x32
                    ('conv2', nn.Conv2d(in_channels=32,
                                        out_channels=64,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)),

                    ('relu2', nn.ReLU()),
                    ('bn2', nn.BatchNorm2d(num_features=64)),
                    ('pool1', nn.MaxPool2d(kernel_size=2)),

                    # 14x14x64
                    ('conv3', nn.Conv2d(in_channels=64,
                                        out_channels=128,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)),

                    ('relu3', nn.ReLU()),
                    ('bn3', nn.BatchNorm2d(num_features=128)),
                    ('pool2', nn.MaxPool2d(kernel_size=2)),

                    # 7x7x128
                    ('conv4', nn.Conv2d(in_channels=128,
                                        out_channels=64,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)),

                    ('relu4', nn.ReLU()),
                    ('bn4', nn.BatchNorm2d(num_features=64)),
                    ('pool3', nn.MaxPool2d(kernel_size=2)),

                    # out 3x3x64

                ]
            )
        )

        self.classifier = nn.Sequential(


            OrderedDict(
                [
                    ('fc1', nn.Linear(in_features=3 * 3 * 64,
                                      out_features=128)),
                    ('dropout1', nn.Dropout2d(p=0.5)),

                    ('fc2', nn.Linear(in_features=128,
                                      out_features=64)),

                    ('dropout2', nn.Dropout2d(p=0.6)),

                    ('fc3', nn.Linear(in_features=64, out_features=10))
                ]
            )

        )

    def forward(self, x):
        out = self.feature(x)
        out = out.view(-1, 64 * 3 *3)
        out = self.classifier(out)
        return out


network = Net()
network.load_state_dict(torch.load("auto_pruning_model.pth"))
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

# 将模型进行转换，并使用转换后的模型进行训练，测试
pruner = SIMDOCPruner(network, total_pruning_iterations=1, sparsity=0.6, debug_info=False)


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # step之后调用pruner的剪枝方法
        pruner.do_pruning()
        print(pruner.current_prune_ratios())

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


print("开始训练...")
train(1)
test()  # 不加这个，后面画图就会报错：x and y must be the same size

print("开始多轮训练...")
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

# 保存MNN模型压缩参数文件，应在剪枝完毕之后进行，建议在保存模型时调用
pruner.save_compress_params("compress_params.bin", append=False)
print("初步训练结束...")

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')

# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)
# with torch.no_grad():
#     output = network(example_data)
# fig = plt.figure()
# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()


# ----------------------------------------------------------- #
#
# continued_network = Net()
# continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
#
# network_state_dict = torch.load('model.pth')
# continued_network.load_state_dict(network_state_dict)
# optimizer_state_dict = torch.load('optimizer.pth')
# continued_optimizer.load_state_dict(optimizer_state_dict)
#
# # 注意不要注释前面的“for epoch in range(1, n_epochs + 1):”部分，
# # 不然报错：x and y must be the same size
# # 为什么是“4”开始呢，因为n_epochs=3，上面用了[1, n_epochs + 1)
# for i in range(4, 9):
#     test_counter.append(i*len(train_loader.dataset))
#     train(i)
#     test()
#
# fig = plt.figure()
# plt.plot(train_counter, train_losses, color='blue')
# plt.scatter(test_counter, test_losses, color='red')
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
# plt.xlabel('number of training examples seen')
# plt.ylabel('negative log likelihood loss')
# plt.show()
