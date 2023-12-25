import torch.nn as nn
from torchvision import transforms as T    ##导入模块


class bottleneck(nn.Module):
    # 搭建5层网络
    def __init__(self, input_features_dim, hidden_1=100, hidden_2=100, output=12): ##在用到Net时，可以自己手动根据不同的数据集进行改动各层的大小。
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(bottleneck, self).__init__()                        
        ##充分利用Sequential函数，将线性层Linear、BatchNormalization和激活函数层Tanh（）连接起来，从而构造一层全连接。
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(input_features_dim, hidden_1),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )
        self.output_layer = nn.Linear(hidden_2, output)
        self.Softmaxmax = nn.LogSoftmax(dim=-1)
        
        ##固定式的前向传播
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.output_layer(x)
        x = self.Softmaxmax(x)

        return x
