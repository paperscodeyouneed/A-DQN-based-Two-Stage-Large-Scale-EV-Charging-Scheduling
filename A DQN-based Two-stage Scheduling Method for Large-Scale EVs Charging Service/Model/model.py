# import sys
# sys.path.append("E:\EV_Charging_Scheduling")

from Environment.environment import *


class EvNet(nn.Module):
    """
        a conv net for DQN

        input_size = (batch_size, 1, 903, 36)
        output_size = (batch_size, 899)

    """
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(8, 8), stride=4)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))
        self.fc_1 = nn.Linear(6976, 4096)
        self.fc_2 = nn.Linear(4096, 2048)
        self.fc_3 = nn.Linear(2048, 1024)
        self.fc_4 = nn.Linear(1024, 899)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        forward propagation
        :param input_: an input element with shape (batch_size X 1-channel X 903 X 36)
        :return: 1-dimensional Tensor
        """
        x = self.conv_1(input_)
        x = f.relu(x)
        x = self.conv_2(x)
        x = f.relu(x)
        x = self.conv_3(x)
        x = f.relu(x)
        size = x.size()[1] * x.size()[2] * x.size()[3]
        x = x.view(-1, size)
        x = self.fc_1(x)
        x = f.relu(x)
        x = self.fc_2(x)
        x = f.relu(x)
        x = self.fc_3(x)
        x = f.relu(x)
        x = self.fc_4(x)
        return x


class CsNet(nn.Module):
    """
        a conv net for DQN
    """
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 6))
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 4))
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 4))
        self.fc_1 = nn.Linear(1600, 1024)
        self.fc_2 = nn.Linear(1024, 512)
        self.fc_3 = nn.Linear(512, 128)
        self.fc_4 = nn.Linear(128, 34)

    def forward(self,
                input_: torch.Tensor) -> torch.Tensor:
        """
        forward propagation
        :param input_: an input element with shape (batch_size X 1-channel X 5 X 36)
        :return: 1-dimensional Tensor
        """
        x = self.conv_1(input_)
        x = f.relu(x)
        x = self.conv_2(x)
        x = f.relu(x)
        x = self.conv_3(x)
        x = f.relu(x)
        size = x.size()[1] * x.size()[2] * x.size()[3]
        x = x.view(-1, size)
        x = self.fc_1(x)
        x = f.relu(x)
        x = self.fc_2(x)
        x = f.relu(x)
        x = self.fc_3(x)
        x = f.relu(x)
        x = self.fc_4(x)
        return x


if __name__ == '__main__':

    env = Environment()

    ev_state = env.get_current_ev_state()
    cs_state = env.get_current_cs_state(0)
    cs_net = CsNet()
    cs_net = cs_net.to("cuda")
    ev_net = EvNet()
    ev_net = ev_net.to("cuda")

    ev_state = torch.Tensor(ev_state).to("cuda").unsqueeze(0)
    ev_out = ev_net(ev_state)

    cs_state = torch.Tensor(cs_state).to("cuda").unsqueeze(0)
    cs_out = cs_net(cs_state)

    # print(ev_state)
    # print(cs_state)

    print("输入状态的形状")
    print(ev_state.shape)  # torch.Size([1, 1, 903, 36])
    print(cs_state.shape)  # torch.Size([2, 1, 5, 36])

    print("输出结果的形状")
    print(ev_out.shape)  # torch.Size([2, 899])
    print(cs_out.shape)  # torch.Size([2, 34])

