import torch
import torch.nn as nn

class MtlClass(nn.Module):
    def __init__(self, num_task):
        super(MtlClass, self).__init__()
        # print(num_task)
        self.num_task = num_task
        self.log_deta2 = nn.Parameter(torch.zeros(num_task, dtype=torch.float32),  requires_grad=True)
        self.soft_max = nn.Softmax(dim=-1)
        # self.pre_weight = nn.Parameter(torch.zeros(num_task, dtype=torch.float32), requires_grad=False)

    def forward(self, losslist):
        pre_list = torch.zeros(self.num_task, dtype=torch.float32)
        total_loss = 0
        for i in range(self.num_task):
            pre = torch.exp(-self.log_deta2[i])
            pre_list[i] = pre
            # total_loss += torch.sum(pre*losslist[i] + self.log_deta2[i], dim=-1)
        print("pre_list is ", pre_list)
        pre_list = self.soft_max(pre_list)
        print("pre_list is ", pre_list)
        for i in range(self.num_task):
            print("the pre is ", pre_list[i])
            total_loss += torch.sum(pre_list[i]*losslist[i] + self.log_deta2[i], dim=-1)

        return total_loss