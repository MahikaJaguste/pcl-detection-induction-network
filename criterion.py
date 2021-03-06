import torch
from torch.nn.modules.loss import _Loss


class Criterion(_Loss):
    def __init__(self, way=5, shot=5):
        super(Criterion, self).__init__()
        self.amount = way * shot

    def forward(self, probs, target):  # (Q,C) (Q)
        target = target[self.amount:]
        # print("--- target in criterion : ", target[self.amount:self.amount+3])
        target_onehot = torch.zeros_like(probs)
        target_onehot = target_onehot.scatter(1, target.reshape(-1, 1), 1)
        # print("--- target_onehot in criterion : ", target_onehot[self.amount:self.amount+3])
        # print("probs shape", len(probs))
        # print("probs = ", probs)
        # print("target shape = ", target.shape[0])
        # print("target_onehot shape = ", target_onehot.shape[0])
        loss = torch.mean((probs - target_onehot) ** 2)
        pred = torch.argmax(probs, dim=1)
        # print("\nPrinting target,pred")
        # print(target,pred)
        # print("pred = ", pred)
        acc = torch.sum(target == pred).float() / target.shape[0]
        return loss, acc
