import numpy as np
import torch


def create_partial_label(one_hot_targets, targets, partial_type, partial_rate):
    new_y = torch.clone(one_hot_targets).float()
    data_num, cls_num = one_hot_targets.shape[0], one_hot_targets.shape[1]
    avgC = 0

    if partial_type == 'binomial':
        for i in range(data_num):
            while True:
                row = torch.clone(new_y[i])
                row[np.where(np.random.binomial(1, partial_rate, cls_num) == 1)] = 1
                # unlabeled data: all cls are positive
                if row.sum() == cls_num:
                    continue
                else:
                    break
            # unlabeled data:all cls are negative
            while torch.sum(row) == 1:
                row[np.random.randint(0, cls_num)] = 1
            avgC += torch.sum(row)
            new_y[i][row.bool()] = 1 / row.sum().float()

    if partial_type == 'pair':
        P = np.eye(cls_num)
        for idx in range(0, cls_num - 1):
            P[idx, idx], P[idx, idx + 1] = 1, partial_rate
        P[cls_num - 1, cls_num - 1], P[cls_num - 1, 0] = 1, partial_rate
        for i in range(data_num):
            row = new_y[i, :]
            idx = targets[i]
            row[np.where(np.random.binomial(1, P[idx, :], cls_num) == 1)] = 1
            avgC += torch.sum(row)
            new_y[i] = row / torch.sum(row)

    avgC = float(avgC) / data_num
    return new_y, avgC