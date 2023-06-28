import torch
import torch.nn.functional as F


def get_mask(partialY):
    pos = partialY.clone().bool()
    neg = torch.logical_not(pos)
    return pos, neg


def ce_loss(outputs, trueY):
    add = outputs * F.one_hot(trueY, outputs.size(1))
    s_outputs = torch.cat([outputs, add.sum(dim=-1, keepdim=True)], dim=-1)
    s_Y = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long, device=outputs.device)
    loss = F.cross_entropy(s_outputs, s_Y)
    return loss


def bc_wn_loss(outputs, partialY, args):
    # loss= - log(p(y)) + (1- y) log(1- p(y))
    # y = 1 holds for all examples
    # loss = - log(p(y))
    pos, neg = get_mask(partialY)
    sm_output = F.softmax(outputs / args.T, dim=1)

    new_pY = sm_output * pos
    new_pY = new_pY / new_pY.sum(dim=1, keepdim=True)

    # NaN preventation
    new_nY = sm_output * neg
    idx = torch.logical_not(new_nY.sum(dim=1) == 0)
    new_nY[idx] = new_nY[idx] / new_nY[idx].sum(dim=1, keepdim=True)
    new_nY[torch.logical_not(idx)] = 0
    pos_max = (new_pY.detach() * outputs).sum(dim=1, keepdim=True)
    neg_max = (new_nY.detach() * outputs).sum(dim=1, keepdim=True)

    s_outputs = torch.cat([pos_max, neg_max], dim=-1)
    s_Y = torch.full((s_outputs.size(0),), 0, dtype=torch.long, device=args.device)
    avg_loss = F.cross_entropy(s_outputs, s_Y)

    return avg_loss


def bc_avg_loss(outputs, partialY, compY, args):
    pos, neg = get_mask(partialY)
    sm_output = F.softmax(outputs / args.T, dim=1)

    pos_max = (partialY.detach() * outputs).sum(dim=1, keepdim=True)
    neg_max = (compY.detach() * outputs).sum(dim=1, keepdim=True)

    s_outputs = torch.cat([pos_max, neg_max], dim=-1)
    s_Y = torch.full((s_outputs.size(0),), 0, dtype=torch.long, device=args.device)
    avg_loss = F.cross_entropy(s_outputs, s_Y)

    return avg_loss


def bc_max_loss(outputs, partialY, args):
    pos, neg = get_mask(partialY)

    pos_max, _ = torch.max(outputs * pos + -1e-8 * neg, dim=-1)
    neg_max, _ = torch.max(outputs * neg + -1e-8 * pos, dim=-1)

    s_outputs = torch.cat([pos_max.reshape(-1, 1), args.T * neg_max.reshape(-1, 1)], dim=-1)
    s_Y = torch.full((s_outputs.size(0),), 0, dtype=torch.long, device=args.device)
    avg_loss = F.cross_entropy(s_outputs, s_Y)

    return avg_loss
