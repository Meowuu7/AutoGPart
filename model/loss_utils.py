# Adapted from https://github.com/SimingYan/HPNet/blob/main/utils/loss_utils.py

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
# from utils.main_utils import npy, v
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from functools import wraps
import time
from torch.autograd import Variable
# from lapsolver import solve_dense

DIVISION_EPS = 1e-10


def v(var, cuda=True, volatile=False):
    if type(var) == torch.Tensor or type(var) == torch.DoubleTensor:
        res = Variable(var.float(), volatile=volatile)
    elif type(var) == np.ndarray:
        res = Variable(torch.from_numpy(var), volatile=volatile)
    if cuda:
        res = res.cuda()
    return res


def npy(var):
    return var.data.cpu().numpy()

def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{} : {}'.format(func.__module__, func.__name__, end - start))
        return r

    return wrapper


def get_one_hot(targets, nb_classes):
    # res = np.eye(nb_classes)[np.array(targets, dtype=np.int8).reshape(-1)]
    # check none-type
    # if targets.min() == -1:
    #   idx = np.argwhere(targets == -1)
    #   res[idx] = 0
    # return res.reshape(list(targets.shape)+[nb_classes])

    one_hot = nn.functional.one_hot(targets, nb_classes)

    return one_hot


def hungarian_matching(pred_x, gt_x, curnmasks, include_background=True):
    """ pred_x, gt_x: B x nmask x nsmp
        curnmasks: B
        return matching_idx: B x nmask x 2 """
    # batch_size = gt_x.shape[0]

    # print(pred_x.shape, gt_x.shape)
    gt_x = np.transpose(gt_x, axes=[1,0])
    pred_x = np.transpose(pred_x, axes=[1,0])

    nmask = gt_x.shape[0]

    matching_score = np.matmul(gt_x,np.transpose(pred_x, axes=[1,0])) # B x nmask x nmask
    matching_score = 1-np.divide(matching_score,
                                 np.expand_dims(np.sum(pred_x, -1), 0) + np.sum(gt_x, -1, keepdims=True) - matching_score+1e-8)
    matching_idx = np.zeros((nmask, 2)).astype('int32')
    curnmask = curnmasks
    # print(curnmask.shape)
    curnmask = int(curnmask)
    curnmask = min(curnmask, pred_x.shape[0])
    row_ind, col_ind = linear_sum_assignment(matching_score[:curnmask,:])
    # row_ind, col_ind = linear_sum_assignment(matching_score[i,:,:])
    matching_idx[:curnmask,0] = row_ind[:curnmask]
    matching_idx[:curnmask,1] = col_ind[:curnmask]
    return torch.from_numpy(matching_idx).long().cuda()


def compute_riou(W_pred, W_gt, pred_ind, gt_ind):
    # W_pred - NxK
    # W_gt - NxK'

    N, _ = W_pred.shape

    pred_ind = pred_ind.unsqueeze(0).repeat(N, 1)
    gt_ind = gt_ind.unsqueeze(0).repeat(N, 1)
    W_gt, W_pred = W_gt.cuda(), W_pred.cuda()

    W_pred_reordered = torch.gather(W_pred, -1, pred_ind)
    W_gt_reordered = torch.gather(W_gt, -1, gt_ind)

    dot = torch.sum(W_gt_reordered * W_pred_reordered, dim=0)  # K
    denominator = torch.sum(W_gt_reordered, dim=0) + torch.sum(
        W_pred_reordered, dim=0) - dot
    mIoU = dot / (denominator + DIVISION_EPS)  # K
    return mIoU


def compute_miou(cluster_pred, I_gt, return_recall=False):
    '''
    compute per-primitive riou loss
    cluster_pred: (1, N)
    I_gt: (1, N), must contains -1
    '''
    assert (cluster_pred.shape[0] == 1)

    one_hot_pred = get_one_hot(cluster_pred,
                               cluster_pred.max() + 1)[0]  # (N, K)

    if I_gt.min() == -1:
        one_hot_gt = get_one_hot(I_gt + 1,
                                 I_gt.max() +
                                 2)[0][:, 1:]  # (N, K'), remove background
    else:
        # print(I_gt.max() + 1)
        # one_hot_gt = get_one_hot(I_gt, I_gt.max() + 1)[0][:, :-1]
        # print(get_one_hot(I_gt, I_gt.max() + 1)[0].shape)
        one_hot_gt = get_one_hot(I_gt, I_gt.max() + 1)[0][:, :]

    curnmasks = torch.sum((torch.sum(one_hot_gt, dim=-1) > 0).float()).item()

    # pred_ind, gt_ind = hungarian_matching(npy(one_hot_pred), npy(one_hot_gt))
    inds = hungarian_matching(npy(one_hot_pred), npy(one_hot_gt), curnmasks)

    pred_ind, gt_ind = inds[:, 1], inds[:, 0]

    riou = compute_riou(one_hot_pred, one_hot_gt, pred_ind, gt_ind)


    # matching_sc =
    # matching_sc = torch.clamp(riou, min=0)
    # # bz x nmasks
    # max_matching_sc, _ = torch.max(matching_sc, dim=-1)
    max_matching_sc = riou.unsqueeze(0)
    gt_conf = (torch.sum(I_gt, dim=0, keepdim=False) > 0.5).long().unsqueeze(0)
    thr = 0.5
    avg_tot_recall = 0.0
    tot_step = 0
    while thr < 0.96:
        cur_recalled_indicator = (max_matching_sc >= thr).float()
        cur_recalled_nn = torch.sum(cur_recalled_indicator * gt_conf, dim=-1) / torch.clamp(
            torch.sum(gt_conf, dim=-1), min=1e-9)
        avg_cur_recall = torch.mean(cur_recalled_nn).item()
        avg_tot_recall += avg_cur_recall
        # if abs(thr - 0.5) < 0.025 or abs(thr - 0.7) < 0.025 or abs(thr - 0.9) < 0.025:
        #     thr_to_recall[thr] = avg_cur_recall
        tot_step += 1
        thr += 0.05
    cur_avg_recall = avg_tot_recall / float(tot_step)

    k = riou.shape[0]
    mean_riou = riou.sum() / k
    if return_recall:
        return mean_riou, cur_avg_recall
    return mean_riou


def compute_type_miou(type_per_point, T_gt, cluster_pred, I_gt):
    '''
    compute per-primitive-instance type iou
    type_per_point: (1, N, K), K = 4
    T_gt: (1, N)
    '''
    assert (type_per_point.shape[0] == 1)

    # get T_pred: (1, N)
    if len(type_per_point.shape) == 3:
        B, N, _ = type_per_point.shape
        T_pred = torch.argmax(type_per_point, dim=-1)  # (B, N)
    else:
        T_pred = type_per_point

    one_hot_pred = get_one_hot(cluster_pred,
                               cluster_pred.max() + 1)[0]  # (N, K)

    if I_gt.min() == -1:
        # (N, K'), remove background
        one_hot_gt = get_one_hot(I_gt + 1,
                                 I_gt.max() + 2)[0][:, 1:]
    else:
        one_hot_gt = get_one_hot(I_gt, I_gt.max() + 1)[0]

    pred_ind, gt_ind = hungarian_matching(npy(one_hot_pred), npy(one_hot_gt))
    type_iou = torch.Tensor([0.0]).to(T_gt.device)
    cnt = 0

    for p_ind, g_ind in zip(pred_ind, gt_ind):
        gt_type_label = T_gt[I_gt == g_ind].mode()[0]
        pred_type_label = T_pred[cluster_pred == p_ind].mode()[0]
        if gt_type_label == pred_type_label:
            type_iou += 1
        cnt += 1

    type_iou /= cnt
    return type_iou


def compute_type_miou_abc(type_per_point, T_gt, cluster_pred, I_gt):
    '''
    compute per-primitive-instance type iou
    type_per_point: (1, N, K), K = 6/10
    T_gt: (1, N)
    '''
    assert (type_per_point.shape[0] == 1)

    # get T_pred: (1, N)
    if len(type_per_point.shape) == 3:
        B, N, _ = type_per_point.shape
        T_pred = torch.argmax(type_per_point, dim=-1)  # (B, N)
    else:
        T_pred = type_per_point

    T_pred[T_pred == 6] = 0
    T_pred[T_pred == 7] = 0
    T_pred[T_pred == 9] = 0
    T_pred[T_pred == 8] = 2

    T_gt[T_gt == 6] = 0
    T_gt[T_gt == 7] = 0
    T_gt[T_gt == 9] = 0
    T_gt[T_gt == 8] = 2

    one_hot_pred = get_one_hot(cluster_pred,
                               cluster_pred.max() + 1)[0]  # (N, K)

    if I_gt.min() == -1:
        # (N, K'), remove background
        one_hot_gt = get_one_hot(I_gt + 1,
                                 I_gt.max() + 2)[0][:, 1:]
    else:
        one_hot_gt = get_one_hot(I_gt, I_gt.max() + 1)[0]

    pred_ind, gt_ind = hungarian_matching(npy(one_hot_pred), npy(one_hot_gt))
    type_iou = torch.Tensor([0.0]).to(T_gt.device)
    cnt = 0
    riou = compute_riou(one_hot_pred, one_hot_gt, pred_ind, gt_ind)
    for p_ind, g_ind in zip(pred_ind, gt_ind):
        gt_type_label = T_gt[I_gt == g_ind].mode()[0]
        try:
            pred_type_label = T_pred[cluster_pred == p_ind].mode()[0]
        except:
            continue
        if gt_type_label == pred_type_label:
            type_iou += 1

        cnt += 1

    type_iou /= cnt
    return type_iou


def compute_embedding_loss(pred_feat, gt_label, gt_prim_type=None, t_pull=0.5, t_push=1.5):
    '''
    pred_feat: (B, N, K)
    gt_label: (B, N)
    '''
    if gt_prim_type is None:
        gt_prim_type = gt_label
    batch_size, num_pts, feat_dim = pred_feat.shape
    device = pred_feat.device
    pull_loss = torch.Tensor([0.0]).to(device)
    push_loss = torch.Tensor([0.0]).to(device)

    # labels_to_push_losses = {}
    labels_to_pull_losses = {}
    labels_to_pull_losses_nn = {}
    for i in range(batch_size):
        num_class = gt_label[i].max() + 2

        embeddings = []

        idx_to_labels = {}

        for j in range(num_class):
            mask = (gt_label[i] == (j - 1))
            feature = pred_feat[i][mask]
            if len(feature) == 0:
                continue

            for kk in range(mask.size(0)):
                if int(mask[kk].item()) == 1:
                    break
            cur_label = int(gt_prim_type[i,kk].item())

            # if cur_label == -1:
            #     continue
            idx_to_labels[len(embeddings)] = cur_label

            embeddings.append(feature)  # (M, K)

        centers = []

        for feature in embeddings:
            center = torch.mean(feature, dim=0).view(1, -1)
            centers.append(center)

        # intra-embedding loss
        pull_loss_tp = torch.Tensor([0.0]).to(device)
        for i, (feature, center) in enumerate(zip(embeddings, centers)):
            dis = torch.norm(feature - center, 2, dim=1) - t_pull
            dis = F.relu(dis)
            pull_loss_tp += torch.mean(dis)

            cur_pull_loss = float(torch.mean(dis).item())
            cur_label = idx_to_labels[i]
            if cur_label not in labels_to_pull_losses:
                labels_to_pull_losses[cur_label] = [cur_pull_loss]
            else:
                labels_to_pull_losses[cur_label].append(cur_pull_loss)


            # if cur_label not in labelsto

        pull_loss = pull_loss + pull_loss_tp / len(embeddings)

        # inter-embedding loss
        centers = torch.cat(centers, dim=0)  # (num_class, K)

        if centers.shape[0] == 1:
            continue

        dst = torch.norm(centers[:, None, :] - centers[None, :, :], 2, dim=2)

        eye = torch.eye(centers.shape[0]).to(device)
        pair_distance = torch.masked_select(dst, eye == 0)

        pair_distance = t_push - pair_distance
        pair_distance = F.relu(pair_distance)
        push_loss += torch.mean(pair_distance)

    for cur_label in labels_to_pull_losses:
        labels_to_pull_losses_nn[cur_label] = len(labels_to_pull_losses[cur_label])
        labels_to_pull_losses[cur_label] = sum(labels_to_pull_losses[cur_label])

    pull_loss = pull_loss / batch_size
    push_loss = push_loss / batch_size
    loss = pull_loss + push_loss
    return loss, pull_loss, push_loss, labels_to_pull_losses, labels_to_pull_losses_nn

def compute_embedding_loss_no_prim(pred_feat, gt_label, gt_prim_type=None, t_pull=0.5, t_push=1.5):
    '''
    pred_feat: (B, N, K)
    gt_label: (B, N)
    '''
    if gt_prim_type is None:
        gt_prim_type = gt_label
    batch_size, num_pts, feat_dim = pred_feat.shape
    device = pred_feat.device
    pull_loss = torch.Tensor([0.0]).to(device)
    push_loss = torch.Tensor([0.0]).to(device)

    # labels_to_push_losses = {}
    labels_to_pull_losses = {}
    labels_to_pull_losses_nn = {}
    for i in range(batch_size):
        num_class = gt_label[i].max() + 2

        embeddings = []

        idx_to_labels = {}

        for j in range(num_class):
            mask = (gt_label[i] == (j - 1))
            feature = pred_feat[i][mask]
            if len(feature) == 0:
                continue

            for kk in range(mask.size(0)):
                if int(mask[kk].item()) == 1:
                    break
            cur_label = int(gt_prim_type[i,kk].item())

            # if cur_label == -1:
            #     continue
            idx_to_labels[len(embeddings)] = cur_label

            embeddings.append(feature)  # (M, K)

        centers = []

        for feature in embeddings:
            center = torch.mean(feature, dim=0).view(1, -1)
            centers.append(center)

        # intra-embedding loss
        pull_loss_tp = torch.Tensor([0.0]).to(device)
        for i, (feature, center) in enumerate(zip(embeddings, centers)):
            dis = torch.norm(feature - center, 2, dim=1) - t_pull
            dis = F.relu(dis)
            pull_loss_tp += torch.mean(dis)

            cur_pull_loss = float(torch.mean(dis).item())
            cur_label = idx_to_labels[i]
            if cur_label not in labels_to_pull_losses:
                labels_to_pull_losses[cur_label] = [cur_pull_loss]
            else:
                labels_to_pull_losses[cur_label].append(cur_pull_loss)


            # if cur_label not in labelsto

        pull_loss = pull_loss + pull_loss_tp / len(embeddings)

        # inter-embedding loss
        centers = torch.cat(centers, dim=0)  # (num_class, K)

        if centers.shape[0] == 1:
            continue

        dst = torch.norm(centers[:, None, :] - centers[None, :, :], 2, dim=2)

        eye = torch.eye(centers.shape[0]).to(device)
        pair_distance = torch.masked_select(dst, eye == 0)

        pair_distance = t_push - pair_distance
        pair_distance = F.relu(pair_distance)
        push_loss += torch.mean(pair_distance)

    for cur_label in labels_to_pull_losses:
        labels_to_pull_losses_nn[cur_label] = len(labels_to_pull_losses[cur_label])
        labels_to_pull_losses[cur_label] = sum(labels_to_pull_losses[cur_label])

    pull_loss = pull_loss / batch_size
    push_loss = push_loss / batch_size
    loss = pull_loss + push_loss
    return loss, pull_loss, push_loss, labels_to_pull_losses, labels_to_pull_losses_nn


def k_means(p, num_class=2):
    b, N, c = p.shape

    IDX = torch.zeros(b, N).to(p.device).long()

    for i in range(b):
        cur_feat = npy(p[i])
        kmeans = KMeans(n_clusters=num_class, random_state=0).fit(cur_feat)
        IDX[i] = v(kmeans.labels_)

    return IDX


def compute_normal_loss(pred, gt):
    b, N, _ = pred.shape
    normal_loss = torch.acos((pred * gt).sum(-1).clamp(-0.99, 0.99))

    normal_loss = normal_loss.sum() / (b * N)

    return normal_loss


def compute_type_loss(pred, gt):
    '''
    pred: (B, N, K)
    gt: (B, N)
    '''

    type_loss = nn.CrossEntropyLoss()
    valid_class = (gt != -1)  # remove background
    gt = gt[valid_class]

    pred = pred[valid_class]

    loss = type_loss(pred, gt)

    return loss


def compute_nnl_loss(pred, gt):
    '''
    pred: (B, N, K)
    gt: (B, N)
    '''

    type_loss = nn.NLLLoss()
    valid_class = (gt != -1)  # remove background
    gt = gt[valid_class]

    pred = pred[valid_class]

    loss = type_loss(pred, gt)

    return loss


def compute_instance_loss(pred, gt):
    '''
    pred: (B, N, K)
    gt: (B, N)
    '''
    type_loss = nn.CrossEntropyLoss()
    valid_class = (gt != -1)  # remove background

    gt = gt[valid_class]

    pred = pred[valid_class]

    loss = type_loss(pred, gt)

    return loss


def compute_param_loss(pred, T_gt, T_param_gt):
    '''
    only add loss to corresponding type
    pred: (B, N, 22)
    T_gt: (B, N)
    T_param_gt: (B, N, 22)
    '''
    param_list = {5: [0, 4], 1: [4, 8], 4: [8, 15], 3: [15, 22]}

    # [0, 4, 8, 15, 22]

    b, N, _ = pred.shape

    # l2_loss = nn.MSELoss(reduction='sum')
    l2_loss = nn.MSELoss()

    total_loss = 0
    length = 0
    cnt = 0
    for b in range(pred.shape[0]):
        for i in [1, 4, 5, 3]:
            index = T_gt[b] == i
            # prediction for this primitive type
            tmp_pred = pred[b][index]
            # gt-parameter for this type
            tmp_gt = T_param_gt[b][index]

            if tmp_pred.shape[0] == 0:
                continue
            if tmp_gt.sum() == 0:  # no parameters to process
                continue

            tmp_pred = tmp_pred[:, param_list[i][0]:param_list[i][1]]
            tmp_gt = tmp_gt[:, param_list[i][0]:param_list[i][1]].float()

            # get valid points of this primitive type
            valid_mask = tmp_gt.sum(1) != 0

            tmp_pred = tmp_pred[valid_mask]
            tmp_gt = tmp_gt[valid_mask]

            if tmp_gt.shape[0] == 0:
                continue

            tmp_loss = l2_loss(tmp_pred, tmp_gt)

            # ignore wrong type label
            if tmp_gt.max() > 10 or tmp_loss > 50:
                continue

            total_loss += tmp_loss

            length += tmp_pred.shape[0]
            cnt += 1

    # TODO: only happened in test phase
    if cnt == 0:
        if torch.isnan(l2_loss(tmp_pred, tmp_gt.float())).sum() > 0:
            return torch.Tensor([0.0]).to(T_gt.device)
        return l2_loss(tmp_pred, tmp_gt.float())

    total_loss = total_loss / cnt

    return total_loss
