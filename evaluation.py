import torch
import numpy as np
import torch.nn.functional as F
from Clustering import Clustering

def inference(model, test_loader, args):
    """
    Perform inference on the test dataset.

    Args:
        model (torch.nn.Module): The trained model.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        args (argparse.Namespace): Argument parser object containing configuration parameters.

    Returns:
        Accuracy (ACC), Normalized Mutual Information (NMI), and Adjusted Rand Index (ARI) scores.
    """
    model.eval()
    aligned_out0 = []
    aligned_out1 = []
    gt_labels = []
    align_labels = torch.zeros(len(test_loader.dataset))

    with torch.no_grad():
        for batch_idx, (x0, x1, labels, labels_Y) in enumerate(test_loader):
            test_num = len(labels)
            test_view0_sample, test_view1_sample, labels = x0.cuda(), x1.cuda(), labels.cuda()
            mu0, mu1, logvar0, logvar1, _, _ = model(test_view0_sample, test_view1_sample)
            std0 = (0.5 * logvar0).exp()
            std1 = (0.5 * logvar1).exp()

            # Normalize features if specified
            if args.feats_norm:
                h0 = F.normalize(mu0) + F.normalize(std0)
                h1 = F.normalize(mu1) + F.normalize(std1)
                C = euclidean_dist(F.normalize(mu0), F.normalize(mu1))
            else:
                h0 = mu0 + std0
                h1 = mu1 + std1
                C = euclidean_dist(mu0, mu1)

            # Realign process
            for i in range(test_num):
                idx = torch.argsort(C[i, :])
                C[:, idx[0]] = float("inf")
                if args.aligned_rate == 1.0:
                    aligned_out0.append((h0[i, :].cpu()).numpy())
                    aligned_out1.append((h1[i, :].cpu()).numpy())
                    if labels[i] == labels_Y[i]:
                        align_labels[args.batch_size * batch_idx + i] = 1
                else:
                    aligned_out0.append((h0[i, :].cpu()).numpy())
                    aligned_out1.append((h1[idx[0], :].cpu()).numpy())
                    if labels[i] == labels_Y[idx[0]]:
                        align_labels[args.batch_size * batch_idx + i] = 1
            gt_labels.extend(labels.cpu().numpy())

        data = [np.array(aligned_out0), np.array(aligned_out1)]
        gt_labels = np.array(gt_labels)
        _, ret = Clustering(data, gt_labels)
        count = torch.sum(align_labels)
        acc = round(ret['kmeans']['accuracy'] * 100, 4)
        nmi = round(ret['kmeans']['NMI'] * 100, 4)
        ari = round(ret['kmeans']['ARI'] * 100, 4)
        car = round((count.item() / len(test_loader.dataset)) * 100, 2)

    return acc, nmi, ari




def euclidean_dist(x, y):
    """
    From https://github.com/XLearning-SCU/2021-CVPR-MvCLN
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """

    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    dist.addmm_(1, -2, x, y.t())
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist