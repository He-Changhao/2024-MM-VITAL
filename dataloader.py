import numpy as np
import scipy.io as sio
import torch
import sklearn.preprocessing as skp

def load_data(args):
    data_list = []
    mat = sio.loadmat(args.dataset_path + args.dataset_name + '.mat')

    dn = data_normalize(norm_type=args.data_norm)

    # Load the dataset based on the dataset name
    if args.dataset_name == 'CUB':
        data_list.append(dn.norm(mat['X'][0][0]).astype(np.float32))
        data_list.append(dn.norm(mat['X'][0][1]).astype(np.float32))
        label = np.squeeze(mat['gt'].astype(np.uint8))

    elif args.dataset_name == 'Scene-15':
        data_list.append(dn.norm(mat['X'][0][0]).astype(np.float32))
        data_list.append(dn.norm(mat['X'][0][1]).astype(np.float32))
        label = np.squeeze(mat['Y'].astype(np.uint8))

    elif args.dataset_name == 'WIKI':
        data_list.append(dn.norm(mat['Img']).astype(np.float32))
        data_list.append(dn.norm(mat['Txt']).astype(np.float32))
        label = np.squeeze(mat['label'].astype(np.uint8))

    elif args.dataset_name == 'NUS-WIDE':
        data_list.append(dn.norm(mat['Img']).astype(np.float32))
        data_list.append(dn.norm(mat['Txt']).astype(np.float32))
        label = np.squeeze(mat['label'].astype(np.uint8))

    elif args.dataset_name == 'Deep Animal':
        data_list.append(dn.norm(mat['X'][0][5].T).astype(np.float32))
        data_list.append(dn.norm(mat['X'][0][6].T).astype(np.float32))
        label = np.squeeze(mat['gt'].astype(np.uint8))

    elif args.dataset_name == 'Deep Caltech-101':
        data_list.append(dn.norm(mat['X'][0][0].T).astype(np.float32))
        data_list.append(dn.norm(mat['X'][0][1].T).astype(np.float32))
        label = np.squeeze(mat['gt'].astype(np.uint8))

    elif args.dataset_name == 'MNIST-USPS':
        data_list.append(dn.norm(mat['X1']).astype(np.float32))
        data_list.append(dn.norm(mat['X2']).astype(np.float32))
        label = np.squeeze(mat['Y'].astype(np.uint8))

    elif args.dataset_name == 'NoisyMNIST':
        data_list.append(dn.norm(mat['X1']).astype(np.float32))
        data_list.append(dn.norm(mat['X2']).astype(np.float32))
        label = np.squeeze(mat['Y'].astype(np.uint8))

    if data_list[0].shape[0] != label.shape[0]:
        raise ValueError("The dataset dimensions are not (num_samples x features_dims)")

    dims_list = [data.shape[1] for data in data_list]
    num_samples = label.shape[0]
    num_classes = len(np.unique(label))
    num_views = len(data_list)

    split_idx = np.random.permutation(num_samples)
    aligned_num = int(np.ceil(args.aligned_rate * num_samples))
    aligned_idx = split_idx[:aligned_num]
    unaligned_idx = split_idx[aligned_num:]

    # Separate aligned and unaligned data
    aligned_X, aligned_Y = data_list[0][aligned_idx], data_list[1][aligned_idx]
    unaligned_X, unaligned_Y = data_list[0][unaligned_idx], data_list[1][unaligned_idx]
    aligned_labels, unaligned_labels = label[aligned_idx], label[unaligned_idx]

    # Prepare training data and testing data
    train_data = []
    train_data.append(aligned_X)
    train_data.append(aligned_Y)
    train_labels = aligned_labels

    test_data = []
    if args.aligned_rate == 1.0:
        test_data.append(aligned_X)
        test_data.append(aligned_Y)
        test_labels = aligned_labels
        test_labels_Y = aligned_labels
    else:
        shuffle_idx = np.random.permutation(len(unaligned_Y))
        unaligned_Y = unaligned_Y[shuffle_idx]
        test_data.append(np.concatenate((aligned_X, unaligned_X)))
        test_data.append(np.concatenate((aligned_Y, unaligned_Y)))
        test_labels = np.concatenate((aligned_labels, unaligned_labels))
        test_labels_Y = np.concatenate((aligned_labels, unaligned_labels[shuffle_idx]))

    # Create data loaders for training and testing
    train_loader = torch.utils.data.DataLoader(two_view_Dataset(train_data, train_labels), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(two_view_Dataset_test(test_data, test_labels, test_labels_Y), batch_size=args.batch_size, shuffle=True)

    return train_loader, test_loader, num_samples, num_classes, num_views, dims_list


class data_normalize():
    def __init__(self, norm_type):
        super(data_normalize, self).__init__()
        self.norm_type = norm_type
    
    def norm(self, x):
        """Normalize data based on the specified normalization type."""
        if self.norm_type == 'standard':
            return skp.scale(x)
        elif self.norm_type == 'l2-norm':
            return skp.normalize(x)
        elif self.norm_type == 'min-max':
            return skp.minmax_scale(x)
        else:
            raise ValueError("The data_norm name is wrong! Choose one from 'standard', 'l2-norm', 'min-max'.")


class two_view_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        """Return the total number of samples."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Get a sample by index."""
        x0 = torch.from_numpy(self.data[0][idx])
        x1 = torch.from_numpy(self.data[1][idx])
        label = self.labels[idx]
        return x0, x1, label


class two_view_Dataset_test(torch.utils.data.Dataset):
    def __init__(self, data, labels, labels_Y):
        self.data = data
        self.labels = labels
        self.labels_Y = labels_Y

    def __len__(self):
        """Return the total number of samples."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Get a sample by index."""
        x0 = torch.from_numpy(self.data[0][idx])
        x1 = torch.from_numpy(self.data[1][idx])
        label = self.labels[idx]
        label_Y = self.labels_Y[idx]
        return x0, x1, label, label_Y