import torch
import torchvision

from torch.utils.data import Dataset

from Utils.load_data import get_multimodal_data, prepare_embeddings, getTokenEmbed, getTargetWeights
from data_preprocessing.BERTtokenizer import BiobertEmbedding

def get_data_loaders(x1_train, x2_train, y1_train, x1_val, x2_val, y1_val, x1_test, x2_test, y1_test, batch_size):

    train_dataset = MultimodalDataset(x1_train, x2_train, y1_train)
    train_loader =  torch.utils.data.dataloader.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, drop_last = True, num_workers = 0, pin_memory = True,                                            
                                            sampler = train_dataset.sampler)
    val_dataset = MultimodalDataset(x1_val, x2_val, y1_val)
    val_loader = torch.utils.data.dataloader.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size, drop_last = True, num_workers = 0, pin_memory= True)#, sampler = val_dataset.sampler)
    test_dataset = MultimodalDataset(x1_test, x2_test, y1_test)
    test_loader = torch.utils.data.dataloader.DataLoader(dataset= test_dataset,
                                              batch_size=batch_size, drop_last = True, num_workers = 0, pin_memory = True)

    return train_loader, val_loader, test_loader



class MultimodalDataset(Dataset):
    def __init__(self, T, X, y, proxy = 0):
        self.T = T
        self.X = X
        self.y = y
        self.proxy = proxy
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.class_weights = self.get_class_weights()
        self.sample_weights = torch.DoubleTensor(self.get_sample_weights())
        self.bert = BiobertEmbedding()
        self.sampler = torch.utils.data.sampler.WeightedRandomSampler(self.sample_weights, len(self.sample_weights))
    def get_sample_weights(self):
        sample_weights = np.zeros(self.y.shape[0])
        for i in range(self.y.shape[0]):
            sample_weights[i] = np.sum(self.y[i,:]*self.class_weights)/np.sum(self.y[i,:])
        sample_weights[0]*=1
        return sample_weights
    def get_class_weights(self):
        weights = np.zeros(self.y.shape[1])
        for c in range(self.y.shape[1]):
            weights[c] = np.sum(self.y[:,c])
            print(weights[c])
        weights = weights/self.y.shape[0]
        for c in range(self.y.shape[1]):
            if weights[c]!=0.0:
                weights[c] = 1/weights[c]
                print(weights[c])
        return weights

    def __len__(self):
        return self.T.shape[0]

    def __getitem__(self, idx):
        img = torchvision.transforms.functional.to_tensor(cv2.imread(self.X[idx]))
        text = torchvision.transforms.functional.to_tensor(cv2.imread(self.T[idx]))
        return text[0,:,:], img, self.y[idx]