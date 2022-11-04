import torch
from tqdm import tqdm
import copy
import os
import numpy as np
import time
import random
import torch.nn as nn
import torch.nn.functional as F
import umap
import os
import numpy as np
import random
from matplotlib import pyplot as plt
plt.switch_backend('agg')

import scipy.misc
import seaborn as sns
from itertools import cycle
from sklearn.manifold import TSNE
import csv
import pandas as pd
import cv2
from Evaluation.evaluate_classification import get_classification_metrics

def seed_everything(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L 
def pytorch_model_run(train_loader, valid_loader, model_obj, model_name, clip=True, n_epochs = 10, batch_size = 24):
    seed_everything()
    model = copy.deepcopy(model_obj)
    model.cuda()

    optimizer =  torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 3)
    kl_ann_factor = frange_cycle_linear(len(train_loader))
    ################################################################################################
    ###############################################################################################
    best_valid_loss = float('inf')
    counter = 0
    torch.autograd.set_detect_anomaly(True)
    # torch.autograd.detect_anomaly(True)
    for epoch in range(n_epochs):
        with tqdm(total=batch_size * len(train_loader)) as epoch_pbar:
            epoch_pbar.set_description(f'Epoch {epoch}')
            start_time = time.time()
            model.train()
            acc_loss = 0.
            rloss = 0.
            rlossr = 0.
            klloss = 0.
            for i, (x1_batch, x2_batch, y_batch) in enumerate(train_loader):
                torch.cuda.empty_cache()
                x1_batch = x1_batch.type(torch.float32).cuda()
                x2_batch = x2_batch.type(torch.float32).cuda()
                y_batch = y_batch.type(torch.float32).cuda()

                y_pred, mu, logvar, y_predr, mur, logvarr = model(x2_batch, x1_batch)

                loss_dict = model.loss_function(y_pred, label, mu, logvar, y_predr, mur, logvarr, kl_ann_factor[i])
                loss = loss_dict['loss']
                optimizer.zero_grad()
                loss.backward()

                if clip:
                    nn.utils.clip_grad_norm_(model.parameters(), 0.01)
                acc_loss += loss.item()
                #
                rloss += loss_dict['Reconstruction_Loss'].item()
                rlossr += loss_dict['Recons_r'].item()
                klloss += loss_dict['KLD'].item()
                optimizer.step()
                torch.cuda.empty_cache()
                avg_loss = acc_loss / (i + 1)
                recon_loss = rloss / (i + 1)
                recon_loss_r = rlossr /(i + 1)
                kldloss = klloss / (i + 1)
                desc = f'Epoch {epoch} - loss {avg_loss:.2f} -recon {recon_loss:.4f} -kld {kldloss:.1f} - reconr {recon_loss_r:.4f}'
                epoch_pbar.set_description(desc)
                epoch_pbar.update(x1_batch.shape[0])

        model.eval()

        acc_loss = 0.
        rloss = 0.
        rlossr = 0
        klloss = 0.
        with tqdm(total=batch_size * len(valid_loader)) as epoch_pbar:
            epoch_pbar.set_description(f'VAL Epoch {epoch}')
            for i, (x1_batch, x2_batch, y_batch) in enumerate(valid_loader):
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    x1_batch = x1_batch.type(torch.float32).cuda()
                    x2_batch = x2_batch.type(torch.float32).cuda()
                    y_batch = y_batch.type(torch.float32).cuda()

          
                    y_pred, mu, logvar, y_predr, mur, logvarr = model(x2_batch, x1_batch)#

                    loss_dict = model.loss_function(y_pred, label, mu, logvar,y_predr,  mur, logvarr, kl_ann_factor[2])
                    val_loss = loss_dict['loss']
                    acc_loss += val_loss.item()
                    rloss += loss_dict['Reconstruction_Loss'].item()
                    rlossr += loss_dict['Recons_r'].item()
                    klloss += loss_dict['KLD'].item()
                    torch.cuda.empty_cache()
                    torch.cuda.empty_cache()
                    avg_val_loss = acc_loss / (i + 1)
                    recon_loss = rloss / (i + 1)
                    recon_loss_r = rlossr /(i + 1)
                    kldloss = klloss / (i + 1)
                desc = f'VAL Epoch {epoch} - loss {avg_loss:.2f} -recon {recon_loss:.4f} -kld {kldloss:.1f} - reconr {recon_loss_r:.2f}'
                epoch_pbar.set_description(desc)
                epoch_pbar.update(x1_batch.shape[0])

        if avg_val_loss < best_valid_loss:
            best_valid_loss = avg_val_loss

            torch.save(model.state_dict(), model_name)

        scheduler.step(avg_val_loss)
        elapsed_time = time.time() - start_time
        print('VAL epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(epoch + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time))
        if avg_val_loss > avg_loss:
            counter += 1
        if counter == 5:
            break
    return model
def accuracy(y_pred, y_true):
    t =torch.autograd.Variable(torch.FloatTensor([0.5]))  # threshold
    out = (y_pred >= t.cuda()).float() * 1
    equals = y_true.float()  ==  out.t()
    return equals

def predict_classification(model_obj, test_loader, device, batch_size = 64, embed_size = 512, reverse_word_map = 1):

    classes = np.zeros((test_len,14),dtype = 'float64')
    classes_ehr =  np.zeros((test_len,14),dtype = 'float64')
    y_true = torch.tensor([], dtype=torch.float).cuda()
    # set model to evaluate model
    model = copy.deepcopy(model_obj)
    model.cuda()
    model.eval()


    RS = 20150101
    counter = 0
    with torch.no_grad():
        for i, (x1_batch, x2_batch, label) in tqdm(enumerate(test_loader)):
            if counter<test_len//batch_size:
                x1_batch = x1_batch.type(torch.float32).cuda()
                x2_batch = x2_batch.type(torch.float32).cuda()
                label = label.type(torch.float32).cuda()

                y_pred, mu, logvar = model.testing(x2_batch)
                classes[counter*batch_size:(counter+1)*batch_size] = F.sigmoid(y_pred).detach().cpu().numpy()
                classes_ehr[counter*batch_size:(counter+1)*batch_size] = F.sigmoid(y_pred_ehr).detach().cpu().numpy()
                y_true = torch.cat((y_true, label), 0)
            else:
                break
            counter+=1

    y_true = y_true.cpu().numpy()
    array = np.array([classes, y_true])
    get_classification_metrics(array)
