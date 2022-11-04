from Utils.attention_modules import Attention, TransformerEncoderLayer, BertPooler, PositionalEncoding
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as visionmodels
import numpy as np


class VKD(nn.Module):
    def __init__(self,
                 latent_dim,
                 bert_embed_size,
                 n_feature_maps,
                 feature_map_size, 
                 max_num_words,
                 class_weights = None,
                 dropout_rate = 0.5,
                 num_transformers = 1,
                 turn_off_recognition_grad = True,
                 **kwargs):
        super(VKD, self).__init__()
        # acivation functions
        self.leakyRelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

        self.MSEloss = nn.MSELoss()
        self.BCEloss = nn.BCEWithLogitsLoss()
        
        self.token_dim = bert_embed_size
        self.n_feature_maps = n_feature_maps
        self.feature_map_size = feature_map_size
        self.latent_dim = latent_dim
        self.max_num_words = max_num_words
        
        if class_weights.any()!=None:
            self.BCElossde = nn.BCEWithLogitsLoss(pos_weight = torch.Tensor(class_weights))
        else:
            self.BCElossde = nn.BCEWithLogitsLoss()

        self.densenet121 = visionmodels.densenet121(pretrained=True)


        modules = list(self.densenet121.children())[:-1]  # delete the last fc layer.
        self.densenet121 = nn.Sequential(*modules)

        self.textpool = BertPooler(self.token_dim)
        self.pe_t = PositionalEncoding(self.token_dim, max_len = self.max_num_words)
        self.transformer_t = TransformerEncoderLayer(d_model = self.token_dim)
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.densenet121linear = nn.Sequential(
            nn.Linear(self.feature_map_size,48),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate))


        

        # # VAE
        self.fc_mu = nn.Sequential(
            nn.Linear(self.max_num_words, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate))
        self.fc_var = nn.Sequential(
            nn.Linear(self.max_num_words, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate))
        self.fcp_mu = nn.Sequential(
            nn.Linear(self.n_feature_maps, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate))

        self.fcp_var = nn.Sequential(
            nn.Linear(self.n_feature_maps, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate))

        modules = []

        dense_dims = [int(self.latent_dim/2), int(self.latent_dim/4)]
        in_channels = self.latent_dim

        for d_dim in dense_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, d_dim),
                    nn.BatchNorm1d(d_dim),
                    nn.Dropout(dropout_rate),
                    nn.LeakyReLU())
            )
            in_channels = d_dim
        modules.append(
            nn.Sequential(
                nn.Linear(in_channels, 14)
                ))
        self.generation = nn.Sequential(*modules)
        self.latent_dim +=self.n_feature_maps
        modules_p = []


        dense_dims = [int(self.latent_dim/2), int(self.latent_dim/4)]
        in_channels = self.latent_dim
        for d_dim in dense_dims:
            modules_p.append(
                nn.Sequential(
                    nn.Linear(in_channels, d_dim),
                    nn.BatchNorm1d(d_dim),
                    nn.Dropout(dropout_rate),
                    nn.LeakyReLU())
            )
            in_channels = d_dim
        modules_p.append(
            nn.Sequential(
                nn.Linear(in_channels, 14)
                ))
        self.generation_p = nn.Sequential(*modules_p)
        recognition_network_layers = [self.fc_mu, self.fc_var, self.generation_p]
        if turn_off_recognition_grad:
            for layer in recognition_network_layers:
                for param in layer.parameters():
                    param.requires_grad = False


    def forward(self, img_input, text_input):
        
        # img features
        img_features = self.densenet121(img_input)
        img_features = self.densenet121linear(img_features.view(-1,1024,49))

        # text features
        text_features = self.transformer_t(self.pe_t(text_input))

        out_p = torch.squeeze(self.pool(img_features))
        out_r = torch.squeeze(self.pool(text_features))

        mu_p = self.fcp_mu(out_p)
        logvar_p  = self.fcp_var(out_p)
        z_p = self.reparameterize(mu_p, logvar_p)

        mu_r = self.fc_mu(out_r)
        logvar_r = self.fc_var(out_r)
        z_r = self.reparameterize(mu_r, logvar_r)
        
        return self.generation_p(torch.cat([z_p,out_p],1)), mu_p, logvar_p, self.generation(z_r), mu_r, logvar_r
    def testing(self, img_input):
        
        # img features
        img_features = self.densenet121(img_input)
        img_features = self.densenet121linear(img_features.view(-1,1024,49))

        
        out_p = torch.squeeze(self.pool(img_features))

        mu_p = self.fcp_mu(out_p)
        logvar_p  = self.fcp_var(out_p)
        z_p = self.reparameterize(mu_p, logvar_p)
        
        return self.generation_p(torch.cat([z_p,out_p],1)), mu_p, logvar_p
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        z = torch.zeros_like(mu)
        
        std = torch.exp(0.5 * logvar)
        for i in range(100):
            eps = torch.randn_like(std)
            z += eps * std + mu
        return z/100
    def reparameterize_single(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        z = torch.zeros_like(mu)
        
        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def kl_loss_single(self, mu, log_var):
        return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    def kl_loss_multi(self,mu_p, logvar_p, mu_r, logvar_r):
        p = torch.distributions.Normal(mu_p, logvar_p)
        r = torch.distributions.Normal(mu_r, logvar_r)
        return torch.distributions.kl_divergence(p, r).mean()
    def loss_function(self,
                      *args,
                      **kwargs):

        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        epsilon = 1e-8

        recons = args[0]
        label = args[1]
        mu = args[2]+epsilon
        logvar = args[3]+epsilon
        recons_r = args[4]
        mu_r = args[5]+epsilon
        logvar_r = args[6]+epsilon
        annealing_factor = args[7]

        recons_loss = self.BCEloss(recons, label)
        recons_loss_r = self.BCEloss(recons_r, label)


        kld_loss=self.kl_loss_multi(mu, logvar,mu_r, logvar_r)
        kld_weight = 1e-3

        loss = recons_loss*recon_weight+kld_loss*kld_weight*annealing_factor+recons_loss_r*recon_weight
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss, 'Recons_r': recons_loss_r,}
