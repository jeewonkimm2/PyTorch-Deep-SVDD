import torch
from torch import optim
import torch.nn.functional as F

import numpy as np
from barbar import Bar

from model import autoencoder, network
from utils.utils import weights_init_normal


class TrainerDeepSVDD:
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device
    

    def pretrain(self):
        """ Pretraining the weights for the deep SVDD network using autoencoder"""
        ae = autoencoder(self.args.latent_dim).to(self.device)
        ae.apply(weights_init_normal)
        optimizer = optim.Adam(ae.parameters(), lr=self.args.lr_ae,
                               weight_decay=self.args.weight_decay_ae)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=self.args.lr_milestones, gamma=0.1)
        
        ae.train()
        for epoch in range(self.args.num_epochs_ae):
            total_loss = 0
            for x, _ in Bar(self.train_loader):
                x = x.float().to(self.device)
                
                optimizer.zero_grad()
                x_hat = ae(x)
                reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                optimizer.step()
                
                total_loss += reconst_loss.item()
            scheduler.step()
            print('Pretraining Autoencoder... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.train_loader)))
        self.save_weights_for_DeepSVDD(ae, self.train_loader) 
    

    def save_weights_for_DeepSVDD(self, model, dataloader):
        """Initialize Deep SVDD weights using the encoder weights of the pretrained autoencoder."""
        c = self.set_c(model, dataloader)
        net = network(self.args.latent_dim).to(self.device)
        state_dict = model.state_dict()
        net.load_state_dict(state_dict, strict=False)
        torch.save({'center': c.cpu().data.numpy().tolist(),
                    'net_dict': net.state_dict()}, 'weights/pretrained_parameters.pth')
    

    def set_c(self, model, dataloader, eps=0.1):
        """Initializing the center for the hypersphere"""
        model.eval()
        z_ = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.float().to(self.device)
                z = model.encode(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c


    def train(self):
        """Training the Deep SVDD model"""
        net = network().to(self.device)
        
        if self.args.pretrain==True:
            state_dict = torch.load('weights/pretrained_parameters.pth')
            net.load_state_dict(state_dict['net_dict'])
            c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            net.apply(weights_init_normal)
            c = torch.randn(self.args.latent_dim).to(self.device)
        
        optimizer = optim.Adam(net.parameters(), lr=self.args.lr,
                               weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=self.args.lr_milestones, gamma=0.1)

        net.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _ in Bar(self.train_loader):
                x = x.float().to(self.device)

                optimizer.zero_grad()
                z = net(x)
                loss = torch.mean(torch.sum((z - c) ** 2, dim=1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            scheduler.step()
            print('Training Deep SVDD... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.train_loader)))
            
        self.net = net
        self.c = c
          
        # 마지막 레이어 t-SNE 시각화

        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        # 마지막 레이어 데이터 정상/비정상
        scores = []
        labels = []
        
        with torch.no_grad():
            for x, y in self.train_loader:
                x = x.float().to(self.device)
                z = net(x)
                score = torch.sum((z - c) ** 2, dim=32)

                scores.append(score.detach().cpu())
                labels.append(y.cpu())
        labels, scores = torch.cat(labels).numpy(), torch.cat(scores).numpy()


        # 마지막 레이어 임베딩
        z_values = []
            
        with torch.no_grad():
            for x, _ in self.train_loader:
                x = x.float().to(self.device)
                x = self.net.conv1(x)
                x = self.net.pool(F.leaky_relu(self.net.bn1(x)))
                x = self.net.conv2(x)
                x = self.net.pool(F.leaky_relu(self.net.bn2(x)))
                x = x.view(x.size(0), -1)
                z = self.net.fc1(x)
                z_values.append(z.detach().cpu().numpy())

        z_values = np.concatenate(z_values, axis=0)
               
        # t-SNE 적용
        tsne = TSNE(n_components=2, random_state=42)
        z_tsne = tsne.fit_transform(z_values)

        # 시각화
        plt.scatter(z_tsne[:, 0], z_tsne[:, 1], color='blue', label='Normal',s=1, alpha=0.5)
        plt.scatter(z_tsne[:, 0], z_tsne[:, 1], color='red', label='Anomaly',s=1, alpha=0.5)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('t-SNE Visualization')
        plt.show()
        
