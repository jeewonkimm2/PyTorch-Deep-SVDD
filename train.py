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

        # t-SNE 시각화

        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        # Extract embeddings and labels
        embeddings = []
        labels = []
        scores = []

        with torch.no_grad():
            for x, y in self.train_loader:                  # 여기서 실제 데이터를 load해야 정확한 label값을 가지고 올 수 있을 것 같은데
                x = x.float().to(self.device)
                z = net(x)
                score = torch.sum((z - c) ** 2, dim=1)

                embeddings.extend(z.cpu().numpy())
                scores.append(score.detach().cpu())
                labels.extend(y.cpu().numpy())

        embeddings = np.array(embeddings)
        labels = np.array(labels)
        scores = np.concatenate(scores) 

        # Print labels, embeddings, and scores
        for label, embedding, score in zip(labels, embeddings, scores):
            print("Label:", label, "Embedding:", embedding, "Score:", score)

        # Apply t-SNE to reduce dimensionality # 마지막 layer에 embedding값을 2차원으로 축소
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_tsne = tsne.fit_transform(embeddings)
        
        # Get center coordinates
        center = np.mean(embeddings_tsne, axis=0)
        print("Center:", center)

        # Calculate distance percentiles
        distances = np.linalg.norm(embeddings_tsne - center, axis=1)
        percentiles = np.percentile(distances, [25, 50, 75])
        c = np.max(distances ) - np.min(distances )
        
        # Plot t-SNE visualization with color-coded labels # 0: label은 normal, 1: label은 anomaly
        plt.scatter(embeddings_tsne[labels == 0, 0], embeddings_tsne[labels == 0, 1], color='blue', label='Normal', s=1, alpha=0.5)
        plt.scatter(embeddings_tsne[labels == 1, 0], embeddings_tsne[labels == 1, 1], color='red', label='Anomaly', s=1, alpha=0.5)
        # scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=scores, cmap='cool', s=1, alpha=0.5)
        
        # Plot center and boundaries based on t-SNE coordinates
        plt.scatter(center[0], center[1], color='green', marker='o', label='Center',s=1, alpha=0.5)
        for percentile in percentiles:
            boundary_radius = c * (percentile / 100)
            boundary_circle = plt.Circle(center, boundary_radius, color='purple', fill=False)
            plt.gca().add_patch(boundary_circle)

        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('t-SNE Visualization of Deep SVDD Embeddings')
        plt.legend()
        plt.show()

        