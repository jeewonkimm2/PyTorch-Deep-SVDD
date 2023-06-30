import torch
from torch import optim
import torch.nn.functional as F

import numpy as np
from barbar import Bar

from model import autoencoder, network
from utils.utils import weights_init_normal

from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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
                # z = model.encode(x)
                z = model.resnet(x)  # model.resnet을 사용하여 인코딩
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
            total_loss = 0  # 각 미니배치에서의 총 손실
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
            
        # 전체 epoch 종료 후 모든 데이터 포인트와 중심 c 간의 거리 계산. 유클리드 거리
        with torch.no_grad():
            distances = torch.norm(net(x) - c, dim=1)
            min_distance = torch.min(distances).item()
            avg_distance = torch.mean(distances).item()
            max_distance = torch.max(distances).item()

        print('distances:{:.3f}',distances)
        print('Min Distance_train: {:.3f}'.format(min_distance))
        print('Avg Distance_train: {:.3f}'.format(avg_distance))
        print('Max Distance_train: {:.3f}'.format(max_distance))

        self.net = net
        self.c = c

        # net.train()
        # for epoch in range(self.args.num_epochs):
        #     total_loss = 0
        #     for x, _ in Bar(self.train_loader):
        #         x = x.float().to(self.device)

        #         optimizer.zero_grad()
        #         z = net(x)
        #         loss = torch.mean(torch.sum((z - c) ** 2, dim=1))
        #         loss.backward()
        #         optimizer.step()

        #         total_loss += loss.item()

                
        #     scheduler.step()
        #     print('Training Deep SVDD... Epoch: {}, Loss: {:.3f}'.format(
        #            epoch, total_loss/len(self.train_loader))) 
        # self.net = net
        # self.c = c       

         # visualization
        self.visualize_embeddings()

    # visualization
    def visualize_embeddings(self):
        self.net.eval()
        embeddings = []
        targets = []
        with torch.no_grad():
            for x, y in self.train_loader:
                x = x.float().to(self.device)
                z = self.net(x)
                embeddings.append(z.cpu().numpy())
                targets.append(y.numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        targets = np.concatenate(targets, axis=0)

        # Apply t-SNE to reduce embeddings to 2D
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_tsne = tsne.fit_transform(embeddings)

        # Plot the embeddings with reduced marker size
        plt.figure(figsize=(8, 8))
        plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=targets, cmap='viridis', s=2)
        plt.colorbar(label='Class')

        # Set the axes limits
        plt.xlim(embeddings_tsne[:, 0].min() - 1, embeddings_tsne[:, 0].max() + 1)
        plt.ylim(embeddings_tsne[:, 1].min() - 1, embeddings_tsne[:, 1].max() + 1)

        # Plot the center point
        c = self.c.cpu().numpy()
        plt.scatter(c[0], c[1], c='red', marker='x', label='Center')

        # # Calculate pairwise distances and average distance to c  마지막 epoch의 distance
        # distances = np.linalg.norm(embeddings - c, axis=1)
        # avg_distance = np.mean(distances)
        # min_distance = np.min(distances)
        # max_distance = np.max(distances)

        # # Draw circle with max distance as radius
        # circle = Circle((c[0], c[1]), max_distance, color='red', fill=True, linestyle='dashed', label='Max Distance')
        # plt.gca().add_patch(circle)

        # # Draw circle with avg distance as radius
        # circle2 = Circle((c[0], c[1]), avg_distance, color='orange', fill=True, linestyle='dashed', label='Avg Distance')
        # plt.gca().add_patch(circle2)

        plt.title('Embeddings Visualization (t-SNE)')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')

        # # Display distance information
        # distance_info = f"Average distance: {avg_distance:.3f}\nMin distance: {min_distance:.3f}\nMax distance: {max_distance:.3f}"
        # plt.text(0.05, 0.95, distance_info, transform=plt.gca().transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        plt.legend()
        plt.show()

        # print('Average distance to c:', avg_distance)
        # print('Minimum distance to c:', min_distance)
        # print('Maximum distance to c:', max_distance)
