import torch

from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def eval(net, c, dataloader, device):
    """Testing the Deep SVDD model"""

    scores = []
    labels = []
    encoded_data = []
    true_labels = []

    net.eval()
    print('Testing...')

    with torch.no_grad():
        for x, y in dataloader:
            x = x.float().to(device)
            y = y.to(device) # Label
            z = net(x) # output : z_dim 32

            score = torch.sum((z - c) ** 2, dim=1)

            scores.append(score.detach().cpu())
            labels.append(y.cpu())

            encoded_data.append(z.detach().cpu().numpy())
            true_labels.append(y.detach().cpu().numpy())

    encoded_data = np.concatenate(encoded_data, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    c = c.cpu().numpy()  # Move c tensor from GPU to CPU and convert to numpy array
    c = np.expand_dims(c, axis=0)  # Add an extra dimension to match the dimensions of encoded_data
    encoded_data = np.append(encoded_data, c, axis=0) # Append the value of c to encoded_data

    # t-SNE 적용
    
    tsne = TSNE(n_components=2)
    print("encoded_data", encoded_data)
    encoded_data_2d = tsne.fit_transform(encoded_data)  # 32->2 dimension
    print("encoded_data_2d", encoded_data_2d)

    # Extract the last appended value c
    center_2d = encoded_data_2d[-1]
    print("c", center_2d)
    encoded_data_2d = encoded_data_2d[:-1]  # Remove the last appended value c

    # Calculate distance percentiles
    distances = np.linalg.norm(encoded_data_2d - center_2d, axis=1)
    percentiles = np.percentile(distances, [25, 50, 75])

    # 시각화

    # Plot t-SNE visualization with color-coded labels # 0: label은 normal, 1: label은 anomaly
    plt.scatter(encoded_data_2d[true_labels == 0, 0], encoded_data_2d[true_labels == 0, 1], color='blue', label='Normal', s=1, alpha=0.5)
    plt.scatter(encoded_data_2d[true_labels == 1, 0], encoded_data_2d[true_labels == 1, 1], color='red', label='Anomaly', s=1, alpha=0.5)

    # Plot center and boundaries based on t-SNE coordinates
    plt.scatter(center_2d[0].item(), center_2d[1].item(), color='green', marker='o', label='Center', s=3)
    for percentile in percentiles:
        boundary_radius = (center_2d[0] * (percentile / 100)).item()
        boundary_circle = plt.Circle(center_2d, boundary_radius, color='purple', fill=False)
        plt.gca().add_patch(boundary_circle)

    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Visualization')
    plt.legend()
    plt.show()

    labels, scores = torch.cat(labels).numpy(), torch.cat(scores).numpy()
    print('ROC AUC score: {:.2f}'.format(roc_auc_score(labels, scores)*100))
    return labels, scores