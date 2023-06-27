# test.py 다중분류
import torch

from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

num_classes = 15

def softmax(scores):
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)  # Reshape to (1, num_classes)
    exp_scores = np.exp(scores)
    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probabilities

def eval(net, c, dataloader, device):
    """Testing the Deep SVDD model"""

    scores = []
    encoded_data = []
    labels = []
    correct_predictions = 0
    total_predictions = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    net.eval()
    print('Testing...')

    with torch.no_grad():
        for x, y in dataloader:
            x = x.float().to(device)
            y = y.to(device) # Label
            z = net(x) # output : z_dim 32

            score = torch.sum((z - c) ** 2, dim=1)

            scores.append(score.detach().cpu())

            encoded_data.append(z.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

            _, predicted_labels = torch.max(score, dim=0)

            correct_predictions += (predicted_labels == y).sum().item()
            total_predictions += y.size(0)

            for i in range(len(y)):
                label = y[i].item()
                class_correct[label] += int(predicted_labels.item() == label)
                class_total[label] += 1

    encoded_data = np.concatenate(encoded_data, axis=0)
    labels = np.concatenate(labels, axis=0)
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

    print("labels",labels)

    labels = labels.astype(np.int64)

    # Calculate distance percentiles
    distances = np.linalg.norm(encoded_data_2d - center_2d, axis=1)
    # percentiles = np.percentile(distances, [25, 50, 75])
    
    # 시각화
    plt.figure(figsize=(10, 8))
    class_names = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
        'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
        'transistor', 'wood', 'zipper']
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink',
              'brown', 'black', 'cyan', 'grey', 'lime', 'navy', 'salmon', 'turquoise']
    
    indices = []
    class_labels = []
    for i in range(len(class_names)):
        class_label = i
        class_indices = np.where(labels == class_label)[0]
        indices.append(class_indices)
        class_labels.append(class_label)
        plt.scatter(encoded_data_2d[class_indices, 0], encoded_data_2d[class_indices, 1],
                    color=colors[i], label=class_names[i], s=1, alpha=0.5)
        
    # # Plot center and boundaries based on t-SNE coordinates
    # plt.scatter(center_2d[0].item(), center_2d[1].item(), color='green', marker='o', label='Center', s=3)
    # for percentile in percentiles:
    #     boundary_radius = (center_2d[0] * (percentile / 100)).item()
    #     boundary_circle = plt.Circle(center_2d, boundary_radius, color='purple', fill=False)
    #     plt.gca().add_patch(boundary_circle)

    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Visualization')

    # plt.legend()
    handles, c_labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(c_labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.show()

    print('class_indices:',indices)
    print('class_label:',class_labels)
    print('labels:',labels)

    
    scores = torch.cat(scores, dim=0)
    scores = scores.cpu().numpy()  # Move scores tensor from GPU to CPU and convert to numpy arrayscores = torch.cat(scores, dim=0)
    # Apply softmax to convert scores to probabilities
    probabilities = softmax(scores)

    print('Probabilities Shape:', probabilities.shape)
    print('Probabilities:', probabilities)

    predicted_labels = np.argmax(probabilities, axis=1)

    print('Labels Size:', labels.shape)
    print('Predicted Labels Size:', predicted_labels.shape)

    #accuracy
    accuracy = correct_predictions / total_predictions * 100
    print('Accuracy: {:.2f}%'.format(accuracy))

    class_accuracy = [class_correct[i] / class_total[i] * 100 for i in range(num_classes)]
    print('Class-wise Accuracy:', class_accuracy)

    # Reshape scores array if necessary
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)  # Reshape to (num_samples, 1)

    # Convert scores to probabilities using softmax
    probabilities = softmax(scores)
    
    from sklearn.preprocessing import LabelBinarizer

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    
    # ROC AUC 점수 계산
    auc_score1 = roc_auc_score(labels, probabilities, multi_class='ovr', average='macro') * 100
    auc_score2 = roc_auc_score(labels, probabilities, multi_class='ovo', average='macro') * 100
    print('ROC AUC score (ovr): {:.2f}'.format(auc_score1))
    print('ROC AUC score (ovo): {:.2f}'.format(auc_score2))
    
    return labels, scores, accuracy, class_accuracy

# # test.py 이진분류
# import torch

# from sklearn.metrics import roc_auc_score
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import numpy as np

# def eval(net, c, dataloader, device):
#     """Testing the Deep SVDD model"""

#     scores = []
#     labels = []
#     encoded_data = []
#     true_labels = []
#     correct_predictions = 0
#     total_predictions = 0

#     net.eval()
#     print('Testing...')

#     with torch.no_grad():
#         for x, y in dataloader:
#             x = x.float().to(device)
#             y = y.to(device) # Label
#             z = net(x) # output : z_dim 32

#             score = torch.sum((z - c) ** 2, dim=1)

#             scores.append(score.detach().cpu())
#             labels.append(y.cpu())

#             encoded_data.append(z.detach().cpu().numpy())
#             true_labels.append(y.detach().cpu().numpy())

#             predicted_labels = torch.round(score).long()
#             correct_predictions += (predicted_labels == y).sum().item()
#             total_predictions += y.size(0)

#     encoded_data = np.concatenate(encoded_data, axis=0)
#     true_labels = np.concatenate(true_labels, axis=0)
#     c = c.cpu().numpy()  # Move c tensor from GPU to CPU and convert to numpy array
#     c = np.expand_dims(c, axis=0)  # Add an extra dimension to match the dimensions of encoded_data
#     encoded_data = np.append(encoded_data, c, axis=0) # Append the value of c to encoded_data

#     # t-SNE 적용
    
#     tsne = TSNE(n_components=2)
#     print("encoded_data", encoded_data)
#     encoded_data_2d = tsne.fit_transform(encoded_data)  # 32->2 dimension
#     print("encoded_data_2d", encoded_data_2d)

#     # Extract the last appended value c
#     center_2d = encoded_data_2d[-1]
#     print("c", center_2d)
#     encoded_data_2d = encoded_data_2d[:-1]  # Remove the last appended value c
    
#     print("labels",labels)
#     print("true_labels", true_labels)

#     # Calculate distance percentiles
#     distances = np.linalg.norm(encoded_data_2d - center_2d, axis=1)
#     percentiles = np.percentile(distances, [25, 50, 75])

#     # 시각화

#     # Plot t-SNE visualization with color-coded labels # 0: label은 normal, 1: label은 anomaly
#     plt.scatter(encoded_data_2d[true_labels == 0, 0], encoded_data_2d[true_labels == 0, 1], color='blue', label='Normal', s=1, alpha=0.5)
#     plt.scatter(encoded_data_2d[true_labels == 1, 0], encoded_data_2d[true_labels == 1, 1], color='red', label='Anomaly', s=1, alpha=0.5)

#     # Plot center and boundaries based on t-SNE coordinates
#     plt.scatter(center_2d[0].item(), center_2d[1].item(), color='green', marker='o', label='Center', s=3)
#     for percentile in percentiles:
#         boundary_radius = (center_2d[0] * (percentile / 100)).item()
#         boundary_circle = plt.Circle(center_2d, boundary_radius, color='purple', fill=False)
#         plt.gca().add_patch(boundary_circle)

#     plt.xlabel('t-SNE Dimension 1')
#     plt.ylabel('t-SNE Dimension 2')
#     plt.title('t-SNE Visualization')
#     plt.legend()
#     plt.show()

#     accuracy = correct_predictions / total_predictions * 100
#     labels, scores = torch.cat(labels).numpy(), torch.cat(scores).numpy()
#     print('ROC AUC score: {:.2f}'.format(roc_auc_score(labels, scores)*100))
#     print('Accuracy: {:.2f}%'.format(accuracy))

#     return labels, scores