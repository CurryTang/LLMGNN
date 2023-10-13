import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

def plot_tsne_with_centers(feature_matrix, label_tensor, center_tensor):
    # Compute T-SNE embeddings for the feature matrix
    tsne = TSNE(n_components=2, random_state=42)
    aug_feature_matrix = np.concatenate((feature_matrix, center_tensor), axis=0)
    aug_embedded_features = tsne.fit_transform(aug_feature_matrix)
    embedded_features = aug_embedded_features[:feature_matrix.shape[0]]
    # center_tensor = aug_embedded_features[feature_matrix.shape[0]:]
    # import ipdb; ipdb.set_trace()
    # Plot nodes based on T-SNE embeddings and color by class labels
    plt.scatter(embedded_features[:, 0], embedded_features[:, 1], c=label_tensor, cmap=plt.cm.get_cmap("jet", np.unique(label_tensor).size), marker='o')
    
    # Compute T-SNE embeddings for the center tensor

    # embedded_centers = tsne.fit_transform(center_tensor)
    embedded_centers = aug_embedded_features[feature_matrix.shape[0]:]
    
    # Plot clustering centers with a different shape
    for idx, center in enumerate(embedded_centers):
        plt.scatter(center[0], center[1], color=plt.cm.jet(idx / np.unique(label_tensor).size), marker='x', s=100, edgecolors='k')
    
    # Display the plot
    plt.colorbar(ticks=range(np.unique(label_tensor).size))
    plt.clim(-0.5, np.unique(label_tensor).size - 0.5)
    plt.show()
    plt.savefig("tsneeee.png")




if __name__ == '__main__':
    data = torch.load("../..//ogb/preprocessed_data/new/cora_fixed_sbert.pt")
    feature_matrix = data.x.numpy()
    label_tensor = data.y.numpy()
    center = torch.load("../../ogb/preprocessed_data/aax/center_x_2708_7.pt")
    center = center.numpy()
    plot_tsne_with_centers(feature_matrix, label_tensor, center)