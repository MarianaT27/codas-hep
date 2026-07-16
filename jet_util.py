import os
import numpy as np
import pandas as pd
import h5py
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import roc_curve, auc

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qcd-tt-jet-tagging-co-da-s-he')

def preprocess_jet_images(jet_images, target_size=(32, 32)):
    """
    Preprocess jet images for CNN
    """
    processed_images = {}
    for key, images in jet_images.items():
        # Resize if needed
        if images.shape[1:] != target_size:
            # Add resizing logic here if needed
            pass
        # Normalize
        processed_images[key] = images / np.max(images)
    return processed_images

def load_processed_data():
    """
    Load processed data with unique IDs.
    
    Returns:
        tuple: (X_train, y_train, train_ids, X_val, y_val, val_ids, X_test, y_test, test_ids)
    """
    # Load training data
    X_train = pd.read_csv(os.path.join(DATA_DIR, 'train/features/cluster_features.csv'))
    y_train = np.load(os.path.join(DATA_DIR, 'train/labels/labels.npy'))
    train_ids = np.load(os.path.join(DATA_DIR, 'train/ids/ids.npy'))

    # Load validation data
    X_val = pd.read_csv(os.path.join(DATA_DIR, 'val/features/cluster_features.csv'))
    y_val = np.load(os.path.join(DATA_DIR, 'val/labels/labels.npy'))
    val_ids = np.load(os.path.join(DATA_DIR, 'val/ids/ids.npy'))
    # Load test data
    X_test = pd.read_csv(os.path.join(DATA_DIR, 'test/features/cluster_features.csv'))
    test_ids = np.load(os.path.join(DATA_DIR, 'test/ids/ids.npy'))
    return X_train, y_train, train_ids, X_val, y_val, val_ids, X_test, test_ids

def load_images():
    """
    Load jet images and labels with unique IDs.
    
    Returns:
        tuple: (X_train_images, y_train, train_ids, X_val_images, y_val, val_ids, X_test_images, y_test, test_ids)
    """
    # Load training data
    y_train = np.load(os.path.join(DATA_DIR, 'train/labels/labels.npy'))
    train_ids = np.load(os.path.join(DATA_DIR, 'train/ids/ids.npy'))
    with h5py.File(os.path.join(DATA_DIR, 'train/images/jet_images.h5'), 'r') as f:
            X_train_images = np.expand_dims(f['images'][:], axis=-1)
    
    # Load validation data
    y_val = np.load(os.path.join(DATA_DIR, 'val/labels/labels.npy'))
    val_ids = np.load(os.path.join(DATA_DIR, 'val/ids/ids.npy'))
    with h5py.File(os.path.join(DATA_DIR, 'val/images/jet_images.h5'), 'r') as f:
            X_val_images = np.expand_dims(f['images'][:], axis=-1)
    
    # Load test data
    test_ids = np.load(os.path.join(DATA_DIR, 'test/ids/ids.npy'))
    with h5py.File(os.path.join(DATA_DIR, 'test/images/jet_images.h5'), 'r') as f:
            X_test_images = np.expand_dims(f['images'][:], axis=-1)
    
    
    return X_train_images, y_train, train_ids, X_val_images, y_val, val_ids, X_test_images, test_ids

def compute_cluster_features_from_images(images):
    """
    Run anti_kt_clustering + extract_cluster_features over a batch of jet
    images and collect the per-jet feature dicts into a DataFrame -- this is
    the image -> numeric-features step, one row per jet.
    """
    rows = [extract_cluster_features(anti_kt_clustering(image)) for image in images]
    return pd.DataFrame(rows)

def load_processed_data_extended(use_cache=True):
    """
    Like load_processed_data(), but recomputes cluster features from the raw
    jet images so the extra shape features in extract_cluster_features()
    (std_cluster_eta, std_cluster_phi, eta_phi_aspect_ratio, delta_r_leading,
    leading_pt_fraction) are included alongside the original ones. Results
    are cached to cluster_features_extended.csv per split so repeat calls
    don't re-run clustering.

    Returns:
        tuple: (X_train, y_train, train_ids, X_val, y_val, val_ids, X_test, test_ids)
    """
    X_train_images, y_train, train_ids, X_val_images, y_val, val_ids, X_test_images, test_ids = load_images()

    splits = {
        'train': X_train_images,
        'val': X_val_images,
        'test': X_test_images,
    }
    features = {}
    for split, images in splits.items():
        cache_path = os.path.join(DATA_DIR, split, 'features', 'cluster_features_extended.csv')
        if use_cache and os.path.exists(cache_path):
            features[split] = pd.read_csv(cache_path)
        else:
            features[split] = compute_cluster_features_from_images(images)
            features[split].to_csv(cache_path, index=False)

    return (features['train'], y_train, train_ids,
            features['val'], y_val, val_ids,
            features['test'], test_ids)

def anti_kt_clustering(image, R=0.4, pt_min=0.1):
    """
    Perform anti-kt clustering on a jet image.
    
    Args:
        image (numpy.ndarray): 2D or 3D array representing the jet image (if 3D, first channel is used)
        R (float): Jet radius parameter
        pt_min (float): Minimum pT threshold for particles
        
    Returns:
        list: List of clusters with their properties
    """
    # Handle 3D images (with channel dimension)
    if len(image.shape) == 3:
        image = image[..., 0]  # Take first channel
    
    # Get non-zero pixels (particles)
    y, x = np.where(image > pt_min)
    pts = image[y, x]
    
    if len(pts) == 0:
        return []
    
    # Convert pixel coordinates to eta-phi space
    # Assuming the image is centered at (15, 15) with 0.1 units per pixel
    eta = (y - 15) * 0.1
    phi = (x - 15) * 0.1
    
    # Create particle list with coordinates and pT
    particles = np.column_stack((eta, phi, pts))
    
    # Calculate distance matrix in eta-phi space
    coords = particles[:, :2]
    dist_matrix = squareform(pdist(coords))
    
    # Anti-kt distance measure
    pt_matrix = np.outer(1/pts, 1/pts)
    anti_kt_dist = dist_matrix**2 * pt_matrix
    
    # Clustering
    n_particles = len(particles)
    clusters = []
    used = np.zeros(n_particles, dtype=bool)
    
    while not all(used):
        # Find minimum distance
        valid_dist = anti_kt_dist.copy()
        valid_dist[used] = np.inf
        valid_dist[:, used] = np.inf
        min_dist = np.min(valid_dist)
        
        if min_dist > R**2:
            # Start new cluster
            idx = np.where(~used)[0][0]
            clusters.append([particles[idx]])
            used[idx] = True
        else:
            # Merge clusters
            i, j = np.where(valid_dist == min_dist)
            i, j = i[0], j[0]
            
            # Find clusters containing i and j
            cluster_i = next((c for c in clusters if any(p[2] == particles[i][2] for p in c)), None)
            cluster_j = next((c for c in clusters if any(p[2] == particles[j][2] for p in c)), None)
            
            if cluster_i is None and cluster_j is None:
                # Create new cluster
                clusters.append([particles[i], particles[j]])
            elif cluster_i is None:
                cluster_j.append(particles[i])
            elif cluster_j is None:
                cluster_i.append(particles[j])
            else:
                # Merge clusters
                cluster_i.extend(cluster_j)
                clusters.remove(cluster_j)
            
            used[i] = True
            used[j] = True
    
    return clusters

def extract_cluster_features(clusters):
    """
    Extract features from clusters.
    
    Args:
        clusters (list): List of clusters from anti-kt clustering
        
    Returns:
        dict: Dictionary of cluster features
    """
    features = {
        'n_clusters': len(clusters),
        'max_cluster_pt': 0.0,
        'mean_cluster_pt': 0.0,
        'std_cluster_pt': 0.0,
        'max_cluster_size': 0,
        'mean_cluster_size': 0.0,
        'std_cluster_size': 0.0,
        'total_pt': 0.0,
        'max_cluster_eta': 0.0,
        'max_cluster_phi': 0.0,
        'mean_cluster_eta': 0.0,
        'mean_cluster_phi': 0.0,
        'cluster_pt_ratio': 0.0,  # Ratio of highest to second highest cluster pT
        'cluster_size_ratio': 0.0,  # Ratio of largest to second largest cluster size
        'std_cluster_eta': 0.0,  # Spread of cluster positions in eta (jet shape)
        'std_cluster_phi': 0.0,  # Spread of cluster positions in phi (jet shape)
        'eta_phi_asymmetry': 0.0,  # (std_eta-std_phi)/(std_eta+std_phi), bounded [-1,1]: >0 eta-elongated, <0 phi-elongated
        'delta_r_leading': 0.0,  # angular separation between the two leading (highest-pT) clusters
        'leading_pt_fraction': 1.0,  # fraction of the jet's total pT carried by the single leading cluster
    }

    if not clusters:
        return features

    cluster_pts = []
    cluster_sizes = []
    cluster_etas = []
    cluster_phis = []

    for cluster in clusters:
        cluster = np.array(cluster)
        pt = np.sum(cluster[:, 2])
        size = len(cluster)
        eta = np.mean(cluster[:, 0])  # eta is first column
        phi = np.mean(cluster[:, 1])  # phi is second column

        cluster_pts.append(pt)
        cluster_sizes.append(size)
        cluster_etas.append(eta)
        cluster_phis.append(phi)

    cluster_pts = np.array(cluster_pts)
    cluster_sizes = np.array(cluster_sizes)
    cluster_etas = np.array(cluster_etas)
    cluster_phis = np.array(cluster_phis)

    # Order clusters by pT (descending) once, keeping eta/phi/size aligned to
    # each cluster -- needed for delta_r_leading and leading_pt_fraction,
    # which care about *which* cluster is leading, not just the pT values.
    order = np.argsort(cluster_pts)[::-1]
    sorted_pts = cluster_pts[order]
    sorted_sizes = cluster_sizes[order]
    sorted_etas = cluster_etas[order]
    sorted_phis = cluster_phis[order]

    # Calculate additional features
    pt_ratio = sorted_pts[0] / sorted_pts[1] if len(sorted_pts) > 1 else 1.0
    size_ratio = sorted_sizes[0] / sorted_sizes[1] if len(sorted_sizes) > 1 else 1.0

    total_pt = np.sum(cluster_pts)
    leading_pt_fraction = sorted_pts[0] / total_pt if total_pt > 0 else 1.0

    if len(sorted_pts) > 1:
        delta_r_leading = np.sqrt((sorted_etas[0] - sorted_etas[1])**2 +
                                   (sorted_phis[0] - sorted_phis[1])**2)
    else:
        delta_r_leading = 0.0

    std_eta = np.std(cluster_etas)
    std_phi = np.std(cluster_phis)
    # A ratio (std_eta / std_phi) blows up whenever std_phi is near zero, even
    # with an epsilon guard -- a handful of such jets end up as extreme
    # outliers that swamp the rest of the distribution. The normalized
    # difference is bounded in [-1, 1] regardless of how small either spread
    # gets, at the cost of being 0 (not 1) for a "round" jet.
    denom = std_eta + std_phi
    eta_phi_asymmetry = (std_eta - std_phi) / denom if denom > 0 else 0.0

    features.update({
        'max_cluster_pt': np.max(cluster_pts),
        'mean_cluster_pt': np.mean(cluster_pts),
        'std_cluster_pt': np.std(cluster_pts),
        'max_cluster_size': np.max(cluster_sizes),
        'mean_cluster_size': np.mean(cluster_sizes),
        'std_cluster_size': np.std(cluster_sizes),
        'total_pt': total_pt,
        'max_cluster_eta': np.max(np.abs(cluster_etas)),
        'max_cluster_phi': np.max(np.abs(cluster_phis)),
        'mean_cluster_eta': np.mean(np.abs(cluster_etas)),
        'mean_cluster_phi': np.mean(np.abs(cluster_phis)),
        'cluster_pt_ratio': pt_ratio,
        'cluster_size_ratio': size_ratio,
        'std_cluster_eta': std_eta,
        'std_cluster_phi': std_phi,
        'eta_phi_asymmetry': eta_phi_asymmetry,
        'delta_r_leading': delta_r_leading,
        'leading_pt_fraction': leading_pt_fraction,
    })

    return features


def get_auc_score(y_true, y_pred_proba): 
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    return roc_auc