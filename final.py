import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Etape1: lire une image et calculer ses histogrammes de couleurs
def calculate_color_histogram(image_path, bins=32):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur : Impossible de lire l'image {image_path}")
        return None
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Etape2: lire une image et calculer ses moments de Hu
def calculate_hu_moments(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur : Impossible de lire l'image {image_path}")
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments)
    return hu_moments.flatten()

# Etape3: Calculer la distance entre deux histogrammes de couleurs (Chi-2)
def chi2_distance(histA, histB):
    return 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + 1e-10))

# Etape4: Calculer la distance entre deux vecteurs de moments de Hu (Euclidienne)
def euclidean_distance(vecA, vecB):
    return np.sqrt(np.sum((vecA - vecB) ** 2))

# Etape5: Fonction globale de similarité entre deux images
def calculate_similarity(histA, histB, huA, huB, weight_color=0.5, weight_shape=10):
    color_dist = chi2_distance(histA, histB)
    shape_dist = euclidean_distance(huA, huB)
    return weight_color * color_dist + weight_shape * shape_dist

# Etape6: Fonction pour vérifier si un fichier est une image
def is_image_file(filename):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
    return filename.lower().endswith(valid_extensions)

# Etape8: Extraire les caractéristiques de toutes les images dans le dataset
def extract_features(dataset_path):
    features = []
    image_paths = []
    
    for image_name in os.listdir(dataset_path):
        if not is_image_file(image_name):
            continue
        
        image_path = os.path.join(dataset_path, image_name)
        image_color_hist = calculate_color_histogram(image_path)
        image_hu_moments = calculate_hu_moments(image_path)
        
        if image_color_hist is None or image_hu_moments is None:
            continue
        
        combined_features = np.hstack([image_color_hist, image_hu_moments])
        features.append(combined_features)
        image_paths.append(image_path)
    
    return features, image_paths

# Étape9: Appliquer k-means et visualiser les clusters
def apply_kmeans(features, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(features)
    return kmeans

# Étape10: Visualiser les clusters avec k-means
def plot_clusters(kmeans, features):
    plt.figure(figsize=(10, 6))
    plt.scatter(features[:, 0], features[:, 1], c=kmeans.labels_, cmap='rainbow')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black', marker='x', s=200)
    plt.title('Clusters de k-means')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Étape11: Trouver les images similaires en utilisant les clusters
def search_similar_images(query_image_path, dataset_path, num_results=10, num_clusters=7):
    query_color_hist = calculate_color_histogram(query_image_path)
    query_hu_moments = calculate_hu_moments(query_image_path)
    
    if query_color_hist is None or query_hu_moments is None:
        print("Erreur lors du calcul des caractéristiques de l'image requête.")
        return []
    
    query_features = np.hstack([query_color_hist, query_hu_moments])
    
    features, image_paths = extract_features(dataset_path)
    features = np.array(features)
    
    kmeans = apply_kmeans(features, num_clusters)
    
    plot_clusters(kmeans, features)
    
    cluster_label = kmeans.predict([query_features])[0]
    cluster_indices = np.where(kmeans.labels_ == cluster_label)[0]
    
    results = []
    for idx in cluster_indices:
        image_path = image_paths[idx]
        image_color_hist = features[idx][:32**3]
        image_hu_moments = features[idx][32**3:]
        similarity = calculate_similarity(query_color_hist, image_color_hist, query_hu_moments, image_hu_moments)
        results.append((os.path.basename(image_path), similarity))
    
    results.sort(key=lambda x: x[1])
    return results[:num_results]

# Fonction pour évaluer les résultats
def evaluate_results(results, ground_truth_labels, dataset_path):
    true_positive = 0
    for result in results:
        image_name = result[0]
        image_path = os.path.join(dataset_path, image_name)
        if ground_truth_labels[image_path] == ground_truth_labels[query_image_path]:
            true_positive += 1
    
    precision = true_positive / len(results)
    recall = true_positive / sum(1 for label in ground_truth_labels.values() if label == ground_truth_labels[query_image_path])
    f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f_measure

# Chemin de l'image requête et du dataset
query_image_path = 'coil-100/obj1__0.png'
dataset_path = 'coil-100'

# Rechercher les images similaires
results = search_similar_images(query_image_path, dataset_path)

# Afficher les résultats
print("Image\tDistance")
for result in results:
    print(f"{result[0]}\t{result[1]:.3f}")