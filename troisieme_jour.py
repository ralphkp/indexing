import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
import time

# Fonction pour charger une image
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image

# Fonction pour calculer l'histogramme de couleurs
def calculate_histogram(image, bins=(32, 32, 32)):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(histogram, histogram)
    return histogram.flatten()

# Fonction pour calculer les moments de Hu
def calculate_hu_moments(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray_image)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments

# Fonction pour extraire les descripteurs des images
def extract_descriptors(base_directory):
    descriptors = []
    image_paths = []
    for image_name in os.listdir(base_directory):
        image_path = os.path.join(base_directory, image_name)
        image = load_image(image_path)
        histogram = calculate_histogram(image)
        hu_moments = calculate_hu_moments(image)
        descriptor = np.hstack([histogram, hu_moments])
        descriptors.append(descriptor)
        image_paths.append(image_path)
    return np.array(descriptors), image_paths

# Fonction pour effectuer le clustering avec k-means
def perform_clustering(descriptors, k=10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(descriptors)
    return kmeans

# Fonction pour trouver les images similaires dans un cluster
def find_similar_in_cluster(query_image_path, cluster, cluster_image_paths, descriptors, N=5):
    query_image = load_image(query_image_path)
    query_histogram = calculate_histogram(query_image)
    query_hu_moments = calculate_hu_moments(query_image)
    query_descriptor = np.hstack([query_histogram, query_hu_moments])
    
    distances = []
    for idx in cluster:
        base_descriptor = descriptors[idx]
        distance = np.linalg.norm(query_descriptor - base_descriptor)  # Utilisation de la norme L2 comme mesure de distance
        distances.append((cluster_image_paths[idx], distance))
    
    distances.sort(key=lambda x: x[1])
    return distances[:N]

# Fonction pour trouver les N images les plus similaires en utilisant le clustering
def find_similar_images_with_clustering(query_image_path, base_directory, kmeans, image_paths, descriptors, N=5):
    query_image = load_image(query_image_path)
    query_histogram = calculate_histogram(query_image)
    query_hu_moments = calculate_hu_moments(query_image)
    query_descriptor = np.hstack([query_histogram, query_hu_moments])
    
    cluster_label = kmeans.predict([query_descriptor])[0]
    cluster_indices = np.where(kmeans.labels_ == cluster_label)[0]
    
    similar_images = find_similar_in_cluster(query_image_path, cluster_indices, image_paths, descriptors, N)
    return similar_images

# Fonction principale pour évaluer le système avec clustering
def evaluate_system_with_clustering(query_images, base_directory, k=10, N=5):
    # Extraire les descripteurs et chemins des images
    descriptors, image_paths = extract_descriptors(base_directory)
    
    # Effectuer le clustering
    kmeans = perform_clustering(descriptors, k=k)
    
    total_time = 0
    precision_scores = []
    recall_scores = []
    
    for query_image_name in query_images:
        query_image_path = os.path.join(base_directory, query_image_name)
        
        if not os.path.exists(query_image_path):
            print(f"L'image requête '{query_image_name}' n'existe pas dans le dossier '{base_directory}'.")
            continue
        
        start_time = time.time()
        similar_images = find_similar_images_with_clustering(query_image_path, base_directory, kmeans, image_paths, descriptors, N)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        
        # Vérification si similar_images est vide
        if not similar_images:
            print(f"Aucune image similaire trouvée pour '{query_image_name}'.")
            continue
        
        print(f"Requête: {query_image_name}")
        print(f"Temps de recherche: {elapsed_time:.4f} secondes")
        
        # Calculer la précision et le rappel
        relevant_retrieved = sum(1 for img, _ in similar_images if img.split('__')[0] == query_image_name.split('__')[0])
        total_relevant = sum(1 for img in os.listdir(base_directory) if img.split('__')[0] == query_image_name.split('__')[0])
        
        precision = relevant_retrieved / N if N > 0 else 0
        recall = relevant_retrieved / total_relevant if total_relevant else 0  # Éviter division par zéro
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        
        print(f"Précision: {precision:.4f}")
        print(f"Rappel: {recall:.4f}")
        print("----------------------------")
    
    # Calculer les moyennes
    if len(query_images) > 0:
        average_time = total_time / len(query_images)
    else:
        average_time = 0
    
    if precision_scores:
        average_precision = np.mean(precision_scores)
    else:
        average_precision = float('nan')
    
    if recall_scores:
        average_recall = np.mean(recall_scores)
    else:
        average_recall = float('nan')
    
    if (average_precision + average_recall) > 0:
        average_f1_score = (2 * average_precision * average_recall) / (average_precision + average_recall)
    else:
        average_f1_score = 0
    
    print("Évaluation globale avec clustering:")
    print(f"Temps moyen de recherche: {average_time:.4f} secondes")
    print(f"Précision moyenne: {average_precision:.4f}")
    print(f"Rappel moyen: {average_recall:.4f}")
    print(f"Score F1 moyen: {average_f1_score:.4f}")

# Chemin du dossier de la base d'images
base_directory = "coil-100"  # Modifier le chemin selon votre dossier

# Liste des images de requête pour évaluation
query_images = ['obj1__100.png', 'obj2__100.png', 'obj3__100.png', 'obj4__100.png', 'obj5__100.png']

# Évaluer l'algorithme avec clustering
evaluate_system_with_clustering(query_images, base_directory, k=5, N=5)
