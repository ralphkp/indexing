import cv2
import numpy as np
import os
from scipy.spatial.distance import euclidean
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class ImageClassifier:
    def __init__(self, base_directory):
        self.base_directory = base_directory
        self.all_labels = []
        self.all_predictions = []
        
    def load_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        return image
    
    def calculate_histogram(self, image, bins=(32, 32, 32)):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
        cv2.normalize(histogram, histogram)
        return histogram.flatten()
    
    def calculate_hu_moments(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        moments = cv2.moments(gray_image)
        hu_moments = cv2.HuMoments(moments).flatten()
        return hu_moments
    
    def compare_images(self, query_histogram, query_hu_moments, base_image):
        base_histogram = self.calculate_histogram(base_image)
        base_hu_moments = self.calculate_hu_moments(base_image)
        
        color_distance = cv2.compareHist(query_histogram, base_histogram, cv2.HISTCMP_CHISQR)
        shape_distance = euclidean(query_hu_moments, base_hu_moments)
        
        global_distance = 0.5 * color_distance + 0.5 * shape_distance
        
        return global_distance
    
    def find_similar_images(self, query_image_path, query_class, N=5):
        query_image = self.load_image(query_image_path)
        query_histogram = self.calculate_histogram(query_image)
        query_hu_moments = self.calculate_hu_moments(query_image)
        
        distances = []
        
        for image_name in os.listdir(self.base_directory):
            base_image_path = os.path.join(self.base_directory, image_name)
            if base_image_path == query_image_path:
                continue
            
            base_image = self.load_image(base_image_path)
            
            # Get the class label of the base image
            base_class = image_name.split('__')[0]
            
            if base_class == query_class:
                # Calculate distance only for the same class vs all others
                distance = self.compare_images(query_histogram, query_hu_moments, base_image)
                distances.append((image_name, distance))
            
            del base_image
        
        distances.sort(key=lambda x: x[1])
        
        return distances[:N]
    
    def evaluate_one_vs_all(self, query_images, N=5):
        for query_image_name in query_images:
            query_image_path = os.path.join(self.base_directory, query_image_name)
            
            if not os.path.exists(query_image_path):
                print(f"L'image requête '{query_image_name}' n'existe pas dans le dossier '{self.base_directory}'.")
                continue
            
            query_class = query_image_name.split('__')[0]
            similar_images = self.find_similar_images(query_image_path, query_class, N)
            
            if not similar_images:
                print(f"Aucune image similaire trouvée pour '{query_image_name}'.")
                continue
            
            true_label = query_class
            for similar_image_name, _ in similar_images:
                predicted_label = similar_image_name.split('__')[0]
                self.all_labels.append(true_label)
                self.all_predictions.append(predicted_label)
    
    def plot_confusion_matrix(self):
        labels = sorted(set(self.all_labels + self.all_predictions))
        cm = confusion_matrix(self.all_labels, self.all_predictions, labels=labels)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
        plt.title("Matrice de Confusion")
        plt.show()

# Exemple d'utilisation
base_directory = "/Users/macbook/Documents/SIMP26M2/Indexation/tp1/coil-100"  # Modifier le chemin selon votre dossier
query_images = ['obj1__0.png', 'obj2__0.png', 'obj3__0.png', 'obj4__0.png', 'obj5__0.png', 'obj6__0.png', 'obj7__0.png', 'obj8__0.png', 'obj9__0.png', 'obj10__0.png']

classifier = ImageClassifier(base_directory)
classifier.evaluate_one_vs_all(query_images)
classifier.plot_confusion_matrix()



# import cv2
# import numpy as np
# import os
# from scipy.spatial.distance import euclidean
# import time
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt

# # Fonction pour charger une image
# def load_image(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"Image not found at {image_path}")
#     return image

# # Fonction pour calculer l'histogramme de couleurs
# def calculate_histogram(image, bins=(32, 32, 32)):
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     histogram = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
#     cv2.normalize(histogram, histogram)
#     return histogram.flatten()

# # Fonction pour calculer les moments de Hu
# def calculate_hu_moments(image):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     moments = cv2.moments(gray_image)
#     hu_moments = cv2.HuMoments(moments).flatten()
#     return hu_moments

# # Fonction pour comparer deux images
# def compare_images(query_histogram, query_hu_moments, base_image):
#     base_histogram = calculate_histogram(base_image)
#     base_hu_moments = calculate_hu_moments(base_image)
    
#     # Comparaison des histogrammes (utilisation de Chi-square)
#     color_distance = cv2.compareHist(query_histogram, base_histogram, cv2.HISTCMP_CHISQR)
    
#     # Comparaison des moments de Hu (distance euclidienne)
#     shape_distance = euclidean(query_hu_moments, base_hu_moments)
    
#     # Calcul de la distance globale avec pondération égale
#     global_distance = 0.5 * color_distance + 0.5 * shape_distance
    
#     return global_distance

# # Fonction pour trouver les N images les plus similaires
# def find_similar_images(query_image_path, base_directory, N=5):
#     query_image = load_image(query_image_path)
#     query_histogram = calculate_histogram(query_image)
#     query_hu_moments = calculate_hu_moments(query_image)
    
#     # Supprimer l'image requête de la mémoire (ne garder que sa description)
#     del query_image
    
#     distances = []
    
#     for image_name in os.listdir(base_directory):
#         base_image_path = os.path.join(base_directory, image_name)
#         if base_image_path == query_image_path:  # Skip the query image itself
#             continue
        
#         base_image = load_image(base_image_path)
        
#         # Calcul de la distance
#         distance = compare_images(query_histogram, query_hu_moments, base_image)
#         distances.append((image_name, distance))
        
#         # Supprimer l'image de la mémoire après utilisation
#         del base_image
    
#     # Tri des distances par ordre croissant
#     distances.sort(key=lambda x: x[1])
    
#     # Retourner les N images les plus proches
#     return distances[:N]

# # Fonction pour évaluer l'algorithme et tracer la matrice de confusion
# def evaluate_and_plot_confusion_matrix(query_images, base_directory, N=5):
#     all_labels = []
#     all_predictions = []
    
#     for query_image_name in query_images:
#         query_image_path = os.path.join(base_directory, query_image_name)
        
#         if not os.path.exists(query_image_path):
#             print(f"L'image requête '{query_image_name}' n'existe pas dans le dossier '{base_directory}'.")
#             continue
        
#         similar_images = find_similar_images(query_image_path, base_directory, N)
        
#         # Vérification si similar_images est vide
#         if not similar_images:
#             print(f"Aucune image similaire trouvée pour '{query_image_name}'.")
#             continue
        
#         # Collecter les labels réels et prédits
#         true_label = query_image_name.split('__')[0]
#         for similar_image_name, _ in similar_images:
#             predicted_label = similar_image_name.split('__')[0]
#             all_labels.append(true_label)
#             all_predictions.append(predicted_label)
    
#     # Calculer la matrice de confusion
#     labels = sorted(set(all_labels + all_predictions))
#     cm = confusion_matrix(all_labels, all_predictions, labels=labels)
    
#     # Afficher la matrice de confusion
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
#     disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
#     plt.title("Matrice de Confusion")
#     plt.show()

# # Chemin du dossier de la base d'images
# base_directory = "/Users/macbook/Documents/SIMP26M2/Indexation/tp1/coil-100"  # Changez ce chemin selon la localisation de votre dossier

# # Liste des images de requête pour évaluation
# query_images = ['obj1__0.png', 'obj2__0.png', 'obj3__0.png', 'obj4__0.png', 'obj5__0.png','obj6__0.png, obj7__0.png, obj8__0.png, obj9__0.png, obj10'] 

# # Évaluer l'algorithme et tracer la matrice de confusion
# evaluate_and_plot_confusion_matrix(query_images, base_directory, N=5)
