import cv2
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Lire l’image d’entrée
def lire_image(chemin_image):
    image = cv2.imread(chemin_image)
    if image is None:
        raise ValueError(f"Impossible de charger l'image : {chemin_image}")
    return image

# Calculer les caractéristiques : Histogrammes de couleurs
def calculer_histogramme(image):
    histogramme = []
    for canal in range(3):  # Pour les canaux R, V, B
        hist = cv2.calcHist([image], [canal], None, [32], [0, 256])
        histogramme.extend(hist.flatten())
    histogramme = np.array(histogramme)
    histogramme /= histogramme.sum()  # Normalisation
    return histogramme

# Calculer les caractéristiques : Moments de Hu pour les formes
def calculer_moments_hu(image):
    image_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(image_gris)
    moments_hu = cv2.HuMoments(moments).flatten()
    return moments_hu

# Calculer la distance entre les histogrammes de couleurs
def distance_histogramme(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

# Calculer la distance entre les moments de Hu
def distance_moments_hu(moments1, moments2):
    return np.linalg.norm(moments1 - moments2)

# Fonction de similarité globale
def calculer_distance_globale(image1, image2):
    hist1 = calculer_histogramme(image1)
    hist2 = calculer_histogramme(image2)
    moments1 = calculer_moments_hu(image1)
    moments2 = calculer_moments_hu(image2)

    dist_couleur = distance_histogramme(hist1, hist2)
    dist_forme = distance_moments_hu(moments1, moments2)
    
    omega1 = 0.5
    omega2 = 0.5
    distance_totale = omega1 * dist_couleur + omega2 * dist_forme
    return distance_totale

# Vérifie si le fichier est une image
def est_une_image(fichier):
    valid_extensions = ('.png')
    return fichier.lower().endswith(valid_extensions)

# Recherche d'images les plus similaires
def recherche_images_similaires(chemin_requete, dossier_images, N=5):
    image_requete = lire_image(chemin_requete)
    distances = []
    autres_images = []

    for fichier in os.listdir(dossier_images):
        if est_une_image(fichier):
            chemin_image = os.path.join(dossier_images, fichier)
            if os.path.isfile(chemin_image):
                try:
                    image_base = lire_image(chemin_image)
                    distance = calculer_distance_globale(image_requete, image_base)
                    distances.append((fichier, distance))
                except ValueError as e:
                    print(e)
    
    distances = sorted(distances, key=lambda x: x[1])
    top_similaires = distances[:N]
    autres_images = distances[N:]

    return top_similaires, autres_images

# Fonction pour extraire le label à partir du chemin de l'image
def extraire_label_de_chemin(chemin):
    return os.path.basename(chemin).split('__')[0] + '__0'

# Générer des prédictions pour chaque image de requête
def generer_predictions_pour_image(chemin_image_requete, dossier_images, N=5):
    y_true = []
    y_pred = []
    
    label_true = extraire_label_de_chemin(chemin_image_requete)
    top_similaires, autres_images = recherche_images_similaires(chemin_image_requete, dossier_images, N=N)
    
    # Obtenir les étiquettes des résultats
    labels_pred_top = [extraire_label_de_chemin(fichier) for fichier, _ in top_similaires]
    labels_pred_autres = [extraire_label_de_chemin(fichier) for fichier, _ in autres_images]
    
    for label_pred in labels_pred_top:
        y_true.append(label_true)
        y_pred.append(label_pred)
    
    for label_pred in labels_pred_autres:
        y_true.append('autres')
        y_pred.append(label_pred)
    
    return y_true, y_pred

# Calculer les métriques de performance
def calculer_metrics(y_true, y_pred, label_true):
    # Définir explicitement les classes pour s'assurer que toutes les valeurs sont prises en compte
    classes = [label_true, 'autres']
    y_true_binaire = [1 if y == label_true else 0 for y in y_true]
    y_pred_binaire = [1 if y == label_true else 0 for y in y_pred]

    cm = confusion_matrix(y_true_binaire, y_pred_binaire, labels=[1, 0])
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
        if len(cm) > 1:
            tp = cm[1, 1]
            fn = cm[1, 0]
            fp = cm[0, 1]
            tn = cm[0, 0]
        elif len(cm) == 1:
            tp = cm[0, 0]
    return tn, fp, fn, tp, cm

# Exemple d'utilisation avec plusieurs images de requête
dossier_images = 'coil-100'
images_requete = [
    'coil-100/obj1__100.png'  # Ajoutez plus d'images de requête ici
]

# Générer et afficher les métriques pour chaque image de requête
for chemin_image_requete in images_requete:
    y_true, y_pred = generer_predictions_pour_image(chemin_image_requete, dossier_images, N=10)
    
    # Définir le label vrai de l'image de requête
    label_true = extraire_label_de_chemin(chemin_image_requete)
    
    # Calculer les métriques
    tn, fp, fn, tp, cm = calculer_metrics(y_true, y_pred, label_true)
    
    # Afficher la matrice de confusion
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[label_true, 'autres'], yticklabels=[label_true, 'autres'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {os.path.basename(chemin_image_requete)}')
    plt.show()
    
    # Afficher le rapport de classification
    print(f'Report for {os.path.basename(chemin_image_requete)}:')
    print(classification_report(y_true, y_pred, target_names=[label_true, 'autres']))
    
    # Afficher les métriques
    print(f'True Positives (TP): {tp}')
    print(f'False Positives (FP): {fp}')
    print(f'False Negatives (FN): {fn}')
    print(f'True Negatives (TN): {tn}')
    print("\n\n")