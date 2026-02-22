import numpy as np
from PIL import Image

def importer_image(fichier):
    img = Image.open(fichier)
    img = img.convert("RGB")
    return np.array(img)


def rgb_a_gris(image):
    return np.average(image, axis=2, weights=[0.299, 0.587, 0.114])


#thresholding simple
def binaire(image):
    resultat = np.zeros_like(image)
    resultat[image < 200] = 1
    moyenne = np.mean(image)
    if moyenne > 127:
        resultat[image < 210] = 1
    else:
        resultat[image > 210] = 1
    return resultat

#thresholding adaptatif
def binarisation(image, k, C):
    resultat = np.zeros_like(image)
    longueur, largeur = image.shape
    image_pad = np.zeros((longueur + k - 1, largeur + k - 1))
    image_pad[k // 2:-(k // 2), k // 2:-(k // 2)] = image
    for i in range(longueur):
        for j in range(largeur):
            seuil = np.mean(image_pad[i:i + k, j:j + k])
            if image[i, j] < seuil - C:
                resultat[i, j] = 1
    return resultat

#thresholding avec méthode d'Otsu
def binarisationOtsu(image):
    #valeur de gris minimum de l'image
    image_min = int(np.min(image))
    #valeur de gris max
    image_max = int(np.max(image))
    #seuil qui minimise la variance intra-classes
    seuil = min(range(image_min+1, image_max), key=lambda seuil: varInter(image, seuil))
    return np.where(image < seuil, 1, 0)

def varInter(image, seuil):
    N = image.size
    res = np.sum(image >= seuil) * np.var(image, where=image >= seuil)
    res += np.sum(image < seuil) * np.var(image, where=image < seuil)
    return res/N



