import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random as rd


def importer_image(fichier):
    img = Image.open(fichier)
    img = img.convert("RGB")
    return np.array(img)


def generar_imagen_rgb():
    alto = 50
    ancho = 100
    imagen = np.ones((alto, ancho, 3), dtype=int) * 255
    letra1 = rd.randint(0, 255)
    letra12 = rd.randint(0, 255)
    letra13 = rd.randint(0, 255)
    letra2 = rd.randint(0, 255)
    letra22 = rd.randint(0, 255)
    letra23 = rd.randint(0, 255)
    letra3 = rd.randint(0, 255)
    letra32 = rd.randint(0, 255)
    letra33 = rd.randint(0, 255)

    imagen[10:40, 10:15] = [letra1, letra12, letra13]
    imagen[10:15, 10:25] = [letra1, letra12, letra13]
    imagen[10:25, 20:25] = [letra1, letra12, letra13]
    imagen[20:25, 10:25] = [letra1, letra12, letra13]
    imagen[25:40, 20:25] = [letra1, letra12, letra13]
    imagen[10:40, 40:45] = [letra2, letra22, letra23]
    imagen[10:15, 40:55] = [letra2, letra22, letra23]
    imagen[35:40, 40:55] = [letra2, letra22, letra23]
    imagen[25:40, 50:55] = [letra2, letra22, letra23]
    imagen[10:40, 70:75] = [letra3, letra32, letra33]
    imagen[10:15, 70:85] = [letra3, letra32, letra33]
    imagen[20:25, 70:80] = [letra3, letra32, letra33]
    imagen[35:40, 70:85] = [letra3, letra32, letra33]

    return imagen


def rgb_a_gris(image):
    return np.average(image, axis=2, weights=[0.299, 0.587, 0.114])


def binarisation(image, k, C):
    resultat = np.zeros_like(image)
    longueur, largeur = image.shape
    image_pad = np.zeros((longueur + k - 1, largeur + k - 1))
    image_pad[k // 2:-(k // 2), k // 2:-(k // 2)] = image
    for i in range(longueur):
        for j in range(largeur):
            seuil = np.mean(image_pad[i:i + k, j:j + k])
            if image[i, j] > seuil - C:
                resultat[i, j] = 1
    return resultat


def binaire(image):
    resultat = np.zeros_like(image)
    resultat[image < 200] = 1
    moyenne = np.mean(image)
    if moyenne > 127:
        resultat[image < 210] = 1
    else:
        resultat[image > 210] = 1
    return resultat



