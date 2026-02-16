from traitement_image import rgb_a_gris, binarisation, generar_imagen_rgb, importer_image, binaire
from decoupage import pre_normalisation, cadrage2, scanner_horizontal, scanner_vertical, cadrage, normaliser, \
    redimensionner, post_decoupage
import numpy as np
import matplotlib.pyplot as plt
from time import time
from IA import IA
import pandas as pd

imagen_test = importer_image("t.png")
imagen_gris = rgb_a_gris(imagen_test)
imagen_binaria = binaire(imagen_gris)
mots_pre = cadrage2(imagen_binaria)
mots_propre = pre_normalisation(mots_pre)

for i, mot in enumerate(mots_propre):
    print(f"Palabra {i}:")
    for lettre in mot:
        img_norm = normaliser(lettre)
        plt.imshow(img_norm, cmap='gray')
        plt.show()


IA=IA([784, 128, 62],0.01)
df_train = pd.read_csv('Magistère/emnist-byclass-train.csv', header=None)
IA.training(df_train)

with open("resultat_ocr.txt", "w") as file:
    for mot in mots_propre:
        lettres_normalisees = []
        for lettre_img in mot:
            img_norm = normaliser(lettre_img)
            lettres_normalisees.append(img_norm)
        file.write(IA.predict(lettres_normalisees))
        file.write("\n")





