'''
from traitement_image import rgb_a_gris, binarisation, generar_imagen_rgb, importer_image, binaire
from decoupage import pre_normalisation, cadrage2, scanner_horizontal, scanner_vertical, cadrage, normaliser, \
    redimensionner, post_decoupage
import numpy as np
import matplotlib.pyplot as plt
from time import time
from IA import IA
import pandas as pd

#Traitement
imagen_test = importer_image("p.png")
imagen_gris = rgb_a_gris(imagen_test)
imagen_binaria = binaire(imagen_gris)

#Decoupage
mots_pre = cadrage2(imagen_binaria)
mot_propre = pre_normalisation(mots_pre)

#revision visuel
for i, mot in enumerate(mot_propre):
    print(f"Palabra {i}:")
    for lettre in mot:
        img_norm = normaliser(lettre)
        plt.imshow(img_norm, cmap='gray')
        plt.show()

IA=IA([784, 128, 62],0.01)
df_train = pd.read_csv('Magistère/emnist-byclass-train.csv', header=None)
df_test = pd.read_csv('Magistère/emnist-byclass-test.csv', header=None)
IA.training(df_train)

with open("Résultat.txt", "w") as f:
    for mot in mot_propre:
        f.write(f"{IA.predict(mot)}\n")
'''

from traitement_image import rgb_a_gris, binarisation, importer_image, binaire
from decoupage import pre_normalisation, cadrage2, normaliser
import numpy as np
import matplotlib.pyplot as plt
from IA import IA
import pandas as pd

# 1. Initialisation et Chargement (Attention : l'entraînement est long)
# Dim : 784 entrées, 128 cachées, 62 sorties (ByClass)
IA = IA([784, 128, 62], 0.01)
df_train = pd.read_csv('Magistère/emnist-byclass-train.csv', header=None)
IA.training(df_train)
# Note : Dans un vrai cas, on chargerait des poids pré-entraînés ici
# Pour l'exemple, on suppose que l'IA est entraînée ou qu'on lance l'entraînement :
# df_train = pd.read_csv('Magistère/emnist-byclass-train.csv', header=None)
# mon_ia.training(df_train)

# 2. Traitement de l'image
imagen_test = importer_image("p.png")
imagen_gris = rgb_a_gris(imagen_test)
imagen_binaria = binaire(imagen_gris)

# 3. Découpage en mots et lettres
mots_pre = cadrage2(imagen_binaria)
mots_propres = pre_normalisation(mots_pre)

# 4. Prédiction et écriture
with (open("Résultat.txt", "w", encoding="utf-8") as f):
    for mot in mots_propres:
        lettres_normalisees = []
        for lettre in mot:
            # CRITIQUE : Il faut normaliser chaque lettre en 28x28 avant l'IA
            img_norm = normaliser(lettre)
            lettres_normalisees.append(img_norm)

        # On prédit le mot entier
        mot_predit = IA.predict(lettres_normalisees)

        print(f"Mot détecté : {mot_predit}")
        f.write(mot_predit + " ")  # Ajoute un espace entre les mots
    f.write("\n")
