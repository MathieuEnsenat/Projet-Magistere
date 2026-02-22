from traitement_image import rgb_a_gris, binarisationOtsu, importer_image
from decoupage import pre_normalisation, cadrage2, normaliser
import matplotlib.pyplot as plt
from IA import IA

mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
imagen_test = importer_image("data/test2.png")
imagen_gris = rgb_a_gris(imagen_test)
imagen_binaria = binarisationOtsu(imagen_gris)
plt.imshow(imagen_binaria, cmap='gray')
plt.show()
mots_pre = cadrage2(imagen_binaria)
mots_propre = pre_normalisation(mots_pre)
reseau = IA([784, 128, 47], 0.01)
reseau.load_csv("data/poids.csv")

for i, mot in enumerate(mots_propre):
    print(f"Palabra {i}:")
    for lettre in mot:
        img_norm = normaliser(lettre)
        plt.imshow(img_norm, cmap='gray')
        plt.show()
        print(reseau.predict([img_norm], mapping))

with open("Résultat.txt", "w", encoding="utf-8") as file:
    for i, mot in enumerate(mots_propre):
        lettres_normalisees = []
        for lettre_img in mot:
            img_norm = normaliser(lettre_img)
            lettres_normalisees.append(img_norm)
        if lettres_normalisees:
            mot_predit = reseau.predict(lettres_normalisees, mapping)
            print(f"Mot {i} reconnu : {mot_predit}")
            file.write(mot_predit + " ")

print("Reconnaissance terminée. Consultez 'Résultat.txt'.")



