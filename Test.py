import numpy as np
from traitement_image import importer_image, rgb_a_gris, binaire
from decoupage import cadrage2, pre_normalisation, normaliser
from IA import IA


def main(chemin_image, chemin_sortie):
    # 1. Chargement et Prétraitement de l'image
    print("Chargement de l'image...")
    img_rgb = importer_image(chemin_image)
    img_gris = rgb_a_gris(img_rgb)
    img_binaire = binaire(img_gris)  # Conversion en Noir et Blanc

    # 2. Segmentation (Découpage en mots puis en lettres)
    print("Segmentation de l'image en mots et lettres...")
    # cadrage2 renvoie une liste de mots, chaque mot est une liste de lettres (images)
    mots_bruts = cadrage2(img_binaire)
    # Nettoyage et post-découpage (gestion des lettres collées)
    mots_propres = pre_normalisation(mots_bruts)

    # 3. Initialisation de l'IA
    # Note : Assure-toi que les dimensions correspondent à ton modèle entraîné
    # Exemple : 784 (28x28) -> 128 (cachée) -> 62 (classes EMNIST)
    input_size = 28 * 28
    classes = 62
    mon_ia = IA([input_size, 128, classes], tauxapp=0.1)

    # /!\ IMPORTANT : Ici tu devrais normalement charger des poids pré-entraînés
    # mon_ia.charger_poids("mes_poids.npy")

    # 4. Reconnaissance et Reconstruction du texte
    print("Conversion en texte...")
    texte_final = []

    for mot in mots_propres:
        lettres_normalisees = []
        for lettre_img in mot:
            # On normalise chaque lettre au format 28x28 attendu par l'IA
            img_norm = normaliser(lettre_img)
            lettres_normalisees.append(img_norm)

        # Utilisation de la méthode predict de la classe IA
        mot_predit = mon_ia.predict(lettres_normalisees)
        texte_final.append(mot_predit)

    # 5. Sauvegarde dans un fichier texte
    phrase = " ".join(texte_final)
    with open(chemin_sortie, "w", encoding="utf-8") as f:
        f.write(phrase)

    print(f"Traitement terminé ! Texte extrait : {phrase}")
    print(f"Résultat sauvegardé dans {chemin_sortie}")


if __name__ == "__main__":
    # Remplace par le chemin de ton image et le nom du fichier voulu
    main("p.png", "Résultat.txt")

