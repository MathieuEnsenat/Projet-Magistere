import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
import random
import os


def generate_arial_dataset(filename="dataset_arial.csv", samples_per_char=100):

    mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    try:
        font_path = "Windows/Fonts/arial.ttf"  # Assure-toi que le fichier est dans le dossier ou utilise un chemin absolu
        font_size = 22
        font = ImageFont.truetype(font_path, font_size)
    except:
        print("Police Arial non trouvée, utilisation de la police par défaut.")
        font = ImageFont.load_default()

    dataset = []

    print(f"Génération de {samples_per_char} exemplaires par caractère...")

    for label, char in enumerate(mapping):
        for _ in range(samples_per_char):
            img = Image.new('L', (40, 40), 0)
            draw = ImageDraw.Draw(img)
            w, h = draw.textsize(char, font=font) if hasattr(draw, 'textsize') else (15, 20)
            draw.text(((40 - w) // 2, (40 - h) // 2), char, fill=255, font=font)
            angle = random.uniform(-15, 15)
            img = img.rotate(angle)
            scale = random.uniform(0.7, 1.2)
            new_size = int(40 * scale)
            img = img.resize((new_size, new_size), Image.BILINEAR)
            final_img = Image.new('L', (28, 28), 0)
            offset_x = (28 - img.size[0]) // 2 + random.randint(-2, 2)
            offset_y = (28 - img.size[1]) // 2 + random.randint(-2, 2)
            final_img.paste(img, (offset_x, offset_y))
            pixels = np.array(final_img).flatten()
            line = np.insert(pixels, 0, label)
            dataset.append(line)

    # 2. Sauvegarde en CSV
    print(f"Sauvegarde de {len(dataset)} lignes dans {filename}...")
    df = pd.DataFrame(dataset)
    df.to_csv(filename, index=False, header=False)
    print("Dataset généré avec succès.")


# Exécution
if __name__ == "__main__":
    #generate_arial_dataset("dataset_arial.csv", samples_per_char=5000)

    def load_arial_dataset(filename="dataset_arial.csv"):
        # 1. Charger le CSV (sans en-tête car le script n'en génère pas)
        print("Chargement des données...")
        df = pd.read_csv(filename, header=None)
        
        # 2. Séparer le label (1ère colonne) des pixels (784 colonnes suivantes)
        y = df.iloc[:, 0].values  # Labels (0 à 61)
        X = df.iloc[:, 1:].values # Pixels (0 à 255)
        
        # 3. Normalisation (très important pour les réseaux de neurones)
        # On passe d'une plage [0, 255] à [0, 1] pour aider la convergence
        X = X.astype('float32') / 255.0
        
        # 4. Redimensionner pour un CNN (Convolutional Neural Network)
        # On repasse de 784 à (28, 28, 1) car les CNN attendent des images 2D + canal couleur
        X = X.reshape(-1, 28, 28)

        
        return X, y

    # Utilisation
    X_train, y_train = load_arial_dataset("dataset_arial.csv")
    print(f"Forme des images : {X_train.shape}") # Devrait être (nb_images, 28, 28, 1)
    print(f"Forme des labels : {y_train.shape}")

    # Afficher la première image chargée
    import matplotlib.pyplot as plt
    for i in range(270000, 270005):
        plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {y_train[i]}")
        plt.show()