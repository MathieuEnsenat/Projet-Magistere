import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
import random
import os


def generate_arial_dataset(filename="dataset_arial.csv", samples_per_char=100):

    mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    try:
        font_path = "/Library/Fonts/Arial.ttf"  # Assure-toi que le fichier est dans le dossier ou utilise un chemin absolu
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
    generate_arial_dataset("dataset_arial.csv", samples_per_char=5000)