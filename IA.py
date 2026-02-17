import numpy as np
import pandas as pd


class IA:
    def __init__(self, dim_couche, tauxapp):
        self.nbcouche = len(dim_couche) - 1
        self.nbneuroneparcouche = dim_couche
        self.tauxapp = tauxapp
        self.poids = {}
        self.biais = {}
        self.cache = {}
        self.initialisation_parametres()

    def initialisation_parametres(self):
        for l in range(1, len(self.nbneuroneparcouche)):
            # Initialisation de He (optimale pour ReLU)
            self.poids[f'W{l}'] = np.random.randn(self.nbneuroneparcouche[l], self.nbneuroneparcouche[l - 1]) * np.sqrt(
                2. / self.nbneuroneparcouche[l - 1])
            self.biais[f'b{l}'] = np.zeros((self.nbneuroneparcouche[l], 1))

    def load_csv(self, filename="poids.csv"):
        try:
            # On charge les données en ignorant les lignes de commentaires si elles existent
            data = np.loadtxt(filename, delimiter=",", comments='#')
            cursor = 0
            for l in range(1, self.nbcouche + 1):
                # Reconstruction de W
                shape_w = (self.nbneuroneparcouche[l], self.nbneuroneparcouche[l - 1])
                size_w = shape_w[0] * shape_w[1]
                self.poids[f'W{l}'] = data[cursor: cursor + size_w].reshape(shape_w)
                cursor += size_w

                # Reconstruction de b
                shape_b = (self.nbneuroneparcouche[l], 1)
                size_b = shape_b[0]
                self.biais[f'b{l}'] = data[cursor: cursor + size_b].reshape(shape_b)
                cursor += size_b
            print(f"Modèle chargé avec succès depuis {filename}")
        except Exception as e:
            print(f"Erreur lors du chargement : {e}")

    def save_csv(self, filename="poids.csv"):
        # On aplatit tout dans une liste unique pour un export propre
        params_to_save = []
        for l in range(1, self.nbcouche + 1):
            params_to_save.append(self.poids[f'W{l}'].flatten())
            params_to_save.append(self.biais[f'b{l}'].flatten())

        flat_list = np.concatenate(params_to_save)
        np.savetxt(filename, flat_list, delimiter=",", header="Poids et Biais concaténés")
        print("Sauvegarde terminée.")

    def activation_relu(self, x):
        return np.maximum(0, x)

    def d_relu(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        # Soustraction du max pour la stabilité numérique (évite exp(large_nombre))
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def Forwardprop(self, Image):
        # S'assurer que l'image est un vecteur colonne
        self.cache["A0"] = Image.reshape(-1, 1)

        for l in range(1, self.nbcouche + 1):
            Z = np.dot(self.poids[f'W{l}'], self.cache[f"A{l - 1}"]) + self.biais[f'b{l}']
            self.cache[f"Z{l}"] = Z
            if l == self.nbcouche:
                self.cache[f"A{l}"] = self.softmax(Z)
            else:
                self.cache[f"A{l}"] = self.activation_relu(Z)

        return np.argmax(self.cache[f"A{self.nbcouche}"])

    def Backwardprop(self, y_true_index):
        # Encodage One-hot
        y = np.zeros((self.nbneuroneparcouche[-1], 1))
        y[y_true_index] = 1

        # Erreur de sortie (Softmax + Cross-Entropy)
        dZ = self.cache[f"A{self.nbcouche}"] - y

        for l in range(self.nbcouche, 0, -1):
            dW = np.dot(dZ, self.cache[f"A{l - 1}"].T)
            db = dZ

            # Calcul du dZ pour la couche précédente AVANT de mettre à jour les poids actuels
            if l > 1:
                dZ = np.dot(self.poids[f'W{l}'].T, dZ) * self.d_relu(self.cache[f"Z{l - 1}"])

            # Mise à jour
            self.poids[f'W{l}'] -= self.tauxapp * dW
            self.biais[f'b{l}'] -= self.tauxapp * db

    # ... (Le reste de tes fonctions training et test sont correctes)

    def save_csv(self, filename="poids.csv"):
        with open(filename, "w") as file:
            for l in range(1, self.nbcouche + 1):
                file.write(f"# Couche W{l} shape={self.poids[f'W{l}'].shape}\n")
                np.savetxt(file, self.poids[f'W{l}'].flatten(), delimiter=",")
                file.write(f"# Biais b{l} shape={self.biais[f'b{l}'].shape}\n")
                np.savetxt(file, self.biais[f'b{l}'].flatten(), delimiter=",")
        print("Sauvegarde terminée.")

    def training(self, df):
        data = df.values
        tauxreussite = 0
        nb_images = len(data)
        for i in range(nb_images):
            y_true = int(data[i, 0])
            image = (data[i, 1:].reshape(-1, 1) / 255.0)
            prediction = self.Forwardprop(image)
            if prediction == y_true:
                tauxreussite += 1
            self.Backwardprop(y_true)
            if i % 10000 == 0 and i > 0:
                print(f"Image {i}/{nb_images} - Précision actuelle: {(tauxreussite / i) * 100:.2f}%")
        self.save_csv("poids.csv")

    def test(self,df):
        data = df.values
        tauxreussite = 0
        for i in range(len(data)):
            y_true = int(data[i, 0])
            image = (data[i, 1:].reshape(-1, 1) / 255.0)
            prediction = self.Forwardprop(image)
            if prediction == y_true:
                tauxreussite += 1
            if i % 10000 == 0 and i > 0:
                print(f"Image {i} - Précision: {(tauxreussite / i) * 100:.2f}%")


    def predict(self, liste_images):
        mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        resultat = ""
        for img in liste_images:
            image_flat = img.flatten().reshape(-1, 1).astype(float)
            if np.max(image_flat) > 1.0:
                image_flat /= 255.0
            prediction_index = self.Forwardprop(image_flat)
            resultat += mapping[prediction_index]
        return resultat

