import numpy as np

class IA:
    def __init__(self, dim_couche, tauxapp):
        self.nbcouche = len(dim_couche) - 1
        self.nbneuroneparcouche = dim_couche
        self.tauxapp = tauxapp
        self.poids = {}
        self.biais = {}
        self.cache = {}
        self.gradients = {}
        self.initialisation_parametres()

    def fonction_activation(self, x):
        # Modification : Utilisation de ReLU au lieu de la Sigmoïde
        return np.maximum(0, x)

    def softmax(self, x):
        # Soustraction du max pour la stabilité numérique
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def d_activation(self, a):
        # Modification : Dérivée de ReLU (1 si l'activation était > 0, sinon 0)
        return (a > 0).astype(float)

    def initialisation_parametres(self):
        for l in range(1, len(self.nbneuroneparcouche)):
            # Modification : Initialisation de He (recommandée pour ReLU)
            std = np.sqrt(2.0 / self.nbneuroneparcouche[l - 1])
            self.poids[f'W{l}'] = np.random.randn(self.nbneuroneparcouche[l], self.nbneuroneparcouche[l - 1]) * std
            # Initialiser les biais à 0 ou à une petite valeur positive (0.01)
            self.biais[f'b{l}'] = np.zeros((self.nbneuroneparcouche[l], 1))

    def Forwardprop(self, Image):
        self.cache["A0"] = Image.reshape(-1, 1)

        for l in range(1, self.nbcouche + 1):
            W = self.poids[f'W{l}']
            b = self.biais[f'b{l}']
            A_prev = self.cache[f"A{l - 1}"]
            Z = np.dot(W, A_prev) + b
            self.cache[f"Z{l}"] = Z

            if l == self.nbcouche:
                A = self.softmax(Z)
            else:
                A = self.fonction_activation(Z)

            self.cache[f"A{l}"] = A

        return np.argmax(self.cache[f"A{self.nbcouche}"])

    def Backwardprop(self, attendu):
        y = np.zeros((self.nbneuroneparcouche[-1], 1))
        y[attendu] = 1

        # Gradient de la couche de sortie (Softmax + Cross-Entropy)
        dZ = self.cache[f"A{self.nbcouche}"] - y

        for l in range(self.nbcouche, 0, -1):
            A_prev = self.cache[f"A{l - 1}"]
            self.gradients[f"dW{l}"] = np.dot(dZ, A_prev.T)
            self.gradients[f"db{l}"] = dZ

            if l > 1:
                W = self.poids[f'W{l}']
                dZ = np.dot(W.T, dZ) * self.d_activation(self.cache[f"A{l - 1}"])

        self.update_parameters()

    def update_parameters(self):
        for l in range(1, self.nbcouche + 1):
            self.poids[f'W{l}'] -= self.tauxapp * self.gradients[f'dW{l}']
            self.biais[f'b{l}'] -= self.tauxapp * self.gradients[f'db{l}']

    def training(self, df):
        data = df.values
        nb_images = len(data)
        print(f"Début de l'entraînement sur {nb_images} images...")

        for i in range(nb_images):
            y_true = int(data[i, 0])
            image = data[i, 1:].reshape(-1, 1) / 255.0
            self.Forwardprop(image)
            self.Backwardprop(y_true)
            if i % 5000 == 0:
                print(f"Images traitées : {i}/{nb_images} ({(i / nb_images) * 100:.2f}%)")

    def predict(self, liste_images):
        # Mappe les index EMNIST vers des caractères lisibles
        mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        resultat = ""
        for img in liste_images:
            # S'assurer que l'image est aplatie et normalisée
            image_flat = img.flatten().reshape(-1, 1).astype(float)
            if np.max(image_flat) > 1.0:
                image_flat /= 255.0

            prediction_index = self.Forwardprop(image_flat)
            resultat += mapping[prediction_index]
        return resultat
