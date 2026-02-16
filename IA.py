import numpy as np
from pandas import read_csv

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
        return 1/(1 + np.exp(-x))
    def softmax(self, x):
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=0)

    def d_activation(self, a):
        return a*(1-a)

    def initialisation_parametres(self):
        for l in range(1, len(self.nbneuroneparcouche)):
            std = np.sqrt(2.0 / self.nbneuroneparcouche[l - 1])
            self.poids[f'W{l}'] = np.random.randn(self.nbneuroneparcouche[l], self.nbneuroneparcouche[l - 1]) * std
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
        tauxreussite = 0
        for i in range(nb_images):
            y_true = int(data[i, 0])
            image = data[i, 1:].reshape(-1, 1) / 255.0
            prediction = self.Forwardprop(image)
            if prediction == y_true:
                tauxreussite += 1
            self.Backwardprop(y_true)
            if i % 5000 == 0 and i > 0:
                score = (tauxreussite / (i + 1)) * 100
                print(f"Image {i}/{nb_images} - Taux de réussite actuel : {score:.2f}%")

    def predict(self, liste_images):
        mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        resultat = ""
        for img in liste_images:
            image_flat = img.flatten().reshape(-1, 1).astype(float)
            # Normalisation automatique si nécessaire
            if np.max(image_flat) > 1.0:
                image_flat /= 255.0

            prediction_index = self.Forwardprop(image_flat)
            resultat += mapping[prediction_index]
        return resultat
