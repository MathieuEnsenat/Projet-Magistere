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
            self.poids[f"W{l}"] = np.random.randn(self.nbneuroneparcouche[l], self.nbneuroneparcouche[l - 1]) * np.sqrt(
                2. / self.nbneuroneparcouche[l - 1])
            self.biais[f"b{l}"] = np.zeros((self.nbneuroneparcouche[l], 1))

    def load_csv(self, filename="data/poids.csv"):
        try:
            # On charge les données en ignorant les lignes de commentaires si elles existent
            data = np.loadtxt(filename, delimiter=",", comments="#")
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

    def activation_relu(self, x):
        return np.maximum(0, x)

    def d_relu(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        # Soustraction du max pour la stabilité numérique (évite exp(large_nombre))
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def Forwardprop(self, x):
        self.cache["A0"] = x.reshape(-1, 1)

        for l in range(1, self.nbcouche+1):
            Z = np.dot(self.poids[f'W{l}'], self.cache[f"A{l - 1}"]) + self.biais[f'b{l}']
            self.cache[f"Z{l}"] = Z
            if l == self.nbcouche:
                self.cache[f"A{l}"] = self.softmax(Z)
            else:
                self.cache[f"A{l}"] = self.activation_relu(Z)

        return np.argmax(self.cache[f"A{self.nbcouche}"])

    def Backwardprop(self, y_true_index, beta=0.9):
        # Encodage One-hot
        y = np.zeros((self.nbneuroneparcouche[-1], 1))
        y[y_true_index] = 1
        dZ = self.cache[f"A{self.nbcouche}"] - y
        for l in range(self.nbcouche, 0, -1):
            dW = np.dot(dZ, self.cache[f"A{l - 1}"].T)
            db = dZ
            if l > 1:
                dZ = np.dot(self.poids[f"W{l}"].T, dZ) * self.d_relu(self.cache[f"Z{l - 1}"])
            self.poids[f"W{l}"] -= self.tauxapp * dW
            self.biais[f"b{l}"] -= self.tauxapp * db

    def save_csv(self, filename="poids.csv"):
        with open(filename, "w") as file:
            for l in range(1, self.nbcouche + 1):
                file.write(f"# Couche W{l} shape={self.poids[f"W{l}"].shape}\n")
                np.savetxt(file, self.poids[f'W{l}'].flatten(), delimiter=",")
                file.write(f"# Biais b{l} shape={self.biais[f"b{l}"].shape}\n")
                np.savetxt(file, self.biais[f'b{l}'].flatten(), delimiter=",")
        print("Sauvegarde terminée.")

    def training(self, x_train, y_train, x_test, y_test, nb_iterations):
        for i in range(nb_iterations):
            for x, y in zip(x_train, y_train):
                self.Forwardprop(x)
                self.Backwardprop(y)
        
            tx_reussite = self.test(x_test, y_test)
            print(f"Itération {i+1}/{nb_iterations} - Précision actuelle: {tx_reussite:.2f}%")
        
        self.save_csv("poids.csv")

    def test(self, x_test, y_test):
        score = 0
        for (x, y) in zip(x_test, y_test):
            prediction = self.Forwardprop(x)
            if prediction == y:
                score += 1
        return score * 100 / len(x_test) #taux de réussite


    def predict(self, liste_images, mapping):
        resultat = ""
        for img in liste_images:
            image_flat = img.flatten().astype(float)
            if np.max(image_flat) > 1.0:
                image_flat /= 255.0
            prediction_index = self.Forwardprop(image_flat)
            resultat += mapping[prediction_index]
        return resultat
    

    

def load_emnist_csv(file_path):
    print(f"Chargement des données de '{file_path}'")
    df = pd.read_csv(file_path, header=None)
    df = np.array(df)    
    np.random.shuffle(df) #mélange des données
    
    y = df[:, 0] #labels
    x = df[:, 1:] #pixels des images
    
    #es images EMNIST sont pivotées à 90° et inversées dans le CSV
    #on les remet à l'endroit
    x = x.reshape(-1, 28, 28)
    x = np.transpose(x, (0, 2, 1)) #corrige la rotation
    x = x.reshape(-1, 784) #on ré-aplatit
    
    x = x / 255.0 #normalisation
    
    return x, y



if __name__ == "__main__":
    from traitement_image import importer_image, rgb_a_gris
    import matplotlib.pyplot as plt

    #importation des données d'entraînement
    #x_train, y_train = load_emnist_csv("Magistère/emnist-byclass-train.csv")
    #x_test, y_test = load_emnist_csv("Magistère/emnist-byclass-test.csv")

    #création du réseau 
    dim_couches = [784, 128, 47]
    eta = 0.01
    reseau = IA(dim_couches, eta)
 
    #phase d'entrainement
    #nb_iterations = 10
    #reseau.training(x_train, y_train, x_test, y_test, nb_iterations)
    
    reseau.load_csv("data/poids.csv")

    #test sur une image
    img = importer_image("data/test_FACILE.png")
    img = rgb_a_gris(img)
    img = 255 - img

    mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

    prediction = reseau.predict([img], mapping)
    plt.imshow(img, cmap="gray")
    plt.title(f"Caractère sur l'image selon l'IA : {prediction}")
    plt.show()