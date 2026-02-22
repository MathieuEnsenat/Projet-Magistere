from pdb import post_mortem
import numpy as np
import matplotlib.pyplot as plt
from numpy.f2py.crackfortran import endifs


def generar_imagen_falsa():
    alto = 60
    ancho = 80
    img = np.zeros((alto, ancho), dtype=int)
    img[5:25, 5:10] = 1
    img[20:25, 5:15] = 1
    img[5:25, 25:40] = 1
    img[10:20, 30:35] = 0
    img[35:55, 10:15] = 1
    img[35:40, 30:50] = 1
    img[35:55, 38:42] = 1

    return img
img = generar_imagen_falsa()
plt.imshow(img, cmap='gray')

def intervalles(liste):
    if not liste:
        return []
    intervalos = []
    start = liste[0]

    for k in range (1, len(liste)):
        if liste[k] != liste[k-1]+1:
            fin = liste[k-1]
            intervalos.append((start,fin))
            start = liste[k]
    intervalos.append((start,liste[-1]))
    return intervalos

def scanner_horizontal(image):
    res_x, res_y = image.shape
    somme = np.sum(image, axis=1)
    lignes = []
    for i in range(res_x):
        if somme[i] != 0:
            lignes.append(i)
    resultat = intervalles(lignes)
    return resultat

def scanner_vertical(image):
    res_x, res_y = image.shape
    somme = np.sum(image, axis=0)
    colonnes = []
    for i in range(res_y):
        if somme[i] != 0:
            colonnes.append(i)
    resultat = intervalles(colonnes)
    return resultat

def cadrage(image):
    caracteres = []
    lignes_avec_caracteres = scanner_horizontal(image)

    for (y_min, y_max) in lignes_avec_caracteres:
        image_ligne = image[y_min : y_max+1, :]
        colonnes_avec_caracteres = scanner_vertical(image_ligne)
        for (x_min, x_max) in colonnes_avec_caracteres:
            caractere = image_ligne[:, x_min : x_max+1]
            caracteres.append(caractere)
    return caracteres

def cadrage2(image):
    mots = []
    lignes_avec_caracteres = scanner_horizontal(image)

    for (y_min, y_max) in lignes_avec_caracteres:
        image_ligne = image[y_min : y_max+1, :]
        intervalles_lettres = scanner_vertical(image_ligne)
        espaces = []
        for k in range(len(intervalles_lettres)-1):
            fin_actuel = intervalles_lettres[k][1]
            debut_suivant = intervalles_lettres[k+1][0]
            distance = debut_suivant - fin_actuel
            espaces.append(distance)
#Calcul de l'espace entre les mots
        seuil_espace = 0
        if len(espaces) > 0:
            moyenne_espace = np.mean(espaces)
            seuil_espace = 2 * moyenne_espace
            seuil_espace = max(seuil_espace, 3)
#Construction des mots
        mot_actuel = []
        for k, (x_min, x_max) in enumerate(intervalles_lettres):
            lettre = image_ligne[:, x_min : x_max+1]
            mot_actuel.append(lettre)
            if k<len(espaces):
                distance_apres = espaces[k]
                if distance_apres > seuil_espace:
                    mots.append(mot_actuel)
                    mot_actuel = []
        if mot_actuel:
            mots.append(mot_actuel)
    return mots

def redimensionner(image, final_hauteur, final_largueur):
    """
    Réduit la taille d'une image d'un facteur donné en lissant les pixels.
    Ex: Image 28x28 avec factor=2 -> Image 14x14
    """
    h, w = image.shape
    new_h, new_w = 26, 26
    factor = h // new_h
    trimmed_img = image[:new_h * factor, :new_w * factor]
    
    # 1. On "découpe" virtuellement l'image en blocs de (factor x factor)
    # On crée une structure à 4 dimensions : (nouvelle_h, factor, nouvelle_w, factor)
    view = trimmed_img.reshape(new_h, factor, new_w, factor)
    
    # 2. On calcule la moyenne sur les axes des blocs (axes 1 et 3)
    resized = view.mean(axis=(1, 3))
    
    return resized


def normaliser(image):
    #suppression des lignes et colonnes vides
    lignes = np.any(image, axis=1)
    cols = np.any(image, axis=0)
    
    ymin, ymax = np.where(lignes)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    img_reduite = image[ymin:ymax+1, xmin:xmax+1]
    
    #on redimensionne le caractère pour qu'il rentre dans un carré de 20x20:
    #on garde le ratio d'aspect
    h, l = img_reduite.shape
    if h > l:
        new_h = 20
        new_l = int(l * (20 / h))
    else:
        new_l = 20
        new_h = int(h * (20 / l))
        
    #redimensionnement
    if new_l == 0: 
        new_l = 1
    if new_h == 0: 
        new_h = 1
    indices_y = np.linspace(0, h - 1, new_h).astype(int)
    indices_x = np.linspace(0, l - 1, new_l).astype(int)
    img_redim = img_reduite[indices_y, :]
    img_redim = img_redim[:, indices_x]
    
    #on crée un cadre noir 28x28 et on met le caractère au centre
    res = np.zeros((28, 28))
    y_debut = (28 - new_h) // 2 
    x_debut = (28 - new_l) // 2
    res[y_debut:y_debut+new_h, x_debut:x_debut+new_l] = img_redim
    
    return flouter(res)

def post_decoupage(imagen_letra):
    h, l = imagen_letra.shape
    if h == 0 or l == 0:
        return [imagen_letra]

    ratio = l / h
    if ratio < 1.05:
        return [imagen_letra]

    histograma = np.sum(imagen_letra, axis=0)
    start = int(l * 0.25)
    end = int(l * 0.75)
    if start >= end:
        return [imagen_letra]

    zona_central = histograma[start:end]
    min_valor = np.min(zona_central)
    indice_relatif = np.argmin(zona_central)
    indice = start + indice_relatif
    max_tinta = np.max(histograma)

    if ratio > 1.35: #decoupage agresif
        umbral_tinta = 0.60
        requerir_centro = False
    else: #decoupage mais on fait gaffe de ne pas deconner
        umbral_tinta = 0.35
        requerir_centro = True

    condicion_valle = min_valor < (max_tinta * umbral_tinta)
    condicion_fina = min_valor < (h * 0.25)
    condicion_centro = True
    posicion_corte_relativa = indice / l

    if requerir_centro:
        if posicion_corte_relativa < 0.40 or posicion_corte_relativa > 0.60:
            condicion_centro = False
    if condicion_valle and condicion_fina and condicion_centro:
        letra_izq = imagen_letra[:, :indice]
        letra_der = imagen_letra[:, indice:]

        if letra_izq.shape[1] < (l * 0.15) or letra_der.shape[1] < (l * 0.15): #eviter les taches
            return [imagen_letra]
        return post_decoupage(letra_izq) + post_decoupage(letra_der)
    return [imagen_letra]

def pre_normalisation(liste_mots_salle):
    liste_mots_propre = []
    for mot in liste_mots_salle:
        nouvelle_mot = []
        for lettre in mot:
            correction = post_decoupage(lettre)
            nouvelle_mot.extend(correction)
        liste_mots_propre.append(nouvelle_mot)
    return liste_mots_propre

def flouter(img):
    #noyau gaussien
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]) / 16
    
    #convolution
    h, l = img.shape
    img_floutee = np.copy(img)
    for y in range(1, h-1):
        for x in range(1, l-1):
            region = img[y-1:y+2, x-1:x+2]
            img_floutee[y, x] = np.sum(region * kernel)
    return img_floutee

