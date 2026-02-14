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
    resultat = np.zeros((final_hauteur, final_largueur))
    orig_hauteur, orig_largueur = image.shape
    proportion_hauteur = orig_hauteur / final_hauteur
    proportion_largueur = orig_largueur / final_largueur

    for i in range(final_hauteur):
        for j in range(final_largueur):
            orig_x = int(i*proportion_hauteur)
            orig_y = int(j*proportion_largueur)

            orig_x = min(orig_x, orig_hauteur - 1) #Securité car sinon crash de index error
            orig_y = min(orig_y, orig_largueur - 1)

            resultat[i,j] = image[orig_x, orig_y]
    return resultat

def normaliser(image):
    taille_finale = 28
    taille_interieur = 20
    h, l = image.shape
    max_dim = max(h, l)
    cuadrado = np.zeros((max_dim, max_dim))
    centre_h = (max_dim -h)//2
    centre_l = (max_dim - l)//2
    cuadrado[centre_h:centre_h+h, centre_l:centre_l+l] = image
    img_20x20 = redimensionner(cuadrado, taille_interieur, taille_interieur)

    resultat = np.zeros((taille_finale, taille_finale))
    marge = (taille_finale-taille_interieur)//2
    resultat[marge:marge+taille_interieur, marge:marge+taille_interieur] = img_20x20
    return resultat


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

