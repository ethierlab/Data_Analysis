import wget
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.fftpack

if not Path('joconde.jpg').is_file():
    # télécharger une copie locale de la joconde
    wget.download('https://pax.ulaval.ca/media/notebook/joconde.jpg')

im = np.array(Image.open("joconde.jpg").convert('L'), dtype=np.int16)
im = im - 128 # on centre entre -128 et 127

Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]])

plt.imshow(im)
plt.show()

def afficher_transformée(img_dct, img_dct_quant, h, w):
    """
    Permet d'afficher la transformée en cosinus
    Args:
       img_dct (matrice numpy): matrice de la transformée en cosinus à 4 dimensions AVANT la quantification
       img_dct_quant (matrice numpy): matrice de la transformée en cosinus à 4 dimensions APRES la quantification
       h (int): hauteur de l'image
       w (int): largeur de l'image
    """

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Image reconstituée avec les transformées en cosinus')
    ax1.imshow(img_dct.reshape(h, w), vmax=np.max(img_dct)*0.01,vmin=0)
    ax1.set_title("Avant la quantification")
    ax2.imshow(img_dct_quant.reshape(h, w), vmax=np.max(img_dct_quant)*0.01,vmin=0)
    ax2.set_title("Après la quantification")

    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Bloc 8x8 de la transformée en cosinus')
    ax1.imshow(img_dct[29, :, 20, :], vmax=np.max(img_dct)*0.01,vmin=0)
    ax1.set_title("Avant la quantification")
    ax2.imshow(img_dct_quant[29, :, 20, :], vmax=np.max(img_dct_quant)*0.01,vmin=0)
    ax2.set_title("Après la quantification")

    plt.show()

m, n = im.shape
h = int(m - (m % 8))
w = int(n - (n % 8))
im_cut = im[0:h,0:w]
im_divided = im_cut.reshape(int(h/8), 8, int(w/8), 8)
im_dct =  scipy.fftpack.dctn(im_divided, axes=(1,3),norm='ortho')
im_dct_quant = np.rint(im_dct / Q[np.newaxis,:, np.newaxis, :])

afficher_transformée(im_dct, im_dct_quant, h, w)