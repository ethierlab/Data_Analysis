from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import wget
import scipy.fft as fft

# copier la joconde dans votre dossier Jupyter
filename = wget.download('https://pax.ulaval.ca/media/notebook/joconde.jpg')

# lire l'image
image = np.array(Image.open(filename).convert('L'))

# afficher l'image
plt.imshow(image, cmap='gray')
plt.show()

def afficher_tf(tf):
    """ Affiche le spectre d'une image

    Args :
    tf -- la transformée de Fourier d'une image
    """
    tf[tf == 0] = 1e-9
    magnitude_spectrum = 20*np.log(np.abs(tf))
    
    plt.imshow(magnitude_spectrum.astype(np.uint8))
    

tf = fft.fft2(image)

y1=int(np.floor((tf.shape[0])/2) - 40)
y2=int(np.floor((tf.shape[0])/2) + 40)
x1=int(np.floor((tf.shape[1])/2) - 40)
x2=int(np.floor((tf.shape[1])/2) + 40)

fft_shift = fft.fftshift(tf)
for y in range(y1,y2):
    for x in range(x1,x2):
        fft_shift[y,x] = 0

img_filtre = fft.ifft2(fft_shift)
img_filtre = np.absolute(img_filtre)
# afficher l'image
plt.imshow(image, cmap='gray')
plt.imshow(img_filtre, cmap='gray')
plt.show()