import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove_background_grabcut(image_path):
    # Carrega a imagem
    image = cv2.imread(image_path)
    mask = np.zeros(image.shape[:2], np.uint8)  # Cria uma máscara vazia

    # Define retângulo inicial para o GrabCut (área de interesse)
    rect = (110, 110, image.shape[1] - 110, image.shape[0] - 155)

    # Aplica o algoritmo GrabCut
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Cria uma máscara binária onde o fundo é 0, primeiro plano é 1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Aplica a máscara na imagem original
    result = image * mask2[:, :, np.newaxis]

    # Mostra a imagem original e a imagem com o fundo removido
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original')
    plt.subplot(122), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), plt.title('Removed Background')
    plt.tight_layout(), plt.show()

# Caminho da imagem que deseja processar
image_path = 'dog.jpeg'

# Chama a função para remover o fundo da imagem usando o GrabCut
remove_background_grabcut(image_path)
