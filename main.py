import numpy as np
import cv2 as cv

quantidades = [60, 82, 114, 150, 205]

for qtd in quantidades:
    img = cv.imread(f"{qtd}.bmp", cv.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Erro ao carregar {qtd}.bmp")
        continue
    
    img = img.astype(np.float32) / 255.0
    blur = cv.blur(img, (65, 65))
    
    dif = img - blur
    dif_8 = np.clip(dif * 255, 0, 255).astype(np.uint8)

    limiar, buffer = cv.threshold(dif_8, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    binary = cv.morphologyEx(buffer, cv.MORPH_OPEN, kernel)
    cv.imshow(f"Binaria - {qtd}", binary)
    
    dist = cv.distanceTransform(binary, cv.DIST_L2, 5)
    cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
    cv.imshow(f"Distancia do centro - {qtd}", dist)
    
    cv.waitKey(0)
    cv.destroyAllWindows()
