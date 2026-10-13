
import numpy as np
import cv2

quantidades = [60, 82, 114, 150, 205]

for qtd in quantidades:
    img = cv2.imread(f"{qtd}.bmp", cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Erro ao carregar {qtd}.bmp")
        continue
    
    img = cv2.imread(f"{qtd}.bmp", cv2.IMREAD_GRAYSCALE)

    img = img.astype(np.float32) / 255.0
    
    # Aplicar filtro de média
    blur = cv2.blur(img, (65, 65))

    # Calcular a diferença
    dif = img - blur

    # Converter para uint8, descartando diferenças negativas
    dif_8 = np.clip(dif * 255, 0, 255).astype(np.uint8)

    # Aplicar Otsu para encontrar o melhor limiar
    limiar, buffer = cv2.threshold( dif_8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Imagem {qtd}.bmp - Limiar de Otsu: {limiar}")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    out = cv2.morphologyEx(buffer, cv2.MORPH_OPEN, kernel)
    
    # Exibir resultados
    cv2.imshow(f"Original - {qtd}", img)
    cv2.imshow(f"Blur - {qtd}", blur)
    cv2.imshow(f"Diferenca - {qtd}", dif_8)
    cv2.imshow(f"Resultado Otsu - {qtd}", out)

    # Criar cópia da imagem binária para o flood fill
    img_fill = out.copy()

    n_linhas, n_colunas = img_fill.shape
    contagem = 0

    for i in range(n_linhas):
        for j in range(n_colunas):
            if img_fill[i, j] == 255:
                cv2.floodFill(img_fill, None, (j, i), 128)
                contagem += 1

    print(f"Imagem {qtd}.bmp: {contagem} objetos")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
