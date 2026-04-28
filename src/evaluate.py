from ultralytics import YOLO
import cv2
import os
import numpy as np
from pyzbar import pyzbar
import pandas as pd

clase = {0: "2020", 1: "2021", 2: "2022", 3: "2023", 4: "2024", 5: "2025", 6: "sbn"}
dataset_inventario = []


def crop_oriented_bbox(image, pts, cls_obj, n):
    # 1. Convertir puntos a float32 y darles forma de (4, 2)
    pts = np.array(pts, dtype="float32").reshape(4, 2)

    # 2. Calcular el ancho y alto del nuevo recorte
    # Distancia entre puntos (Euclidiana)
    width_a = np.sqrt(((pts[2][0] - pts[3][0]) ** 2) + ((pts[2][1] - pts[3][1]) ** 2))
    width_b = np.sqrt(((pts[1][0] - pts[0][0]) ** 2) + ((pts[1][1] - pts[0][1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((pts[1][0] - pts[2][0]) ** 2) + ((pts[1][1] - pts[2][1]) ** 2))
    height_b = np.sqrt(((pts[0][0] - pts[3][0]) ** 2) + ((pts[0][1] - pts[3][1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # 3. Definir los puntos de destino para la imagen "aplanada"
    dst_pts = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    # 4. Calcular la matriz de transformación y aplicar el warp
    matrix = cv2.getPerspectiveTransform(pts, dst_pts)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    img_flip = cv2.flip(warped, 1)
    f_clase = clase[int(cls_obj)]
    filename = f"recorte_{f_clase}_{n}.jpg"
    final_path = os.path.join(folder_path, filename)
    cv2.imwrite(final_path, img_flip)

    datos = read_barcode(final_path)
    for d in datos:
        #print(f"Contenido: {d['data']} | Inventario:{f_clase} | Tipo: {d['type']}")
        dataset_inventario.append(
            {
                "Archivo_Original": file_name,
                "Codigo_Texto": d["data"],
                "Clase_Inventario": f_clase,
                "Tipo_Codigo": d["type"],
                "Numero_Recorte": n,
            }
        )


def read_barcode(image_path):
    # 1. Cargar la imagen recortada
    img = cv2.imread(image_path)
    if img is None:
        return None

    # 2. Buscar y decodificar códigos de barras
    # PyZbar detecta automáticamente el tipo (Code128, EAN, QR, etc.)
    barcodes = pyzbar.decode(img)

    results = []
    for barcode in barcodes:
        # Extraer los datos en formato string
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type
        results.append({"data": barcode_data, "type": barcode_type})

    return results


trained_model = YOLO("runs/obb/etiquetas_obb-3/weights/best.pt")
results = trained_model("../dataset/test/*.jpg")

for result in results:
    full_path = result.path
    file_name = os.path.basename(full_path)
    # print(f"Imagen {file_name}")
    # print("-------------------")
    folder_path = os.path.join("../output_predicciones", file_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Carpeta creada: {folder_path}")
    image = result.orig_img

    cls_detected = result.obb.cls
    tensor_points = result.obb.xyxyxyxy
    n = 1
    for tensor_points_obj, cls_obj in zip(tensor_points, cls_detected):
        # Convertimos el tensor en numpy
        points = tensor_points_obj.cpu().numpy().squeeze()
        # Rendimensionamos y convertimos a enteros
        points = points.reshape((4, 2)).astype(np.int32)

        wraper = crop_oriented_bbox(image, points, cls_obj, n)
        n += 1


if dataset_inventario:
    df = pd.DataFrame(dataset_inventario)
    nombre_csv = "reporte_final_inventario.csv"
    
    # Exportamos a CSV (usamos utf-8-sig para que Excel lo abra sin errores de caracteres)
    df.to_csv(nombre_csv, index=False, encoding='utf-8-sig')
    print(f"\n¡Proceso completado! Dataset guardado como: {nombre_csv}")
else:
    print("No se encontraron códigos de barras para guardar.")