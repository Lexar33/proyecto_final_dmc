import torch
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import imutils
import numpy as np

#print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

def train_model():
    model = YOLO('yolov8n-obb.pt')
    model.to('cuda')
    results= model.train(
        data='data.yaml',      # Ruta al archivo configurado arriba
        epochs=25,                     # Número de vueltas al dataset
        #imgsz=640,                    # Tamaño de imagen
        batch=12,                      # Ajustar según la memoria de tu GPU
        name='etiquetas_obb',
        #device=0,                     # Activa la GPU
        # Parámetros clave para Transfer Learning:
        freeze=10,                     # Congela las primeras 10 capas (capas base de visión)
        lr0=0.01,                      # Tasa de aprendizaje inicial
        augment=True                   # Activa aumentación de datos (giros, brillo, etc.)   
    )

if __name__ == '__main__':
    train_model()