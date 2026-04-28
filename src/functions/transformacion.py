#!pip install pillow pillow-heif 
import os
from PIL import Image
from pillow_heif import register_heif_opener
from pathlib import Path

# Registrar el decodificador de HEIF en Pillow
register_heif_opener()

def convertir_heic(ruta_entrada, formato_salida="JPEG"):
    """
    Convierte un archivo HEIC a JPG o PNG.
    formato_salida: "JPEG" o "PNG"
    """
    # Generar el nombre de salida
    nombre_base = os.path.splitext(ruta_entrada)[0]
    extension = ".jpg" if formato_salida == "JPEG" else ".png"
    ruta_salida = nombre_base + extension

    try:
        with Image.open(ruta_entrada) as img:
            # Si es JPG, es mejor convertir a RGB (quita la transparencia si existe)
            if formato_salida == "JPEG":
                img = img.convert("RGB")
            
            img.save(ruta_salida, formato_salida, quality=95)
            print(f"✅ Convertido: {ruta_salida}")
            
    except Exception as e:
        print(f"❌ Error al convertir {ruta_entrada}: {e}")


def rename():
    number="0"
    script_dir = Path(__file__).parent
    for item in script_dir.iterdir():
        if item.is_file():
            number=f"{int(number) + 1}"
            new_name=number+".jpg"
            os.rename(item.name,new_name)

    print("finalizado renombrar")

# Ejemplo: Convertir todos los HEIC en la carpeta actual

directorio = "./dataset"

for archivo in os.listdir(directorio):
    if archivo.lower().endswith(".heic"):
        convertir_heic(os.path.join(directorio, archivo), "JPEG")

rename()