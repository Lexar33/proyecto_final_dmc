from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
from pathlib import Path
from fastapi.staticfiles import StaticFiles
import csv
import os 
from fastapi import HTTPException
import pandas as pd
from src.evaluate import evaluate
import uuid
import shutil

# call the app
app = FastAPI(title="API")
app.mount("/output_predicciones", StaticFiles(directory="output_predicciones"), name="output_predicciones")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

CSV_FILE = Path("registros_archivos.csv")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("output_predicciones")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, limita esto a tu dominio
    allow_methods=["*"],
    allow_headers=["*"],
)


# Root Endpoint
@app.get("/")
def root():
    return {"API": "Este es un modelo para predecir Default."}


@app.get("/listar")
def get_all_files():
    if not CSV_FILE.exists():
        return []  # Devolvemos array vacío si no hay registros aún

    lista_archivos = []

    with open(CSV_FILE, mode="r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        for fila in reader:
            lista_archivos.append(fila)

    return lista_archivos

@app.get("/list-original/{image_id}")
async def list_original(image_id: str):
    clean_id = image_id.replace("'", "")
    file_path = UPLOAD_DIR / clean_id
    if not file_path.exists():
        return {"error": "La imagen original no existe"}

    # Devolvemos el archivo para que el navegador lo muestre
    return FileResponse(file_path)

@app.get("/list-prediction/{image_id}")
async def list_prediction(image_id: str):
        clean_id = image_id.replace("'", "")
        f_path = OUTPUT_DIR / clean_id / "reporte_final.csv"
        if not f_path.exists():
            raise HTTPException(status_code=404, detail="El archivo de reporte no existe para este ID")

        try:
            df = pd.read_csv(f_path, encoding="utf-8-sig")
            data = df.to_dict(orient="records")
            print(f_path)
            return data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al leer el CSV: {str(e)}")

@app.get("/list-images/{image_id}")
async def list_images(image_id: str):
    clean_id = image_id.replace("'", "")
    
    folder_path = OUTPUT_DIR / clean_id
    #print(folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        return []  # O lanza un 404 si prefieres

    # Listamos solo archivos de imagen
    archivos = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    return archivos


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    extension = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{extension}"
    file_path = UPLOAD_DIR / unique_filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # --- GUARDA EL CSV ---
    file_exists = CSV_FILE.exists()

    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        # Si el archivo es nuevo, podemos escribir una cabecera opcional
        if not file_exists:
            writer.writerow(["nombre_original", "uuid_name", "ruta"])

        # Agregamos el nuevo registro
        writer.writerow([file.filename, unique_filename, str(file_path)])
    # --------------------------------
    return {
        "message": "Archivo guardado con éxito",
        "original_name": file.filename,
        "uuid_name": unique_filename,
        "path": str(file_path),
    }


@app.post("/predict")
def predict(nombre_archivo: str = Body(..., embed=True)):
    print(nombre_archivo)
    df = evaluate("uploads/" + nombre_archivo, 0)
    import pandas as pd

    if isinstance(df, pd.DataFrame):
        data_json = df.to_dict(orient="records")

        return {"status": "success", "data": data_json}

    else:
        print(f"La evaluación falló o devolvió un código: {df}")
    return "ok"


if __name__ == "__main__":
    uvicorn.run("app:app", reload=True)
