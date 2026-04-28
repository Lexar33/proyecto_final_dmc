from fastapi import FastAPI
import uvicorn

# call the app
app = FastAPI(title="API",docs_url="/my-custom-docs")

# Root Endpoint
@app.get("/")
def root():
    return {"API": "Este es un modelo para predecir Default."}



if __name__ == "__main__":
    uvicorn.run("app:app", reload=True)