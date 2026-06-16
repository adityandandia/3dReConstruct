from fastapi import FastAPI
import uvicorn
from api_routes import router

app = FastAPI(title="3D Scanner Backend")
app.include_router(router)

if __name__ == "__main__":
    # Ensure you are running this from the backend folder
    uvicorn.run(app, host="0.0.0.0", port=8000)
