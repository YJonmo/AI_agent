import io

from utils import (
    cifar_transforms,
    cifar_model,
    cifar_classes,
    pneumonia_transforms,
    pneumonia_model,
    pneumonia_classes
)


from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from torchvision import transforms
from PIL import Image
import torch

print('imported')

# Initialize FastAPI
app = FastAPI()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pneumonia_model = pneumonia_model.to(device)
cifar_model = cifar_model.to(device)

# Define preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/generic_classifier/")
async def cifar_classifier(file: UploadFile = File(...)):
    # Read and preprocess the image
    content = await file.read()
    image = Image.open(io.BytesIO(content))
    input_tensor = cifar_transforms(image).to(device)

    # Perform inference
    with torch.no_grad():
        output = cifar_model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted = cifar_classes[predicted]
    # Return the classification result
    return JSONResponse(content={"class_id": predicted})


@app.post("/pneumonia_classifier/")
async def pneumonia_classifier(file: UploadFile = File(...)):
    # Read and preprocess the image
    content = await file.read()
    image = Image.open(io.BytesIO(content))
    input_tensor = pneumonia_transforms(image).to(device)

    # Perform inference
    with torch.no_grad():
        output = pneumonia_model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted = pneumonia_classes[predicted]
    # Return the classification result
    return JSONResponse(content={"diagnostic result": predicted})


from fastapi.responses import HTMLResponse




# Home route with HTML form for uploading the image
@app.get("/", response_class=HTMLResponse)
async def home():
    content = """
    <html>
        <body>
            <form action="/generic_classifier/" method="post" enctype="multipart/form-data">
                <input name="file" type="file">
                <input type="submit">
            </form>
            <form action="/pneumonia_classifier/" method="post" enctype="multipart/form-data">
                <input name="file" type="file">
                <input type="submit">
            </form>
        </body>
    </html>
    """
    return content