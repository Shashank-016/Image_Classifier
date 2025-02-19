from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Serve static files (CSS, images if needed)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Assuming binary classification (e.g., Cat vs Dog)
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# Set up Jinja2 for templating
templates = Jinja2Templates(directory="templates")

# Image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Serve HTML page
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Image prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    label = "Dog" if prediction == 1 else "Cat"  # Change based on class mapping
    return {"filename": file.filename, "prediction": label}
