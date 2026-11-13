from fastapi import FastAPI, File, UploadFile
import torch
from torchvision import transforms
from PIL import Image
import io
from src.models.pretrained_models import MetalDefectEfficientNet

app = FastAPI(title="Metal Defect Classifier API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MetalDefectEfficientNet(num_classes=10) 
model.load_state_dict(torch.load("artifacts/efficientnet_best.pth", map_location=device))
model.to(device)
model.eval()

inference_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLASS_NAMES = ["crease", "crescent_gap", "crescent_gap", "oil_spot", "punching_hole", "rolled_pit", "silk_spot", "waist folding", "water_spot", "welding_line"] 

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    tensor = inference_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    return {
        "defect_type": CLASS_NAMES[predicted.item()],
        "confidence": float(confidence.item())
    }