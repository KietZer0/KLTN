from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms
from transformers import ViTForImageClassification
import sys

# Initialize FastAPI
app = FastAPI()

label_names = {
    0: "111111",
    1: "111112",
    2: "111113",
    3: "111115",
    4: "111118"
    # Add more as needed
}

# Định nghĩa hàm để load mô hình
def load_model(model_path, num_classes):
    model = ViTForImageClassification.from_pretrained("KietZer0/ViT_LFW_Model4")
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Định nghĩa hàm để dự đoán từ ảnh
def predict_image(model, image, transform):
    image = image.convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.logits, 1)
    return predicted.item()

# Định nghĩa transform để chuyển đổi hình ảnh
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Đường dẫn đến mô hình đã lưu
model_path = "model_round_2.pth"  # Thay bằng đường dẫn thực tế của mô hình đã lưu
num_classes = 5  # Số lớp của mô hình
model = load_model(model_path, num_classes)

# Endpoint nhận ảnh từ client và trả về dự đoán
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        predicted_class = predict_image(model, image, transform)
        return {"predicted_class": label_names[predicted_class]}
    except Exception as e:
        return {"error": str(e)}

# Chạy ứng dụng FastAPI với địa chỉ IP từ dòng lệnh
if __name__ == "__main__":
    import uvicorn
    
    # Lấy địa chỉ IP từ dòng lệnh nếu được cung cấp, mặc định là 127.0.0.1
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    
    uvicorn.run(app, host=host, port=8000)
