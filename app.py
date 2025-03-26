import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Load pretrained ResNet50 model and modify final layer
class EyeDiseaseModel(nn.Module):
    def __init__(self):
        super(EyeDiseaseModel, self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 2),
            nn.Sigmoid()  # Output two probabilities (0 to 1)
        )

    def forward(self, x):
        return self.model(x)

# Load the model
model = EyeDiseaseModel()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Streamlit app
st.title("👁️ Eye Disease Detection")
st.write("Upload a retina image to get predictions for **Cataract** and **Diabetic Retinopathy**")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        cataract_prob = output[0][0].item() * 100
        dr_prob = output[0][1].item() * 100

    st.subheader("📊 Prediction Results:")
    st.write(f"**Cataract Probability:** {cataract_prob:.2f}%")
    st.write(f"**Diabetic Retinopathy Probability:** {dr_prob:.2f}%")
