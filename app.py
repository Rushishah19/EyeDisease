import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
from huggingface_hub import hf_hub_download

# === Load model from Hugging Face Hub ===
model_path = hf_hub_download(
    repo_id="RushiShah19/EyeDisease",
    filename="model.pth"
)

# === Load model (VGG16) with 3-class output ===
model = models.vgg16(weights=None)
model.classifier[6] = torch.nn.Linear(4096, 3)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# === Image preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Utility: Get color based on percentage ===
def get_color_code(value):
    if value <= 34:
        return "#00cc44"  # green
    elif value <= 70:
        return "#e6e600"  # yellow
    else:
        return "#ff3333"  # red

# === Streamlit UI ===
st.title("👁️ Eye Disease Detection")
st.write("Upload a retina image to get predictions for **Normal**, **Cataract**, and **Diabetic Retinopathy**")

uploaded_file = st.file_uploader("Choose a retina image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)[0] * 100

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("📊 Prediction Results:")

        labels = ["Normal", "Diabetic Retinopathy", "Cataract"]
        for i, label in enumerate(labels):
            value = probs[i].item()
            color = get_color_code(value)

            # Colored text label
            st.markdown(
                f"**{label} Probability:** <span style='color:{color}'>{value:.2f}%</span>",
                unsafe_allow_html=True
            )

            # Line scale (custom progress bar)
            st.markdown(f"""
                <div style="background-color: #e0e0e0; border-radius: 10px; height: 20px; width: 100%;">
                    <div style="width: {value}%; background-color: {color}; height: 100%; border-radius: 10px;"></div>
                </div>
                <br>
            """, unsafe_allow_html=True)
