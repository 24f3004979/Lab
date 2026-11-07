import torch
import torchvision
import streamlit as st
from PIL import Image
from torchvision import transforms

# ---------------- CONFIG: edit these ----------------
MODEL_PATH = "/home/madhav/workstation/Lab/experiment/models/Satellite-Classifier-model.pkl"
CLASS_NAMES = ["cloudy", "desert", "green_area", "water"]  # must match label index order
device = "cuda" if torch.cuda.is_available() else "cpu"

# IMPORTANT: paste the exact same steps as your notebook's test_transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_model():
    # weights_only=False is needed if the file holds a whole pickled model.
    # Only do this for files you created yourself.
    obj = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    if isinstance(obj, torch.nn.Module):
        # Case 1: the full model was saved
        model = obj
    else:
        # Case 2: only a state_dict was saved, so rebuild the architecture first
        model = torchvision.models.efficientnet_b0(weights=None)
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(1280, len(CLASS_NAMES)),
        )
        model.load_state_dict(obj)

    return model.to(device).eval()


model = load_model()

# Simple UI with streamlit

st.title("Satellite Image Classifier")
st.write("Upload a satellite image and the model will predict its class.")

file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu()

    pred = int(probs.argmax())
    st.subheader(f"Prediction: {CLASS_NAMES[pred]}")
    st.write(f"Confidence: {probs[pred].item():.2%}")
    st.bar_chart({name: float(p) for name, p in zip(CLASS_NAMES, probs)})