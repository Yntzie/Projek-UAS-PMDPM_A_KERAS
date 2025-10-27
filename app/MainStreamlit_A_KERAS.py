import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image

IMG_SIZE = (160, 160)
CLASS_NAMES = ['bika ambon', 'kerak telor', 'papeda', 'plecing kangkung']

MODEL_PATH = "models/cleaned_model_tfkeras.onnx"

# Load ONNX model
session = ort.InferenceSession(MODEL_PATH)

def predict_image(file_obj):
    img = Image.open(file_obj).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1,H,W,3)
    
    # Nama input dan output tergantung model (biasanya 'input_1')
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    preds = session.run([output_name], {input_name: arr})[0][0]
    probs = np.exp(preds) / np.sum(np.exp(preds))
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], probs

st.title("🍽 Klasifikasi Makanan Tradisional Indonesia (ONNX Version)")

uploads = st.file_uploader(
    "Upload gambar makanan (boleh banyak)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if st.button("Prediksi"):
    if not uploads:
        st.error("Silakan upload minimal satu gambar dulu.")
    else:
        for f in uploads:
            label, probs = predict_image(f)
            st.write(f"**File:** {f.name}")
            st.write(f"**Prediksi:** {label}")
            for cls_name, score in zip(CLASS_NAMES, probs):
                st.write(f"- {cls_name}: {score * 100:.2f}%")
            st.image(Image.open(f), caption=f.name, use_column_width=True)
            st.markdown("---")
