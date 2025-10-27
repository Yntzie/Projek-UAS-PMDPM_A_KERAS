import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

IMG_SIZE = (160, 160)  # samakan dgn ukuran training kamu
CLASS_NAMES = ['bika ambon', 'kerak telor', 'papeda', 'plecing kangkung']

MODEL_PATH = "models/BestModel_CustomCNN_A_KERAS.tflite"  # <- hasil save ulang bersih

# Load model yang udah bersih
model = keras.models.load_model(MODEL_PATH, compile=False)

def predict_image(file_obj):
    img = Image.open(file_obj).convert("RGB")
    img = img.resize(IMG_SIZE)

    arr = keras.utils.img_to_array(img)
    arr = arr / 255.0  # harus sama seperti training
    arr = np.expand_dims(arr, axis=0)  # shape (1,H,W,3)

    preds = model.predict(arr)
    probs = tf.nn.softmax(preds[0]).numpy()

    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], probs

st.title("Klasifikasi Makanan Tradisional Indonesia 🍽")

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
