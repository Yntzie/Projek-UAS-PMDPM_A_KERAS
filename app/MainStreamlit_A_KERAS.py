import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf  # buat fallback interpreter
import os

CLASS_NAMES = ['bika ambon', 'kerak telor', 'papeda', 'plecing kangkung']
IMG_SIZE = (224, 224)  # pakai shape training asli kamu

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TFLITE_PATH = os.path.join(BASE_DIR, "..", "models", "BestModel_CustomCNN_A_KERAS.tflite")

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_image(file_obj):
    img = Image.open(file_obj).convert("RGB")
    img = img.resize(IMG_SIZE)

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1,H,W,3)

    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]  # shape (4,)
    probs = tf.nn.softmax(preds).numpy()

    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], probs, img

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
            label, probs, img_disp = predict_image(f)

            st.subheader(f"File: {f.name}")
            st.write(f"Prediksi: **{label}**")
            for cls_name, score in zip(CLASS_NAMES, probs):
                st.write(f"- {cls_name}: {score * 100:.2f}%")

            st.image(img_disp, caption=f.name, use_column_width=True)
            st.markdown("---")
