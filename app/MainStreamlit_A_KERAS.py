import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# KONFIGURASI MODEL
# =========================
IMG_SIZE = (224, 224)  # HARUS sama dengan input model CustomCNN kamu
CLASS_NAMES = ['bika ambon', 'kerak telor', 'papeda', 'plecing kangkung']

MODEL_PATH = "models/BestModel_CustomCNN_A_KERAS.tflite"  # path .tflite hasil konversi


# =========================
# LOAD TFLITE INTERPRETER
# =========================
@st.cache_resource
def load_tflite_interpreter(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Debug info kalau mau cek
    # st.write("Input details:", input_details)
    # st.write("Output details:", output_details)

    return interpreter, input_details, output_details


interpreter, input_details, output_details = load_tflite_interpreter(MODEL_PATH)


# =========================
# FUNGSI PREDIKSI
# =========================
def predict_image(file_obj):
    # 1. Baca gambar
    img = Image.open(file_obj).convert("RGB")
    img = img.resize(IMG_SIZE)

    # 2. Preprocess -> float32 [0..1], shape (1,H,W,3)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    # 3. Masukkan ke interpreter TFLite
    #    Sesuaikan dtype & index input
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    # Pastikan tipe datanya cocok
    arr_for_model = arr.astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_index, arr_for_model)

    # 4. Inference
    interpreter.invoke()

    # 5. Ambil output
    output_data = interpreter.get_tensor(output_index)  # shape (1,4)
    probs = output_data[0]  # shape (4,)

    # 6. Ambil label dengan skor tertinggi
    idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[idx]

    return pred_label, probs


# =========================
# UI STREAMLIT
# =========================
st.title("🍽 Klasifikasi Makanan Tradisional Indonesia")
st.write("Upload gambar makanan, lalu aku coba tebak apakah itu: "
         "`bika ambon / kerak telor / papeda / plecing kangkung`.")

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

            st.subheader(f.name)
            st.write(f"**Prediksi:** {label}")
            st.write("Skor probabilitas:")
