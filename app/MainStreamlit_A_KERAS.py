import streamlit as st
import numpy as np
from PIL import Image

# =====================================================
# KONFIGURASI MODEL
# =====================================================
IMG_SIZE = (224, 224)  # input model CNN kamu
CLASS_NAMES = ['bika ambon', 'kerak telor', 'papeda', 'plecing kangkung']
MODEL_PATH = "models/BestModel_CustomCNN_A_KERAS.tflite"

# =====================================================
# Coba import interpreter ringan dulu
# =====================================================
InterpreterImpl = None
use_tflite_runtime = False
load_error = None

try:
    from tflite_runtime.interpreter import Interpreter as TFLiteRuntimeInterpreter
    InterpreterImpl = TFLiteRuntimeInterpreter
    use_tflite_runtime = True
except Exception as e1:
    load_error = e1
    try:
        import tensorflow as tf
        InterpreterImpl = tf.lite.Interpreter
        use_tflite_runtime = False
    except Exception as e2:
        # Ini kasus fatal: gak ada runtime sama sekali
        InterpreterImpl = None
        load_error = (e1, e2)

@st.cache_resource
def load_interpreter(model_path):
    if InterpreterImpl is None:
        raise RuntimeError(
            f"Tidak ada interpreter TFLite yang bisa dipakai.\nDetail error: {load_error}"
        )

    interpreter = InterpreterImpl(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details, use_tflite_runtime

interpreter, input_details, output_details, runtime_flag = load_interpreter(MODEL_PATH)

# =====================================================
# Fungsi prediksi
# =====================================================
def predict_image(file_obj):
    # 1. Baca dan resize gambar
    img = Image.open(file_obj).convert("RGB")
    img = img.resize(IMG_SIZE)

    # 2. Preprocess ke float32 [0..1], shape (1,H,W,3)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    # 3. Siapkan input utk interpreter
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    arr_for_model = arr.astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_index, arr_for_model)

    # 4. Inference
    interpreter.invoke()

    # 5. Output
    output_data = interpreter.get_tensor(output_index)  # (1,4)
    probs = output_data[0]  # (4,)

    idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[idx]

    return pred_label, probs


# =====================================================
# UI
# =====================================================
st.title("🍽 Klasifikasi Makanan Tradisional Indonesia")
st.write("Model CNN custom (.tflite)")

if runtime_flag:
    st.caption("Runtime: tflite_runtime (ringan, no TensorFlow)")
else:
    st.caption("Runtime: TensorFlow Lite Interpreter (full TF)")

uploads = st.file_uploader(
    "Upload gambar makanan",
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
            st.write("Probabilitas kelas:")
            for cls_name, score in zip(CLASS_NAMES, probs):
                st.write(f"- {cls_name}: {score * 100:.2f}%")

            st.image(Image.open(f), caption=f"Input: {f.name}", use_column_width=True)
            st.markdown("---")
