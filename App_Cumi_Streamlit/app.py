import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image

# Set Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Varian Cumi", page_icon="🦑", layout="centered")

# ==========================================
# 1. FUNGSI PEMUATAN MODEL (DENGAN CACHE)
# ==========================================
# @st.cache_resource agar model hanya di-load 1 kali saat aplikasi pertama kali dibuka
@st.cache_resource 
def load_model():
    # A. Bangun ulang arsitektur Sang Juara (EfficientNetB0)
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x) # Dropout sesuai hasil tuning pemenang
    outputs = layers.Dense(4, activation='softmax')(x)
    
    model = models.Model(inputs=base_model.input, outputs=outputs)
    
    # B. Load bobot dari file
    model.load_weights("Model_Final_Streamlit.weights.h5")
    return model

# ==========================================
# 2. INISIALISASI
# ==========================================
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    model_loaded = False

CLASSES = ['Cendol', 'CK', 'SMP DW', 'SMP UP']

# ==========================================
# 3. TAMPILAN USER INTERFACE (UI)
# ==========================================
st.title("🦑 Sistem Sortir Varian Cumi")
st.write("""
Aplikasi cerdas untuk mendeteksi kualitas dan varian cumi ekspor.
Silakan unggah foto cumi atau gunakan kamera untuk memprediksi.
""")

st.divider()

# Opsi Input: Kamera atau Upload File
input_method = st.radio("Pilih Metode Input Gambar:", ("Unggah Gambar", "Gunakan Kamera"))

image_file = None
if input_method == "Unggah Gambar":
    image_file = st.file_uploader("Pilih file gambar cumi...", type=["jpg", "jpeg", "png"])
else:
    image_file = st.camera_input("Ambil foto cumi")

# ==========================================
# 4. LOGIKA PREDIKSI
# ==========================================
if image_file is not None:
    # Tampilkan gambar yang diunggah
    image = Image.open(image_file)
    st.image(image, caption="Gambar yang akan diproses", use_column_width=True)
    
    # Tombol Prediksi
    if st.button("🔍 Analisis Varian Cumi", type="primary"):
        if model_loaded:
            with st.spinner("Model sedang menganalisis tekstur cumi..."):
                # Preprocessing Gambar
                img_resized = image.resize((224, 224)) # Ubah ukuran sesuai input model
                img_array = tf.keras.utils.img_to_array(img_resized)
                img_batch = np.expand_dims(img_array, axis=0) # Tambahkan dimensi batch
                
                # Prediksi
                predictions = model.predict(img_batch)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                
                predicted_label = CLASSES[predicted_class_idx]
                
                # Menampilkan Hasil
                st.success("Analisis Selesai!")
                st.subheader(f"Prediksi: **{predicted_label}**")
                st.progress(float(confidence))
                st.write(f"Tingkat Keyakinan (Confidence): **{confidence * 100:.2f}%**")
                
                # Menampilkan detail probabilitas semua kelas (Opsional, bagus untuk dosen)
                with st.expander("Lihat Detail Probabilitas Semua Kelas"):
                    for i, class_name in enumerate(CLASSES):
                        st.write(f"- {class_name}: {predictions[0][i] * 100:.2f}%")
        else:
            st.error("Model belum siap, silakan refresh halaman.")