import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from PIL import Image
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Road Guardian AI",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS UNTUK TAMPILAN MODERN ---
st.markdown("""
<style>
    /* Import Font Keren */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;700&family=Orbitron:wght@500;900&display=swap');

    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Judul Utama */
    h1 {
        font-family: 'Orbitron', sans-serif;
        color: #FFC107; /* Kuning Konstruksi */
        text-shadow: 0 0 10px rgba(255, 193, 7, 0.5);
        text-align: center;
        font-size: 3rem !important;
    }
    
    /* Subjudul */
    .subtitle {
        text-align: center;
        font-family: 'Roboto', sans-serif;
        color: #aaaaaa;
        margin-bottom: 30px;
    }

    /* Card Statistik */
    .stat-card {
        background-color: #262730;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #444;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stat-number {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.5rem;
        color: #FF5252;
    }
    .stat-label {
        color: #ddd;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #1e1e1e;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNGSI CACHE MODEL ---
@st.cache_resource
def load_model(model_path):
    """Memuat model YOLOv8 sekali saja untuk performa."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# --- FUNGSI DETEKSI GAMBAR ---
def detect_image(model, image, conf_threshold):
    # Konversi PIL Image ke format OpenCV
    img_array = np.array(image)
    
    # Run inference
    results = model(img_array, conf=conf_threshold)
    
    # Plot hasil deteksi pada gambar
    res_plotted = results[0].plot()
    
    # Hitung jumlah deteksi
    count = len(results[0].boxes)
    
    return res_plotted, count

# --- FUNGSI DETEKSI VIDEO ---
def detect_video(model, video_path, conf_threshold):
    cap = cv2.VideoCapture(video_path)
    st_frame = st.empty() # Placeholder untuk video
    
    stop_button = st.button("⏹️ Hentikan Proses Video")
    
    total_detections = 0
    frame_count = 0
    
    while cap.isOpened():
        if stop_button:
            break
            
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # YOLO Processing
        results = model(frame, conf=conf_threshold)
        res_plotted = results[0].plot()
        
        # Hitung deteksi di frame ini
        det_in_frame = len(results[0].boxes)
        total_detections += det_in_frame
        
        # Tampilkan frame (Konversi BGR ke RGB untuk Streamlit)
        st_frame.image(res_plotted, channels="BGR", caption=f"Processing Frame {frame_count} | Detected: {det_in_frame}")
    
    cap.release()
    return total_detections

# --- MAIN UI ---
def main():
    # HEADER
    st.markdown("<h1>🚧 Road Guardian AI</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Sistem Deteksi Kerusakan Jalan Otomatis Berbasis YOLOv8</div>", unsafe_allow_html=True)

    # SIDEBAR CONFIG
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=50)
        st.header("⚙️ Konfigurasi")
        
        # Pilihan Model (Bisa diganti path model custom kamu nanti)
        # Jika kamu punya file 'best.pt' hasil training, letakkan di folder yang sama
        model_source = st.selectbox("Pilih Model", ["YOLOv8n (Pretrained Demo)", "Custom Model (best.pt)"])
        
        if model_source == "Custom Model (best.pt)":
            model_path = "best.pt" 
            if not os.path.exists(model_path):
                st.warning("⚠️ File 'best.pt' tidak ditemukan. Menggunakan model standar.")
                model_path = "yolov8n.pt" 
        else:
            model_path = "yolov8n.pt" # Default YOLO buat demo (bisa deteksi mobil/orang dulu sbg tes)

        # Slider Confidence
        conf = st.slider("Sensitivitas Deteksi (Confidence)", 0.0, 1.0, 0.45)
        
        st.divider()
        st.info("Aplikasi ini menggunakan Computer Vision untuk mendeteksi lubang (potholes) dan retakan jalan secara real-time.")

    # LOAD MODEL
    model = load_model(model_path)

    # TABS UTAMA
    tab1, tab2 = st.tabs(["📸 Deteksi Gambar", "🎥 Deteksi Video"])

    # --- TAB 1: GAMBAR ---
    with tab1:
        uploaded_img = st.file_uploader("Upload Foto Jalan", type=['jpg', 'png', 'jpeg'])
        
        if uploaded_img and model:
            col1, col2 = st.columns(2)
            
            # Tampilkan Gambar Asli
            image = Image.open(uploaded_img)
            with col1:
                st.markdown("### Gambar Asli")
                st.image(image, use_container_width=True)
            
            # Tombol Proses
            if st.button("🔍 Analisis Kerusakan", key="btn_img"):
                with st.spinner("Sedang memindai permukaan jalan..."):
                    res_img, count = detect_image(model, image, conf)
                    
                    with col2:
                        st.markdown("### Hasil Deteksi")
                        st.image(res_img, use_container_width=True)
                
                # Statistik Hasil
                st.markdown("---")
                st.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-number'>{count}</div>
                    <div class='stat-label'>Titik Kerusakan Terdeteksi</div>
                </div>
                """, unsafe_allow_html=True)
                
                if count > 0:
                    st.error("⚠️ PERINGATAN: Jalan ini membutuhkan perbaikan segera!")
                else:
                    st.success("✅ AMAN: Tidak ditemukan kerusakan signifikan.")

    # --- TAB 2: VIDEO ---
    with tab2:
        uploaded_vid = st.file_uploader("Upload Video Dashcam", type=['mp4', 'avi', 'mov'])
        
        if uploaded_vid and model:
            # Simpan video ke temp file karena OpenCV butuh path file
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_vid.read())
            
            st.video(tfile.name) # Preview video asli
            
            if st.button("▶️ Mulai Analisis Video", key="btn_vid"):
                st.markdown("### 🔴 Live Analysis")
                total_defects = detect_video(model, tfile.name, conf)
                
                st.success(f"Analisis Selesai! Total akumulasi deteksi: {total_defects}")

if __name__ == "__main__":
    main()