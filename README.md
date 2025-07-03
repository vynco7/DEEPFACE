# DEEPFACE
# Cyber Fast Detector

Deteksi wajah + emosi + umur + kontrol gesture tangan untuk play/pause YouTube.  
Cocok untuk sistem kontrol cyber, AI demo, atau proyek interaktif.

## 🚀 Fitur
- Deteksi wajah + kotak hijau
- Prediksi umur & ekspresi (DeepFace)
- Deteksi tangan & hitung jari
- Gesture jempol & telunjuk menyatu → Play YouTube
- Gesture jempol & telunjuk pisah → Pause YouTube
- Panel status samping

## 💻 Cara Jalankan

1️⃣ Aktifkan virtual environment:
bash
source ~/myenv/bin/activate
2️⃣ Install modul (kalau belum):

pip install -r requirements.txt

3️⃣ Jalankan script:

python kameradeepface.py

⚡ Kontrol Gesture

    Jempol & telunjuk nyatu: Play (tekan space)

    Jempol & telunjuk pisah: Pause (tekan space)

✅ Hotkey

    ESC → Keluar program

💬 Info Tambahan

    Pastikan kamera menyala dan wajah terlihat jelas

    DeepFace hanya pakai age & emotion untuk kecepatan

    Panel samping menampilkan status real-time




## ✅ **Isi file `requirements.txt`**

txt
opencv-python
mediapipe
numpy
pyautogui
deepface

✅ File struktur final di folder home

/home/finnnz
│
├── kameradeepface.py        # script utama
├── README.md                # penjelasan cara pakai
├── requirements.txt         # daftar modul
├── myenv/                   # virtual environment
└── (file .py lain jika ada)

💣 Langkah terakhir (setelah pindah)

cd ~
source myenv/bin/activate
pip install -r requirements.txt
python kameradeepface.py
