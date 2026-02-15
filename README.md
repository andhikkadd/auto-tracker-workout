# Auto Tracker Exercise 🏋️‍♂️📷

Auto Tracker Exercise adalah project **computer vision** berbasis webcam yang bisa:
- Mendeteksi pose tubuh pakai MediaPipe
- Menghitung repetisi latihan secara otomatis
- Menampilkan UI real-time (HUD, FPS, tombol reset)
- Audio feedback (beep) setiap repetisi terhitung

 dibuat untuk **eksplorasi + latihan bikin project yang beneran selesai**, bukan cuma mangkrak 😅

---

## ✨ Features
- 📷 Real-time webcam tracking
- 🔢 Rep counter otomatis + total
- 🧠 Landmark smoothing (buat ngurangin jitter/noise)
- 🔔 Beep sound setiap repetisi (biar tau kalo keitung)
- 🖐️ Reset counter via “touch” tombol di layar (pakai jari/index landmark)
- 📊 FPS counter (biar keliatan keren walaupun kadang drop)

---

## Quick Start

### 1) Clone
```bash
git clone https://github.com/andhikkadd/auto-tracker-workout.git
cd auto-tracker-workout
```
### 2) Install dependencies
```bash
pip install -r requirements.txt
```
### 3) Run
```bash
python main.py
```

---

## Troubleshooting
Camera not opening
- Tutup aplikasi lain yang menggunakan kamera (Zoom, Meet, OBS).
- Coba ubah camera index di main.py:
```bash
cap = cv2.VideoCapture(0)  # coba 1 atau 2 kalau 0 tidak bekerja
```
Rep tidak terhitung / deteksi tidak stabil
- Pastikan pencahayaan cukup
- Tubuh terlihat jelas di kamera
- Jangan terlalu jauh dari kamera

---

## Notes
- Masih terdapat beberapa hal untuk improvement yg lebih stabil dan bug 😅
