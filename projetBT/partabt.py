import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Menggunakan HTML dan CSS untuk membuat tulisan berjalan
marquee_html = """
    <div style="overflow:hidden; white-space:nowrap;">
        <marquee behavior="scroll" direction="left" scrollamount="10">
            <h2 style="color: white;"> Universitas Bali Internasional</h2>
            <h5 style="color: blue;"> @ksnparta!</h5>
        </marquee>
    </div>
"""
st.markdown(marquee_html, unsafe_allow_html=True)
# Fungsi untuk mendeteksi koloni pada gambar
def detect_colonies(image):
    # Mengubah gambar ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Deteksi tepi untuk membantu memisahkan koloni
    edges = cv2.Canny(gray, 50, 150)
    
    # Menggunakan threshold untuk membedakan koloni
    ret, thresh = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
    
    # Menemukan kontur dari koloni
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Menggambar kotak pembatas di sekitar setiap koloni
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > 50:  # Hanya deteksi objek besar
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, len(contours)

# Fungsi utama Streamlit untuk membuat UI dan menjalankan deteksi koloni
def main():
    st.title("Deteksi Koloni Bakteri di Cawan Petri")

    # Upload gambar
    uploaded_file = st.file_uploader("Unggah gambar cawan petri", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Membaca gambar yang diunggah
        image = np.array(Image.open(uploaded_file))
        
        # Menampilkan gambar asli
        st.image(image, caption='Gambar Asli', use_column_width=True)
        
        # Deteksi koloni pada gambar
        processed_image, colony_count = detect_colonies(image.copy())
        
        # Menampilkan gambar hasil deteksi
        st.image(processed_image, caption=f'Deteksi Koloni (Jumlah: {colony_count})', use_column_width=True)
        
        # Menampilkan jumlah koloni yang terdeteksi
        st.write(f"Jumlah koloni yang terdeteksi: {colony_count}")

if __name__ == "__main__":
    main()
