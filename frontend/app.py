import streamlit as st
import requests
from PIL import Image

st.title("Industrial Metal Defect Classifier")
st.write("Upload a metal surface image for real-time inference.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
  
    image = Image.open(uploaded_file)
    st.image(image, caption="Scanned Surface", use_container_width=True)
    
    if st.button("Run Inspection"):
        with st.spinner("Connecting to FastAPI Engine..."):
            try:
                
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                
               
                response = requests.post("http://127.0.0.1:8000/predict", files=files)
                
                
                if response.status_code == 200:
                    result = response.json()
                    confidence_pct = result['confidence'] * 100
                    
                    st.success(f"**Detected Defect:** {result['defect_type']}")
                    st.info(f"**Network Confidence:** {confidence_pct:.2f}%")
                else:
                    st.error(f"API Error: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Fatal: Cannot reach the inference engine. Is FastAPI running?")