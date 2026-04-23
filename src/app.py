# src/app.py
import streamlit as st
import requests
import os

st.set_page_config(page_title="AcaDoc AI", page_icon="🩺")

st.title("🩺 AcaDoc AI")
st.subheader("Textbook-Grounded Medical Tutor")

# Issue 5: Check for textbook data
data_dir = "./data"
pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")] if os.path.exists(data_dir) else []

if not pdf_files:
    st.warning("⚠️ No textbook loaded. Place a medical PDF in the /data folder and restart.")
else:
    st.info(f"📚 Loaded: {pdf_files[0]}")

query = st.text_input("Ask a medical question (MBBS/PG level):")

if st.button("Query Textbook"):
    if query:
        try:
            # Issue 4: Connect to Port 8001
            response = requests.post("http://localhost:8001/api/chat", json={"query": query})
            if response.status_code == 200:
                data = response.json()
                st.write("### Answer")
                st.write(data['answer'])
                
                if data.get('citations'):
                    with st.expander("Sources & Citations"):
                        for c in data['citations']:
                            st.write(f"- {c}")
            else:
                st.error(f"API Error: {response.status_code}")
        except Exception as e:
            # Issue 4: Connection validation
            st.error("❌ Connection Error: Is the AcaDoc API running on port 8001?")
    else:
        st.warning("Please enter a question.")
