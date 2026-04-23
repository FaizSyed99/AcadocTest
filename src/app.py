# src/app.py
# Technical Plan §10: Start Small, Iterate - Frontend
import streamlit as st
import requests
import json

# Configuration
API_URL = "http://localhost:8001/api/chat"

st.set_page_config(page_title="AcaDoc AI", page_icon="🩺", layout="centered")

# Branding (Technical Plan §1)
st.title("🩺 AcaDoc AI")
st.markdown("*Textbook-Grounded Medical Assistant for Indian Medical Students*")
st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("📚 View Sources"):
                for cite in message["citations"]:
                    st.write(f"- {cite}")

# Chat Input
if prompt := st.chat_input("Ask a medical question (e.g., Clinical features of...)"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("🔍 Consulting textbooks..."):
            try:
                response = requests.post(API_URL, json={"query": prompt})
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    citations = data["citations"]
                    confidence = data["confidence"]
                    is_sufficient = data["is_sufficient"]

                    # Display Answer
                    st.markdown(answer)
                    
                    # Validation Status (§9)
                    if is_sufficient:
                        st.success(f"✅ Grounded Response (Confidence: {confidence:.2f})")
                        if citations:
                            with st.expander("📚 View Sources"):
                                for cite in citations:
                                    st.write(f"- {cite}")
                    else:
                        st.warning("⚠️ Fallback: Context Insufficient")
                    
                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "citations": citations
                    })
                else:
                    st.error("🔥 Error communicating with API.")
            except Exception as e:
                st.error(f"❌ Connection Error: Ensure API is running on port 8001. ({e})")

# Sidebar for Context Info
with st.sidebar:
    st.header("AcaDoc AI POC v1.2")
    st.info("""
    **Architecture (§8):**
    - Retriever: ChromaDB
    - Validator: Gated Context
    - Generator: Deterministic (T=0.0)
    """)
    st.warning("This is a POC for educational purposes only. Always consult standard textbooks.")
