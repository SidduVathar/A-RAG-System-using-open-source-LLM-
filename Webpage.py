import streamlit as st
from fpdf import FPDF
import tempfile
import os
from datetime import datetime




# ===============================================================================
# Import necessary components from AI.py
# Make sure AI.py is in the same directory as app.py, or in your Python path
from AIAssistance import model, index, chunks, llm, retrieve_relevant_chunks, clean_text, pdf_paths, pdfplumber, np, faiss
# ================================================================================

# --- Page Configuration ---
st.set_page_config(page_title="AI Chat Assistant", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS (Optional: for fine-tuning beyond Streamlit's defaults) ---
st.markdown("""
    <style>
        /* General body styling (optional) */
        body {
            font-family: 'Arial', sans-serif;
        }

        /* Sidebar styling */
        .css-1d391kg { /* Target Streamlit sidebar */
            /* background-color: #f0f2f6; */ /* Example: Light gray sidebar */
        }

        /* Chat messages styling */
        .stChatMessage {
            border-radius: 10px;
            padding: 10px 15px;
            margin-bottom: 10px;
        }
        
        /* Center title and description */
        .centered-title {
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .centered-desc {
            text-align: center;
            color: gray;
            margin-bottom: 2rem;
        }
        /* Ensure chat input is at the bottom (st.chat_input handles this well) */
        /* Forcing chat container to take available height */
        .main .block-container {
            padding-bottom: 5rem; /* Adjust if input bar overlaps last message */
        }
    </style>
""", unsafe_allow_html=True)

# --- Title and Description ---
st.markdown("""
    <h2 class='centered-title' style='font-size:1.5rem; margin-bottom:0.2rem;'>🧠 AI Chat Assistant</h2>
    <p class='centered-desc' style='font-size:1rem; margin-bottom:1.2rem;'>I'm here to help! Ask me anything, or upload a PDF to discuss its content.</p>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_uploaded_file" not in st.session_state:
    st.session_state.current_uploaded_file = None # To store info about the active file

# --- Helper Functions ---
# Now using the actual RAG logic
def query_ai(user_input, uploaded_file_name=None):

    if uploaded_file_name:

        prefix = f"Regarding the file '{uploaded_file_name}', "
    else:
        prefix = ""


    relevant_chunks = retrieve_relevant_chunks(user_input, model, index, chunks)
    context = "\n".join(relevant_chunks)
    response = llm.complete(prompt=f"{context}\n\nQuestion: {user_input}\nAnswer:")
    
    return f"{prefix}{response}"


def export_to_pdf(chat_history):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    pdf.set_fill_color(240, 240, 240) # Light grey for assistant messages
    line_height = 8
    margin = 10

    for entry in chat_history:
        role = entry['role']
        content = entry['content']
        timestamp = entry.get('timestamp', '') # Get timestamp if available

        is_user = role == "user"

        pdf.set_font("Arial", "B" if is_user else "", 10)
        if is_user:
            pdf.set_text_color(0, 0, 0) # Black for user
            pdf.cell(0, line_height, f"You ({timestamp}):", ln=1)
        else:
            pdf.set_text_color(50, 50, 50) # Dark grey for AI
            pdf.cell(0, line_height, f"AI ({timestamp}):", ln=1)

        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(0, 0, 0) # Reset to black for content

        # Handle multi-line content
        lines = content.split('\n')
        for line in lines:
            # Basic word wrapping
            if pdf.get_string_width(line) > (pdf.w - 2 * margin):
                words = line.split(' ')
                current_line = ""
                for word in words:
                    if pdf.get_string_width(current_line + word + " ") < (pdf.w - 2 * margin):
                        current_line += word + " "
                    else:
                        pdf.multi_cell(0, line_height, current_line.strip(), border=0, align='L', fill=not is_user and role=="assistant")
                        current_line = word + " "
                pdf.multi_cell(0, line_height, current_line.strip(), border=0, align='L', fill=not is_user and role=="assistant")
            else:
                pdf.multi_cell(0, line_height, line, border=0, align='L', fill=not is_user and role=="assistant")

        pdf.ln(line_height / 2) # Add a small space after each message block

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    return temp_file.name

# --- Sidebar ---
with st.sidebar:
    st.markdown("## Controls")

    uploaded_file = st.file_uploader("➕ Upload PDF Report", type="pdf", key="pdf_uploader")

    
    if uploaded_file is not None:
        # Check if it's a new file or the same one
        if st.session_state.current_uploaded_file != uploaded_file.name:
            st.session_state.current_uploaded_file = uploaded_file.name
            st.success(f"Uploaded: {uploaded_file.name}. This file's content is not yet dynamically indexed for RAG. Queries will use pre-loaded documents.")
      
            file_upload_message = f"File '{uploaded_file.name}' uploaded successfully. Ask me questions about it!."
            st.session_state.messages.append({"role": "assistant", "content": file_upload_message, "timestamp": datetime.now().strftime("%H:%M:%S")})

    elif st.session_state.current_uploaded_file:
        st.info(f"Previously uploaded: '{st.session_state.current_uploaded_file}'. Upload a new one to change.")


    if st.button("🗑️ Delete Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_uploaded_file = None # Clear active file on history delete
        st.rerun()

    st.button("WielandSalesIQ", use_container_width=True, on_click=None)

    if st.session_state.messages: # Only show export if there's history
        pdf_path_export = export_to_pdf(st.session_state.messages)
        with open(pdf_path_export, "rb") as f:
            st.download_button(
                label="📄 Export Chat to PDF",
                data=f,
                file_name=f"AIChat_Export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        os.remove(pdf_path_export) # Clean up temp file

    st.markdown("---")


# --- Display Chat History ---
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.info("No messages yet. Start by typing a message below or uploading a document.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.caption(f"<small>{message['timestamp']}</small>", unsafe_allow_html=True)


# --- Chat Input Area (at the bottom) ---
prompt_placeholder = "Ask your question..."
if st.session_state.current_uploaded_file:
    prompt_placeholder = f"Ask about '{st.session_state.current_uploaded_file}' or type a general query..."

if prompt := st.chat_input(prompt_placeholder, key="main_chat_input"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": datetime.now().strftime("%H:%M:%S")})

    # Get AI response
    with st.spinner("AI is thinking..."):
        ai_response_content = query_ai(prompt, st.session_state.current_uploaded_file)
    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response_content, "timestamp": datetime.now().strftime("%H:%M:%S")})

    # Rerun to display the new messages
    st.rerun()













