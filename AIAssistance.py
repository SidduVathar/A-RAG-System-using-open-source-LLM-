# pip install --upgrade llama-index
# %pip install pdfplumber
# %pip install pypdf2
# %pip install nltk
# %pip install faiss-cpu

# from nltk.tokenize import word_tokenize, sent_tokenize
from sentence_transformers import SentenceTransformer
from llama_index.llms.ollama import Ollama
from fpdf import FPDF
import os
import glob
import pdfplumber
import nltk
import re
from nltk.corpus import stopwords
import faiss
import numpy as np
nltk.download('stopwords')


# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "meta-llama/Llama-2-7b-chat-hf"  # Replace with your chosen model name
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)



# Set the directory containing your PDF files
pdf_dir = "--The files location"

# Get all the file with .pdf extension using glob
pdf_paths = glob.glob(os.path.join(pdf_dir, "*.pdf"))

full_text = ""

# Iterate over each PDF file
for pdf_path in pdf_paths:
    # Open the PDF with pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        # Loop through each page and extract text with layout preservation
        for page in pdf.pages:
            text = page.extract_text(layout=True)
            if text:
                full_text += text + "\n"    
                # print(text)
    # print("\n")  # Add spacing between documents


# Preload English stop‑words (optional)
STOPWORDS = set(stopwords.words('english'))

# Function to clean text
def clean_text(text: str, remove_stopwords: bool = False) -> str:
    full_text
    # To Normalize whitespace (merge line breaks and multiple spaces)
    text = re.sub(r'\s+', ' ', text)  # removes \n, \t, etc., replacing with single space:contentReference[oaicite:5]{index=5}
    # To Remove non‑alphanumeric characters (except . , ; : ? ! ' " -)
    text = re.sub(r'[^A-Za-z0-9\.\,\;\:\?\!\'\"\-\s]', '', text)  # strips emojis, symbols, etc.:contentReference[oaicite:6]{index=6}
    # To Remove watermark/confidential notes (case‑insensitive)
    text = re.sub(r'(?i)\b(confidential|watermark|draft)\b', '', text)  # drops common watermark terms:contentReference[oaicite:7]{index=7}
    # To Trim leading/trailing spaces
    text = text.strip()
    if remove_stopwords:
        # Optionally remove stop‑words
        tokens = text.split()
        tokens = [tok for tok in tokens if tok.lower() not in STOPWORDS]  # filters out 'the', 'and', etc.:contentReference[oaicite:8]{index=8}
        text = ' '.join(tokens)

    return text


# print("\nCleaned text:")
cleaned_text = clean_text(full_text, remove_stopwords=True)
# print(cleaned_text)

nltk.download('punkt')
# print(nltk.data.find('tokenizers/punkt'))

def chunk_text(text, max_length):
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += ' ' + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# Example usage
chunks = chunk_text(cleaned_text, max_length=100)
# for i, chunk in enumerate(chunks, 1):
#     print(f"Chunk {i}: {chunk}")

# Load a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for each chunk
embeddings = model.encode(chunks, convert_to_numpy=True)

# Convert embeddings to float32
embeddings = np.array(embeddings).astype('float32')

# Initialize FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the index
index.add(embeddings)

def retrieve_relevant_chunks(query, model, index, chunks, top_k=5):
    # Encode the quer
    query_embedding = model.encode([query]).astype('float32')

    # Search the index
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve corresponding chunks
    return [chunks[i] for i in indices[0]]


faiss.write_index(index, "faiss_index.idx")
# To load later:
index = faiss.read_index("faiss_index.idx")

# Initialize the LLM
llm = Ollama(model="llama2", request_timeout=5000.0)

def query_ai(user_input):
    ## Write the logic and instruction here on your requirements.
    relevant_chunks = retrieve_relevant_chunks(user_input, model, index, chunks)
    context = "\n".join(relevant_chunks)
    response = llm.complete(prompt=f"{context}\n\nQuestion: {user_input}\nAnswer:")
    return response


