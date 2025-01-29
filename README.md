# QA Bot for Financial Data (RAG Model + Interactive Interface) (PART - 2) (PART - 1 is in the plchatbot.ipynb file in the repo)

## Overview
This project is a **Retrieval-Augmented Generation (RAG) model** designed as a **Question-Answering (QA) bot** for analyzing **Profit & Loss (P&L) statements** extracted from **PDF documents**. It retrieves relevant financial information and generates accurate responses using **Cohere LLM, FAISS, and HuggingFace embeddings**.

Additionally, the project includes an **interactive Gradio interface**, allowing users to:
- Upload **PDF documents containing P&L tables**.
- Ask **financial questions** in real-time.
- Retrieve **relevant segments** alongside AI-generated responses.

## Features
 **RAG Model** for **accurate financial insights**  
 **PDF Processing** to extract key financial metrics  
 **FAISS Vector Store** for efficient retrieval  
 **Gradio UI** for real-time user interaction  
 **Dockerized Deployment** for easy setup & scaling  

---

## **Setup Instructions**
### ** Clone the Repository**
```bash
git clone https://github.com/jenniferteresageorge/qa-bot.git
```

### ** Install Dependencies**
#### **Locally (without Docker)**
Ensure you have **Python 3.9+** installed, then run:
```bash
pip install -r requirements.txt
```

#### **Using Docker (Recommended)**
Ensure **Docker** is installed, then build the image:
```bash
docker build -t qa-bot .
```
Run the container:
```bash
docker run -p 7860:7860 qa-bot
```
Gradio UI will be accessible at: **http://localhost:7860**

---

##  **Usage Instructions**
### **Locally (Python Script)**
1. Place your **P&L PDF file** in the project folder. I already have one included in the repo, download and use that.
2. Run the Python script:
   ```bash
   python app.py
   ```
3. Open the **Gradio interface** in your browser at `http://localhost:7860`
4. **Upload the PDF**, then ask financial questions.

### **Via Docker**
1. Run the container using:
   ```bash
   docker run -p 7860:7860 qa-bot
   ```
2. Open **http://localhost:7860** and interact with the QA bot.

---

## **Technical Approach**
### ** Data Processing**
- Extract **text** from **PDF** (using `PyPDF2`).
- Split text into **chunks** (using `RecursiveCharacterTextSplitter`).

### ** Embedding & Retrieval**
- Generate **embeddings** using **HuggingFace (`all-MiniLM-L6-v2`)**.
- Store embeddings in **FAISS vector database**.
- Retrieve **top relevant chunks** for answering queries.

### ** RAG Model (Retrieval-Augmented Generation)**
- Uses **Cohere LLM** for generating responses.
- Uses **custom prompt template** optimized for financial data.
- Generates **precise answers** based on retrieved P&L segments.

### ** Interactive UI (Gradio)**
- Users can **upload PDFs**.
- Ask **financial queries** (e.g., "What is net profit for Q3 2023?").
- See **retrieved P&L data** + AI-generated answers.

---

##  **Deployment Guide**
### **Containerization with Docker**
1. **Build Image:**
   ```bash
   docker build -t qa-bot .
   ```
2. **Run Container:**
   ```bash
   docker run -p 7860:7860 qa-bot
   ```


##  **Example Queries & Outputs**
### **Example Questions**
 "What is the total revenue for Q2 2023?"
 "Show the operating margin for the past 6 months."
 "How do net income and expenses compare for Q1 2024?" 

 

