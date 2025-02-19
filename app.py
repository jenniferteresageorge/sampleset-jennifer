# necessary Imports
from PyPDF2 import PdfReader
from langchain_community.llms import Cohere
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts  import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
import gradio as gr
# Example PDF Path
pdf_file = "Sample Financial Statement.pdf"
# extracting pdf data
pdf_text = ""
pdf_reader = PdfReader(pdf_file)
for page in pdf_reader.pages:
    pdf_text += page.extract_text()

# merging all the text

all_text = pdf_text
len(all_text)
# splitting the text into chunks for embeddings creation

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200, # This is helpul to handle the data loss while chunking.
        length_function = len,
        separators=['\n', '\n\n', ' ', '']
    )

chunks = text_splitter.split_text(text = all_text)
len(chunks)
import os
os.environ['HuggingFaceHub_API_Token']= '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'
os.environ['GOOGLE_API_KEY']= "AIzaSyCoGAkfKk2JcAUS829HSSEo-Tnz72yP0fo"
os.environ['cohere_api_key'] = "jTIqO0PIhrTAV6OalQjp8U6MvhvrbAu6Y6aeZ67K"
# Initializing embeddings model

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
# Indexing the data using FAISS
vectorstore = FAISS.from_texts(chunks, embedding = embeddings)
# creating retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
# Prompt template for financial metrics
prompt_template = """You are a financial assistant specialized in Profit & Loss (P&L) statements.
Answer the question as precise as possible using the provided context. If the answer is
not contained entirely in the context, Answer to your best knowledge" \n\n
Context: \n {context}?\n
Question: \n {question} \n
Answer:"""
prompt = PromptTemplate.from_template(template=prompt_template)
# function to create a single string of relevant documents given by Faiss.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# RAG Chain

def generate_answer(question):
    cohere_llm = Cohere(model="command", temperature=0.6, cohere_api_key=os.getenv('cohere_api_key'))

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | cohere_llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question)

# ## EXAMPLE 1 (irrelevant question response)
# ans = generate_answer("Show the operating margin for the last 6 months.")
# print(ans)

# #EXAMPLE 2 (direct answer)
# ans = generate_answer("What are the total expenses for Q2 2023?")
# print(ans)

# #EXAMPLE 3 (circling around and answering)
# ans = generate_answer("What is the net profit for the last quarter? if not available, then calulate")
# print(ans)
# ans = generate_answer("Show the revenue growth for the past year.")
# print(ans)

## PART - 2
def process_pdf(pdf_file):
    """Extracts text from a user-uploaded PDF."""
    pdf_text = ""
    pdf_reader = PdfReader(pdf_file.name)
    for page in pdf_reader.pages:
        pdf_text += page.extract_text() + "\n"

    # Splitting text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=250, separators=['\n', '\n\n', ' ', '']
    )
    chunks = text_splitter.split_text(text=pdf_text)

    # Indexing the data using FAISS
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    return retriever

retriever = None  # Global retriever variable
def upload_pdf(pdf_file):
    global retriever
    retriever = process_pdf(pdf_file)
    return "PDF uploaded and processed successfully! You can now ask financial questions."

# Define the prompt template
prompt_template = """You are a financial assistant specialized in Profit & Loss (P&L) statements.
Answer the question as precisely as possible using the provided context. If the answer is
not contained in the context, say "answer not available in context".

Context: \n{context}\n
Question: \n{question}\n
Answer:"""

prompt = PromptTemplate.from_template(template=prompt_template)
def generate_answer(question):
    if retriever is None:
        return "Please upload a PDF first."

    cohere_llm = Cohere(model="command", temperature=0.1, cohere_api_key=os.getenv('cohere_api_key'))

    rag_chain = (
        {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)), "question": RunnablePassthrough()}
        | prompt
        | cohere_llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question)
# Gradio UI
demo = gr.Blocks()

with demo:
    gr.Markdown("""# Financial QA Chatbot
    Upload a **Profit & Loss (P&L) Statement PDF**, and ask financial questions in real-time.
    """)

    with gr.Row():
        pdf_input = gr.File(label="Upload P&L PDF")
        upload_button = gr.Button("Process PDF")

    upload_status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        question_input = gr.Textbox(label="Ask a question about the financial data")
        submit_button = gr.Button("Get Answer")

    answer_output = gr.Textbox(label="Answer")

    upload_button.click(upload_pdf, inputs=[pdf_input], outputs=[upload_status])
    submit_button.click(generate_answer, inputs=[question_input], outputs=[answer_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
