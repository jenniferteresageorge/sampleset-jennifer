import os
from PyPDF2 import PdfReader
from langchain_community.llms import Cohere
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import gradio as gr

# Load API keys from environment variables
os.environ['HuggingFaceHub_API_Token'] = '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'
os.environ['cohere_api_key'] = 'ZR5JcWWBXe5aWmMd0QnTi6PvKGCCUXNi1B6QND3E'

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Define text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400, separators=['\n', '\n\n', ' ', ''])

# Define prompt template
prompt_template = PromptTemplate.from_template(
    """
    You are a financial assistant specialized in Profit & Loss (P&L) statements.
    Answer the question as precisely as possible using the provided context.
    If the answer is not available, say "answer not available in context".
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
)

# Function to process PDF and create retriever
def process_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file.name)
    pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    chunks = text_splitter.split_text(text=pdf_text)
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

retriever = None  # Global retriever variable

def upload_pdf(pdf_file):
    global retriever
    retriever = process_pdf(pdf_file)
    return "PDF uploaded and processed successfully!"

def generate_answer(question):
    global retriever
    if retriever is None:
        return "Please upload a PDF first."
    
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    cohere_llm = Cohere(model="command", temperature=0.9, cohere_api_key=os.getenv('cohere_api_key'))
    rag_chain = ({"context": lambda _: context, "question": RunnablePassthrough()} | prompt_template | cohere_llm | StrOutputParser())
    
    return rag_chain.invoke(question)

# Gradio UI
demo = gr.Blocks()

with demo:
    gr.Markdown("""# Financial QA Chatbot\nUpload a **Profit & Loss (P&L) Statement PDF**, and ask financial questions in real-time.""")
    pdf_input = gr.File(label="Upload P&L PDF")
    upload_button = gr.Button("Process PDF")
    upload_status = gr.Textbox(label="Status", interactive=False)
    question_input = gr.Textbox(label="Ask a question about the financial data")
    submit_button = gr.Button("Get Answer")
    answer_output = gr.Textbox(label="Answer")
    
    upload_button.click(upload_pdf, inputs=[pdf_input], outputs=[upload_status])
    submit_button.click(generate_answer, inputs=[question_input], outputs=[answer_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
