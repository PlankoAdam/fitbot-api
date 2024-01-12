from langchain_community.llms import llamacpp
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

model = llamacpp.LlamaCpp(
    model_path='./llms/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    temperature=0.1,
    max_tokens=1500,
    stop=['\n'],
    verbose=True,
    echo=True,
    n_ctx=1024,
)

embedder = HuggingFaceBgeEmbeddings(
    model_name = "BAAI/bge-base-en-v1.5",
    model_kwargs = {'device': 'cpu'},
    encode_kwargs = {'normalize_embeddings': True},
    query_instruction="Represent this sentence for searching relevant passages:"
)

template = """[INST]Answer the question based only on the following context. If the question can't be answered based on the context, say that you are unable to answer the question:
'...{context}...'

Question: {question}[/INST]
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=0, separators=['.', '?', '!']
)

def split(text):
    chunks = splitter.split_text(text=text)
    return chunks

def processTXT(fname):
    f = open('./uploads/%s' % fname, 'r', encoding="utf-8");
    text = f.read()
    f.close()
    return split(text)

def processPDF(fname):
    f = open('./uploads/%s' % fname, 'rb')
    pdf = PdfReader(f);
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    f.close()
    return split(text)

def getContext(textChunks, query):
    db = FAISS.from_texts(textChunks, embedding=embedder)
    retriever = db.as_retriever()
    retriever_result = retriever.invoke(query)
    return retriever_result[0].page_content

def answer(context, query):
    chain =  prompt | model | StrOutputParser()
    return chain.invoke({"context": context, "question": query})