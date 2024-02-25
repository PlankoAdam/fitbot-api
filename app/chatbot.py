from langchain_community.llms import llamacpp
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter
from langchain_core.messages import AIMessage, HumanMessage

model = llamacpp.LlamaCpp(
    model_path="app/llms/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0,
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

template = """[INST]You are a chatbot having a conversation with a human.
Given the following extracted parts of a long document and a question, create a final answer.
Context:
'...{context}...'[/INST]

Human: {human}
Chatbot:"""

prompt = PromptTemplate(template=template, input_variables=["context", "human"])

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=150, separators=['.', '?', '!']
)

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{human}"),
    ]
)
contextualize_q_chain = contextualize_q_prompt | model | StrOutputParser()

rag_chain = prompt | model | StrOutputParser()

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

def getContext(textChunks, query, msgs):
    db = FAISS.from_texts(textChunks, embedding=embedder)
    retriever = db.as_retriever()

    print('msgs:')
    print(*msgs, sep = '\n')
    print('msgs len: ' + str(len(msgs)))
    if (len(msgs) > 1):
        history = []
        for i in range(len(msgs)):
            if(i%2 == 0):
                history.append(HumanMessage(content=msgs[i]))
            else:
                history.append(AIMessage(content=msgs[i]))
        standalone_q = contextualize_q_chain.invoke({
            "history": history,
             "human": query
        })
    else:
        standalone_q = query

    print('standalone q: ' + standalone_q)
    retriever_result = retriever.invoke(standalone_q)
    return retriever_result[0].page_content

def answer(textChunks, query, msgs):
    ctx = getContext(textChunks, query,msgs);
    print('ctx: ' + ctx)

    ans = rag_chain.invoke({"context": ctx, "human": query})
    return {"answer": ans, "context": ctx}