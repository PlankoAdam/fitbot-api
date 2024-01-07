from langchain.llms import llamacpp
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


model = llamacpp.LlamaCpp(model_path='app/llms/llama-2-7b.Q4_K_M.gguf',
                        temperature=0.5,
                        max_tokens=1024,
                        stop=['Human:'],
                        # streaming=True,
                        # callbacks=[StreamingStdOutCallbackHandler()],
                        verbose=False,
                        )

template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
Current conversation: {history} Human: {input}
AI:"""

prompt = PromptTemplate(
    input_variables=["history", "input"], template=template
)

memory = ConversationBufferWindowMemory(
    llm=model,
    k=6,
    return_messages=False
)

chain = ConversationChain(
    llm=model,
    prompt=prompt,
    verbose=False,
    memory=memory
)

def generate(userMassage):
    return chain.predict(input=userMassage).strip()

def clearHistory():
    memory.clear()