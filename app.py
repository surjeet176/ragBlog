from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import (
    RunnableBinding,
    RunnableLambda, 
    RunnableAssign
)
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import os

# os.environ["LANGCHAIN_TRACING_V2"] = "true"

load_dotenv() 


def format_input(inputs):
    return f"Question: {inputs['question']}"


def get_faq_chain():
    embeddings = OpenAIEmbeddings(
        model='text-embedding-3-large'
    )

    loader = TextLoader("faq.txt")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(documents)
    db = Chroma.from_documents(
        docs, embeddings
    )
    retriever = db.as_retriever()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                '''You are helful Chatbot that takes Question and context. 
                    You will only answer based on the context. Give the Answer directly like a conversation, do not mention of the context.
                    If you don't know the answer, say that you don't know. Do not make up an answer.
                ''',
            ),
            (
                "human", 
                "[Question] : {question},\n[Context] : {context}"
            ),
        ]
    )
    open_ai_gpt = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    prompt_chain = RunnableBinding(
        bound=RunnableAssign(
            mapper={
                "context" : RunnableLambda(format_input) | retriever
            }
        )
    )
    faq_chain = prompt_chain | prompt | open_ai_gpt
    return faq_chain