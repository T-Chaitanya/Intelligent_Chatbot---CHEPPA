# uses OpenAI embeddings to build a retriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
api_key = ''
with open('data.pkl', 'rb') as f:
        docs = pickle.load(f)
embeddings = OpenAIEmbeddings(api_key=api_key)
# Creates the document retriever using docs and embeddings
db = FAISS.from_documents(docs, embeddings)



retriever = db.as_retriever(search_kwargs={'k': 3})
