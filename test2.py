from flask import Flask, request, Response, render_template, jsonify
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

app = Flask(__name__)
# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = ''
# This is the prompt used
template = """
You are Intelligent Chatbot CHEPPA, an assistant for question-answering. You are developed by Chaitanya Tata a graduate student at MSU. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Use ten sentences maximum and keep the answer concise.
Use only the context for your answers, do not make up information.
query: {query}
{context} 
"""

# Converts the prompt into a prompt template
prompt = ChatPromptTemplate.from_template(template)
# Using OpenAI model, by default gpt 3.5 Turbo
model = ChatOpenAI()

# Initialize embeddings with the API key
embeddings = OpenAIEmbeddings()

# Load the FAISS index from a file
db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={'k': 7})
# Construction of the chain
chain = (
    # The initial dictionary uses the retriever and user supplied query
        {"context": retriever,
         "query": RunnablePassthrough()}
        # Feeds that context and query into the prompt then model & lastly
        # uses the ouput parser, do query for the data.
        | prompt | model | StrOutputParser()

)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # def generate_response():
    full_response = chain.invoke(user_query)

    return jsonify({"message": full_response})


if __name__ == '__main__':
    app.run(debug=True)
