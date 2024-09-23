import pickle
import os
import time  # For simulating delay in response streaming
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_objectbox.vectorstores import ObjectBox
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = ''

# Pull prompt using LangSmith
prompt = hub.pull("rlm/rag-prompt")

# Directory path where documents are stored
directory_path = "../docs"
# Filepath for the pickled vector store
pickle_file_path = "data.pkl"
# Function to create and save the vector store if not exists
def create_vector_store():
    # Initialize the DirectoryLoader with the directory path
    loader = DirectoryLoader(directory_path)
    # Load documents from the directory
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()
    # Create ObjectBox vector store
    vector = ObjectBox.from_documents(documents, embeddings, embedding_dimensions=768)
    # Save the vector store to a pickle file
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(vector, f)
    return vector


# Load or create the vector store
if os.path.exists(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(data)
    vector = ObjectBox.from_documents(documents, OpenAIEmbeddings(), embedding_dimensions=768)


# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o")

# Create the QA chain using the pulled prompt
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vector.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)


# Function to handle QA retrieval and yield response word by word
def get_answer(user_query):
    if user_query:
        # Get the full response
        full_response = qa_chain(user_query)
        words = full_response.split()

        # Yield each word with a delay to simulate streaming
        for word in words:
            yield word
            time.sleep(0.1)  # Adjust the delay as needed
    else:
        yield "Please enter a question."


