# Intelligent Chatbot – CHEPPA

CHEPPA is an intelligent chatbot designed to handle queries related to universities and locations, particularly Montclair State University. It leverages LangChain, vector databases, and large language models (LLMs) to provide contextually relevant and accurate responses. This project focuses on improving the user experience by integrating Retrieval-Augmented Generation (RAG) techniques for efficient information retrieval.

## Features

- **University-Based Queries**: Provides information about courses, faculty, campus, events, and other academic details.
- **Location-Based Queries**: Assists users in navigating and finding locations on campus.
- **Context-Aware Responses**: Uses LangChain and vector databases to retrieve precise and contextually accurate information.
- **Conversational Interface**: Allows users to interact with the system through a user-friendly chat interface.

## Technologies Used

- **LangChain**: Framework for building applications using LLMs.
- **OpenAI GPT-4**: Provides the language model for understanding and generating responses.
- **FAISS**: Vector database for similarity search and clustering.
- **Flask**: Backend for the web-based user interface.
- **HTML/CSS/JavaScript**: Frontend for creating an interactive chat interface.

## System Requirements

### Software
- Python 3.8+
- Libraries: `langchain`, `openai`, `faiss`, `pickle5`, `flask`

### Hardware
- **Development**:
  - CPU: Multi-core processor
  - RAM: 16 GB or more
  - Storage: SSD for faster data access

- **Deployment**:
  - Server: High-performance cloud or on-premises server
  - RAM: 32 GB or more
  - GPU (optional): For faster processing during model training

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/T-Chaitanya/Intelligent_Chatbot---CHEPPA.git
   cd Intelligent_Chatbot---CHEPPA
   ```

2. **Install Dependencies**:
   Use `pip` to install the required Python packages.
   ```bash
   pip install -r requirements.txt
   ```

3. **Set OpenAI API Key**:
   Export your OpenAI API key to use GPT models:
   ```bash
   export OPENAI_API_KEY='your-api-key'
   ```

4. **Run the Application**:
   Launch the Flask application:
   ```bash
   python app.py
   ```

5. **Access the Web Interface**:
   Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## How It Works

- **Document Loading**: Loads university-related PDFs and splits them into manageable text chunks.
- **Embedding Creation**: Converts text chunks into vector embeddings using OpenAI embeddings.
- **Similarity Search**: Utilizes FAISS to search for semantically similar text based on user queries.
- **Response Generation**: Uses the OpenAI GPT model to generate natural language responses from retrieved document chunks.

## Future Enhancements

- **Knowledge Base Expansion**: Add more data to improve the chatbot’s coverage of university information.
- **Location Integration**: Incorporate GPS and campus maps to enhance location-based queries.
- **Multilingual Support**: Add support for other languages to cater to a broader audience.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
