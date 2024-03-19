from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import dotenv_values

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Retrivers
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# Setting up the chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


app = Flask(__name__)
CORS(app, origins=["*"], supports_credentials=True)


def _get_api_key() -> str:
    """Retrieves secret api key from configuration file"""
    config = dotenv_values(".env")
    return config["API_KEY"]


def _get_openai_llm(open_ai_key) -> ChatOpenAI:
    """Instantiates and returns an llm object"""
    return ChatOpenAI(openai_api_key=open_ai_key)


def _build_vectorstore_index(open_ai_key: str) -> object:
    """
    Creates a vectorstore for large amounts of data.
    Input value: a string of openai_key.
    Output value: an object of the vectorstore.
    """
    
    loader = TextLoader("./doc.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key=open_ai_key)
    return FAISS.from_documents(texts, embeddings)


def _create_prompt_for_inputs() -> ChatPromptTemplate:
    """
    Creates a prompt as most LLM apps add user's input into a prompt for better results on search (make search more precise).
    Output value: a ChatPromptTemplate instance.
    """

    return ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
        <context>
        {context}
        </context>
        Question: {input}""")


def _establish_retrieval_chain(llm, prompt, db):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever()
    return create_retrieval_chain(retriever, document_chain)

def ask_llm(question):
    """
    Uses langchain to implement RAG technique.
    Creates a vectorstore from a single document and establishes a retrieval chain.
    The retrieval chain on receiving an input looks up the vectorstore => 
    passes the relevant extracted info to the promt with the input itself.
    The LLM processes the prompt, not the standalone question.

    Input value: a string that represents user's question.
    Output value: response string.
    """

    open_ai_key = _get_api_key()
    llm = _get_openai_llm(open_ai_key)
    db = _build_vectorstore_index(open_ai_key)
    prompt = _create_prompt_for_inputs()

    retrieval_chain = _establish_retrieval_chain(llm, prompt, db)
    response = retrieval_chain.invoke({"input": question})

    return response

@app.route('/askllm', methods=['POST'])
def prepare_answer_with_rag():
    """
    Listens to POST requests that should include JSON data with 'question' key. 
    The value should represent a question posed for LLM.
    Output value: JSON object with response.
    """

    if request.method == 'POST':
        # Assuming JSON data is sent in the request
        user_input = request.get_json()["question"]

        if user_input:
            response = ask_llm(user_input)
            
            response_message = {"message": response["answer"]}
            return jsonify(response_message), 200
        else:
            error_message = {"error": "Invalid JSON format in POST data"}
            return jsonify(error_message), 400


if __name__ == '__main__':
    app.run(debug=True)