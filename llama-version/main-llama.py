from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

# Retrivers
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

# Setting up the chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

###################################################################################################################
# configuration for the model
llm = Ollama(model="llama2")
# llm.invoke("how can langsmith help with testing?")
user_input = input("What is your question? ")


# create vector-db from a document
loader = TextLoader("../doc.txt")
documents = loader.load()

embeddings = OllamaEmbeddings(temperature=0.7)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
# text_embeddings = zip(docs, embeddings)
vector = FAISS.from_documents(docs, embeddings)


# create a retrieval chain:
# 1) create prompt - Most LLM applications do not pass user input directly into an LLM. 
# #Usually they will add the user input to a larger piece of text, called a prompt template, 
# that provides additional context on the specific task at hand
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {user_input}""")
                                          
print("PROMPT", prompt)

# 2) create chain - will take an incoming question, look up relevant documents, t
# hen pass those documents along with the original question into an LLM and ask it to answer the original question
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": user_input})
print(response["answer"])