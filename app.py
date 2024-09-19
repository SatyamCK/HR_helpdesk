import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from dotenv import load_dotenv
from conn import *
import numpy as np
import faiss
from langchain.docstore import InMemoryDocstore
import os
import tempfile
load_dotenv()
GOOGLE_API_KEY=os.environ["GOOGLE_API_KEY"]
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    convert_system_message_to_human=True
)

st.set_page_config(
    page_title="Document Summarizer Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
#load the document
def load_documents(file_path):
    try:
        # Use PyPDFLoader to load and split PDF documents
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        #  Split the extracted text into documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)  # Updated method to split raw text
        return docs
    except Exception as e:
        st.error(f"Error loading PDF file: {str(e)}")
        return []
def create_vector_store(docs):
    try:
        # Use Google Generative AI embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        # Create the FAISS vector store from the documents and embeddings
        vectorstore = FAISS.from_documents(docs, embeddings)
        save_chunks_and_embeddings(docs,embeddings)
        return vectorstore
    
    except ModuleNotFoundError as e:
        st.error(f"Module not found: {e}")
        raise
    
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        raise

    

def save_chunks_and_embeddings(docs, embeddings):
    for i, doc in enumerate(docs):
        doc_embedding = embeddings.embed_query(doc.page_content)
        insert_query(doc.page_content, doc_embedding) 

def insert_document_into_db(doc):
    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as temp_file: 
        temp_file.write(doc.read())
        temp_file_path=temp_file.name
    docs=load_documents(temp_file_path)
    if docs:
        create_vector_store(docs)
        st.success("Document uploaded succesfully")
    else:
        st.error("Failed to process the document")    

    os.remove(temp_file_path)



def create_faiss_retriever_from_db():
    try:
        docs, embeddings = fetch_chunks_and_embedding()
 
        # Check if docs and embeddings are present and their lengths match
        if not docs or not embeddings or len(docs) != len(embeddings):
            raise ValueError("The number of documents does not match the number of embeddings.")
 
        # Convert embeddings to numpy array
        embeddings_np = np.array(embeddings)
 
        # Initialize FAISS index
        faiss_index = faiss.IndexFlatL2(embeddings_np.shape[1])
        faiss_index.add(embeddings_np)  # Add embeddings to FAISS index
 
        # Create documents in LangChain format
        documents = [Document(page_content=doc) for doc in docs]
 
        # Create a mapping from FAISS index to document store IDs
        index_to_docstore_id = {i: str(i) for i in range(len(embeddings))}
 
        # Create document store using LangChain's InMemoryDocstore
        docstore = InMemoryDocstore(dict(zip(index_to_docstore_id.values(), documents)))
 
        # Use the Google Gemini model for embedding (assuming it's properly configured)
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
 
        # Create FAISS vector store
        vectorstore = FAISS(
            embedding_function=embedding_model,
            index=faiss_index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
 
        # Return the retriever
        return vectorstore.as_retriever(search_kwargs={"k": 5})
 
    except Exception as e:
        print("FAISS retriever creation failed :", e)
        return None


def generate_answer(query, retriever):
    try:
        if retriever is None:
            raise ValueError("Retriever is not initialized")
        
        relevant_docs = retriever.get_relevant_documents(query)
        # print(relevant_docs)
        context = "\n".join([doc.page_content for doc in relevant_docs])

        # prompt = f"Based on the documents:\n{context}\n\nQuestion: {query}\nAnswer:"
        # Create a prompt template
        prompt_template = """
            You are an intelligent assistant trained to provide insightful responses to user questions. A document is provided to you. Your task is to generate a clear and concise answer from the provided document. If you are not able to find the answer in the provided document, generate a response related to the word present in the question. Provide a response in a more detailed manner.
 
            Context : {context}
            Question: {question}

            Instructions:
                1. Greeting : Respond to any greetings in a warm and professional manner.
                2. Introduction: Provide a brief introduction that frames the response to the user’s question.
                3. Detailed Explanation: Dive into the core details, breaking down the answer into key points. Provide context, reasoning, and examples if needed.
                4. Examples/Analogies: Where applicable, illustrate your points with examples or analogies that relate to the user’s query.
                5. Additional Insights: Offer any further insights or suggestions related to the topic. If the user’s question can benefit from an extended explanation, provide it here.
                6. Conclusion: Summarize the response, reiterating key points or providing a call to action (e.g., "Let me know if you need more information").
                7. Invite the user to ask more questions or clarify their request further.

        """


        # Use PromptTemplate to structure the input
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template
        )
        
        # Initialize the model
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.5,
            convert_system_message_to_human=True
        )
        
        # Create a QA chain with the retriever and the model
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        
        
        # Format the prompt using the retrieved context and user's question
        formatted_prompt = prompt.format(context=context, question=query)
        
        # Run the chain with the formatted prompt
        result = qa_chain({"query": formatted_prompt})
        
        # Separate the answer and source documents
        answer = result['result']
        source_docs = result['source_documents']
        
        return answer, source_docs
    except Exception as e:
        print("Answer generation failed : ", e)
        return None



#st.title("HR Helpdesk Chatbot")
# docs = load_documents("./policy_workplace_concerns.pdf")
# docs = load_documents("./Handbook.pdf")
# # st.write(docs)
# vectorstore = create_vector_store(docs)
# # st.write(vectorstore)
# retriever = get_retriever(vectorstore)
# st.write(retriever)

st.markdown("""
<style>
    .stSidebar {
        background-color: #87ceeb !important;
        
    }
    .sidebarh1{
        border: 2px solid grey;
        border-radius : 12px;
        padding : 12px;
    }
    </style>

    """, unsafe_allow_html=True)

st.title(" ")
with st.sidebar.markdown("", unsafe_allow_html=True):
    st.image('./images/logo.png', width = 300)
upload_file=st.file_uploader('Choose your PDF file', type='pdf')
if st.button('Upload'):
    if upload_file:
        insert_document_into_db(upload_file)
    else:
        st.error('Please select a file to upload')    
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

query = st.chat_input("Hi How can I help you?")
retriever = create_faiss_retriever_from_db()

if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role":"user", "content":query})
    answer, source_docs = generate_answer(query, retriever)
    # response = f"Echo {query}"
    with st.chat_message('assistant'):
        st.markdown(answer)
    st.session_state.messages.append({"role":"assistant","content":answer})
