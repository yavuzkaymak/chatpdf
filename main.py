from typing import List
from langchain_community.vectorstores.supabase import SupabaseVectorStore
from langchain_core.vectorstores.base import VectorStoreRetriever
from streamlit.runtime.uploaded_file_manager import UploadedFile
import streamlit as st
from vector_store.supabase import SupabaseConnection
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from llama_cloud_services import LlamaParse
from llama_cloud_services.parse.utils import ResultType
from llama_index.core.schema import Document  as LlamaDocument
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from config import OllamaSettings

def main():
  
    llm = OllamaLLM(
        base_url=OllamaSettings.URL,
        model = OllamaSettings.CHAT_MODEL,
        verbose=True,
    )   
  
    embeddings = OllamaEmbeddings(
        base_url = OllamaSettings.URL,
        model = OllamaSettings.EMBEDDINGS_MODEL
    )

    vector_store: SupabaseVectorStore = SupabaseConnection(embedding=embeddings).vector_store # type: ignore
    retriever: VectorStoreRetriever = vector_store.as_retriever()

    system_prompt = (
        "You are an assistant to a stock analyst for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Your answers should be comprehensive and "
        "giving the context."
        "\n\n"
        "{context}"
    )

    question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, question_prompt) 

    qa_chain = create_stuff_documents_chain(llm, question_prompt)

    convo_qa_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
 
    st.set_page_config(page_title='PDF Analyst', page_icon='📦', )
    st.header('Ask the Analyst about your PDF ✨', anchor='top')

    parser = LlamaParse(result_type=ResultType.MD)

    pdf: UploadedFile | None = st.file_uploader('Upload a PDF file', type='pdf')       

    if pdf:
        docs: List[LlamaDocument] = parser.load_data(file_path=pdf.read(), extra_info={"file_name": f"{pdf.name}"})   
        lang_docs = [doc.to_langchain_format() for doc in docs]
        vector_store.add_documents(lang_docs)  

    question = st.text_input('Ask a question about the PDF', max_chars=1000)
    if question:
        
        response  =  convo_qa_chain.invoke(
            {
                "input": question,
                "chat_history": [],
            }
        )  
        print(response)       
        st.write(response["answer"])    

if __name__ == '__main__':
    main()    