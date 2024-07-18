import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
import pickle
import os
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain


#side contents
with st.sidebar:
    st.logo("icons/forgelogo_1.png")
    # st.title('Forge_GPT')
    # st.image("icons/forgelogo_1.png", caption='FORGEGPT')
    st.markdown('''
    # FORGEGPT
    - # [Forge Innovation & Ventures](https://www.forge-iv.co/)
    - # [Forge Incubation](https://www.forge-iv.co/incubation)
    - # [Protosem](https://www.protosem.tech/) 
''')
add_vertical_space(5)

st.logo("icons/forgelogo_1.png")
# st.image("icons/forgelogo_1.png")
st.header('Made in FORGE ❤️ Made for FORGE')

def main():
    st.header("Chat with PDF 💬")

    #Upload a PDF File
    pdf = st.file_uploader("Upload your PDF",type='pdf')
    st.write(pdf.name)
    #st.write(pdf)

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        st.write(pdf_reader)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        # st.write(chunks)

        #Embeddings
       
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
           # st.write('Embeddings loaded from the disk')
        else:
            embeddings = OllamaEmbeddings(model ="llama3")
            VectorStore = FAISS.from_texts(chunks, embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        #Accept user question
        query = st.chat_input("Ask Questions about the PDF file: ")
        # query = st.text_input("Ask Questions about the PDF file: ")
        st.write(query)

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = Ollama(model="llama3")
            chain =load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            # response = chain.invoke({"input_documents":docs, "question":query})
            st.write(response)
            # text = st.text_area(response)

            # print(f"Title:{docs.metadata['title']}, Source:{docs.metadata['source']}")

            # st.write(docs)
            #st.write('Embeddings Computation Completed')
        # st.write(text)

   
    
if __name__ == '__main__':
    main()
