from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
import os

class RAGPipeline():
    def __init__(self, llm_model:str = "deepseek-r1:1.5b", persist_dir = "rag_chroma_db"):
        self.llm_model = llm_model
        self.llm = OllamaLLM(model=self.llm_model)
        self.vectorstore = None
        self.persist_directory = persist_dir

    def load_pdf_doc(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        full_text = " ".join([doc.page_content for doc in documents])
        return full_text

    def split_document(self, document_text):
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500, 
        chunk_overlap = 50
        )

        chunks = text_splitter.split_text(document_text)

        return chunks

    def get_chunk_embeddings(self, docchunks):
        if os.path.exists(self.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=OllamaEmbeddings(model="nomic-embed-text")
            )
            print("\n[INFO]: Loading existing vector database")
        else:
            embedding_model = OllamaEmbeddings(model="nomic-embed-text")

            self.vectorstore = Chroma.from_texts(
                texts=docchunks,
                embedding = embedding_model,
                persist_directory=self.persist_directory
            )
            self.vectorstore.persist()
            print("\n[INFO]: Creating new vector database")

    def initialize_retriever(self, k:int = 3):
        retriever = self.vectorstore.as_retriever(
            search_type = "similarity",
            search_kwargs={"k": k}
        )
        return retriever

    def load_prompt_template(self, prompt_file):
        with open(prompt_file, 'r') as f:
            entire_text = f.read()
            template_str = entire_text.strip()

        # prompts = entire_text.split('---')

        # prompt_dict = {}
        # current_key = None
        # current_text = []

        # for line in prompts: 
        #     stripped_line = line.strip()
        #     if stripped_line.startswith("# Prompt"):
        #         if current_key and current_text:
        #             prompt_dict[current_key] = "\n".join(current_text).strip() 
        #             #save the earlier key and text to the dictionary, new # Prompt encountered, so, set new prompt key and current_text
        #         current_key = stripped_line.split(':')[1].strip()
        #         current_text = []
        #     elif current_key:
        #         current_text.append(stripped_line)

        # if current_key and current_text:
        #     prompt_dict[current_key] = "\n".join(current_text).strip()

        # if prompt_key not in prompt_dict:
        #     raise ValueError(f"The provided prompt key {prompt_key} not found in {prompt_file}")
        
        single_prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=template_str
        )

        return single_prompt_template


    def build_chain(self, prompt_w_context):
        retriever = self.initialize_retriever(k=3)
        combine_docs_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt_w_context
        )
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        return retrieval_chain
    
    def run_query(self, user_query:str, retrieval_chain):
        response = retrieval_chain.invoke({
            "input": user_query
        })
        return response["answer"]
    
if __name__ == "__main__":
    pipeline = RAGPipeline()
    pdf_text = pipeline.load_pdf_doc("Attention.pdf")
    chunks = pipeline.split_document(pdf_text)
    pipeline.get_chunk_embeddings(chunks) # create the vector database

    prompt_template = pipeline.load_prompt_template(
        "prompt_templates.txt",
    )

    retrieval_chain = pipeline.build_chain(prompt_template)

    user_question = "why does a simple mechanism like self-attention work so well?"
    
    answer = pipeline.run_query(
        user_query=user_question,
        retrieval_chain=retrieval_chain)
    
    print(answer)
