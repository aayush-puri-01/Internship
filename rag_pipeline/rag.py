from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
import os
from rag_pipeline.reranker import QueryBasedReranker

class RAGPipeline():
    def __init__(self, llm_model:str = "deepseek-r1:1.5b", persist_dir = "rag_pipeline/rag_chroma_db", use_reranker: bool = True, one_liner: bool = True):
        self.llm_model = llm_model

        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        self.llm = OllamaLLM(model=self.llm_model, callbacks=callback_manager)

        self.vectorstore = None
        self.persist_directory = persist_dir

        self.use_reranker = use_reranker

        self.one_liner = one_liner

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
    
    def reranker_build_and_respond(self, reranked_chunks, prompt_w_context, user_question):
        combine_docs_chain = create_stuff_documents_chain(
            llm = self.llm,
            prompt = prompt_w_context
        )
        response = combine_docs_chain.invoke(
            {
                "context" : reranked_chunks,
                "input" : user_question
            }
        )


    
    def run_query(self, user_query:str, retrieval_chain):
        response = retrieval_chain.invoke({
            "input": user_query
        },
        stream = True)
        return response["answer"]
    

    
    def respond_one_liner(self, one_liner_prompt_template):
        prompt_template = self.load_prompt_template(
            one_liner_prompt_template,
        )

        retrieval_chain = pipeline.build_chain(prompt_template)
        
        answer = pipeline.run_query(
            user_query=user_question,
            retrieval_chain=retrieval_chain)
        
        return answer
            

    
    def respond_paragraph(self, paragraph_prompt_template):
        prompt_template = self.load_prompt_template(
            paragraph_prompt_template,
        )

        retrieval_chain = pipeline.build_chain(prompt_template)
        
        answer = pipeline.run_query(
            user_query=user_question,
            retrieval_chain=retrieval_chain)
            
        return answer
    
if __name__ == "__main__":
    pipeline = RAGPipeline()
    pdf_text = pipeline.load_pdf_doc("Attention.pdf")
    chunks = pipeline.split_document(pdf_text)
    pipeline.get_chunk_embeddings(chunks) # create the vector database

    user_question = "How does positional embedding work?"

    if pipeline.use_reranker == False:

        if pipeline.one_liner == True:
            pipeline.respond_one_liner("prompt_templates_oneline.txt")

        else:
            pipeline.respond_paragraph("prompt_templates.txt")
           

    else: 

        reranker = QueryBasedReranker()

        retriever = pipeline.initialize_retriever(k=5)

        chunks = retriever.invoke(user_question)

        queries_for_chunks, chunks = reranker.generate_chunk_based_queries(chunks)

        question_embedding, chunk_queries_embeddings = reranker.generate_embeddings(user_question, queries_for_chunks)

        scores = reranker.calculate_similarity_scores(question_embedding, chunk_queries_embeddings)

        reranked_chunks = reranker.rerank_chunks(scores, chunks, top_k = 3)

        if pipeline.one_liner == True:
            prompt_template = pipeline.load_prompt_template(
                "prompt_templates_oneline.txt",
            )
        else:
            prompt_template = pipeline.load_prompt_template(
                "prompt_templates.txt",
            )           
        final_answer = pipeline.reranker_build_and_respond(reranked_chunks, prompt_template, user_question=user_question)
        print(final_answer)
