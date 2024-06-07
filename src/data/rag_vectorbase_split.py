import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


CHUNK_SIZE = 2000
CHUNK_OVERLAP=100

class TextVBLoader():
    def __init__(self, transcripts_list_punct):
        self.transcripts_list_punct = transcripts_list_punct
        
        
    def get_splited_transcripts(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        docs = [Document(page_content=text) for text in self.transcripts_list_punct]
        docs_split = text_splitter.split_documents(docs)
        
        return docs_split 
    
    def load_docs_vb(self, docs_split):
            
        model_emb = "sentence-transformers/distiluse-base-multilingual-cased-v1"
        model_kwargs = {"device":"cuda" if torch.cuda.is_available() else "cpu"}
        encode_kwargs = {"normalize_embeddings": False}

        embeddings = HuggingFaceEmbeddings(model_name = model_emb,
                                        model_kwargs = model_kwargs,
                                        encode_kwargs = encode_kwargs)

        # model_emb = "YandexGPTEmbeddings"
        save_db = '../../data/interim/vectorstore/divertito_db_CHUNK_SIZE{}CHUNK_OVERLAP{}_{}'.format(CHUNK_SIZE,
                                                                                                CHUNK_OVERLAP,
                                                                                                model_emb)
        try:
            print("Found existing index")
            db = FAISS.load_local(save_db, embeddings)
        except RuntimeError:
            db = FAISS.from_documents(docs_split,embeddings)
            db.save_local(save_db)
            
        return db
    
    def load_db(self):
        docs_split = self.get_splited_transcripts()
        db = self.load_docs_vb(docs_split)
        return db