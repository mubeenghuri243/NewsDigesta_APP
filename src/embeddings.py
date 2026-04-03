from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb 
import hashlib
import numpy as np 
from typing import List, Any
client = chromadb.PersistentClient(path="./chroma_db")

collection=client.get_or_create_collection("News_articles")
class EmbeddingPipeline:
    def __init__(self , model, chunk_size:int=1000 ,chunk_overlap:int=200):
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.model=model
        
        pass
    # Text splitting getting into chunks 
    def chunk_articles(self,articles:List[Any])->List[Any]:
        
        spliter=RecursiveCharacterTextSplitter(
        
        chunk_size=self.chunk_size,
        chunk_overlap=self.chunk_overlap,
        length_function=len,
        separators=["\n\n","\n"," ",""]
       )
        chunks=spliter.split_documents(articles)
        print(f"[INFO] split: {len(articles)} into: {len(chunks)} chunks")
        return chunks
        
    def embed_chunks(self, chunks:List[Any])->np.ndarray:
        text=[chunk.page_content for chunk in chunks]
        print(f"[INF] Genarte Embedding for :{len(text)} chunks")
        embeddings=self.model.encode(text,show_progress_bar=True)
        print(f"[INFO] embedding shape:{embeddings.shape}")
        return embeddings
    def embed_query(self,query:str):
        return self.model.encode(query)
    def store_in_chromadb(self,query, articles:List[Any] , embeddings:np.ndarray):
        if len(articles)!= len(embeddings):
            raise ValueError(f"number of the articles must match the number of the embed_chunks")
        print(f"adding articles:{len(articles)} for query:{query}")
        # prepare data for chromadb
        ids=[]
        metadatas=[]
        documents_text=[]
        embeddings_list=[]
        for i, (doc, embedding) in enumerate(zip(articles, embeddings)):
            # Generate unique uuid
            
            doc_id = hashlib.md5((str(i) + doc.metadata.get("source_name","") + doc.page_content).encode()).hexdigest()
            
            existing=collection.get(ids=[doc_id])
            if existing ['ids']:
                print("Dublipacte found , skipping..")
                continue
            ids.append(doc_id)
            #prepare metadata
            metadata=dict(doc.metadata)
            metadata={key:("" if value is None else value) for key , value in metadata.items()}
            metadata['doc_index']=i
            metadata['content_length']=len(doc.page_content)
            metadata['query']=query
            metadatas.append(metadata)
            
            
            # Documents content
            documents_text.append(doc.page_content)
            # Embeddings
            embeddings_list.append(embedding.tolist())
            #Try to add collection
        if ids:
            collection.upsert(
                ids=ids,
                metadatas=metadatas,
                embeddings=embeddings_list ,
                documents=documents_text,
            )
            print(f"Added {len(ids)} new docs")
        else:
            print("No new documents to add")
        
        
    
        
    
