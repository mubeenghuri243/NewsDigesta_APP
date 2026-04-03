from src.embeddings import EmbeddingPipeline
from typing import List , Any
from src.embeddings import collection
class RAGRetriever:
    
    def __init__(self,embedding_pipeline:EmbeddingPipeline ):
        self.embedding_pipline=embedding_pipeline
        
        pass
    def retriever(self, query:str, top_k:int=5, score_threshold:float=0.0) -> List[dict[str , Any]]:
        print(f"Retrivel documents for query: {query}")
        print(f"Top_k:{top_k}, Score_Threshold:{score_threshold}")
        
        # Generate Embedding query
        query_embeddings=self.embedding_pipline.embed_query(query)
        
        try:
            results =collection.query(
                query_embeddings=[query_embeddings.tolist()],
                n_results=top_k
            )
            retrieved_doc=[]
            if results['documents'] and results['documents'][0]:
                documents=results['documents'][0]
                metadatas=results['metadatas'][0]
                distances=results['distances'][0]
                ids=results['ids'][0]
                for i,(doc_id, document, metadata, distance) in enumerate(zip(ids,documents, metadatas, distances)):
                    similarity_score=1/(1+distance)
                    if similarity_score >= score_threshold:
                        retrieved_doc.append(
                            {
                                "id":doc_id,
                                "content":document,
                                "metadata":metadata,
                                "distance":distance,
                                "similarity_score":similarity_score,
                                "rank":i+1
                            }
                        )
                print(f"Retrieved {len(retrieved_doc)} documents after filtering" )
            else:
                print("No documents retrieved")
        except Exception as e:
            print(f"error during retrieved:{e}") 
            raise
        return retrieved_doc 
    

