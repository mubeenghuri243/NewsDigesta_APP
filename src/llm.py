from src.config import GROQ_API_KEY


from groq import Groq 
client=Groq(api_key=GROQ_API_KEY)
def generate_response(query,retriever, session_context="", top_k=3):
    
    
    
    results=retriever.retriever(query,top_k)
    print(results)
    # context = "\n\n".join([doc['content'] for doc in results]) if results else ""# her articles sy just content nikalo
    news_context="\n\n".join([
        f"{doc['metadata'].get('source_name' , 'unknown')}:\n{doc['content']}" for doc in results 
        ])
    
    if not news_context:
        return "No relevant context found."
    
    prompt = f"""You are a professional news analyst.Summarize the key points and answer clearly."Do not use markdown or asterisks.Plain text only.Keep your answer short — maximum 5 sentences only."
        sesssion_context:
        {session_context}
        news_context:
            {news_context}
        Question:
            {query}
        Answer:
    """
    response=client.chat.completions.create(model="llama-3.1-8b-instant", temperature=0.1,messages=[{"role": "system", "content": "You are a helpful news assistant..."},{"role": "user", "content": prompt}])
    answer=response.choices[0].message.content
    return answer
    
