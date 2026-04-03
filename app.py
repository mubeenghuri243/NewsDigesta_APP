from flask import Flask , render_template ,  request, jsonify, send_from_directory , session
from src.embeddings import EmbeddingPipeline 
from src.config import SECRET_KEY
# from src.news_fetcher import fetch_news
from src.retriever import RAGRetriever
from src.llm import generate_response
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
import os
from src.embeddings import collection
import plotly.express as px
import math 


model=SentenceTransformer("BAAI/bge-small-en-v1.5")
print("[DEBUG] Model loaded ONCE")


app=Flask(__name__)

app.secret_key=SECRET_KEY
pipeline=EmbeddingPipeline(model=model)
fetch_doc=RAGRetriever(pipeline)
articles_store = {}  
def get_trending_chart(query):
    results=collection.get()
    date_counts={}
    for metadata in results["metadatas"]:
        if metadata.get('query')==query:
            pub_date=metadata.get('publishedAt')[:10]
            date_counts[pub_date]=date_counts.get(pub_date,0)+1
        
    sorted_date=sorted(date_counts.keys())
    counts=[date_counts[date] for date in sorted_date]
    fig=px.line(x=sorted_date, y=counts, title=f"New Trends:{query}", labels={'x':'Date', 'y':'Articles'})
    chart_html=fig.to_html(full_html=False)
    return chart_html
def get_articles_from_chromadb(query):
    """ChromaDB se articles nikalo — query ke mutabiq"""
    results = collection.get()
    articles = []
    seen = set()

    for i, metadata in enumerate(results["metadatas"]):
        if metadata.get("query") == query:
            doc_content = results["documents"][i]
            # Duplicate avoid karo
            content_key = doc_content[:100]
            if content_key in seen:
                continue
            seen.add(content_key)
            articles.append(Document(
                page_content=doc_content,
                metadata=metadata
            ))
    return articles

    
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/search", methods=["GET", "POST"])
def search():
    if request.method=="POST":
        query=request.form.get("query")
    else:
        query=request.args.get("query")
        
    if not query:
        return render_template("index.html" , error="Enter a query ")
    session['last_query']=query
    page=request.args.get('page', 1 , type=int)
    per_page=10
    try:
        
        articles=get_articles_from_chromadb(query)
       
        articles_store[query]=[
            {
               "content":a.metadata.get('full_content', a.page_content),
               "source":a.metadata.get('source_name',''),
               "author":a.metadata.get('author',''),
               "url":a.metadata.get('url',''),
               "date":a.metadata.get('publishedAt',''),
               "sentiment":a.metadata.get('sentiment','')
            }
            for a in articles
        ]
        session['current_query']=query
        
        if not articles:
            return render_template("result.html", query=query, answer="No news found", articles=[])
        start=(page-1) * per_page
        end=page * per_page
        paginated_articles=articles[start:end]
        total_pages=math.ceil(len(articles)/per_page )
        
        response=generate_response(query , fetch_doc)
        chart_html=get_trending_chart(query)
        return render_template("result.html", query=query, answer=response, articles=paginated_articles, total_pages=total_pages, page=page, chart_html=chart_html)
            
    except Exception as e:
            return f"Error:{str(e)}"


@app.route("/article/<int:index>", methods=["GET","POST"])
def article_page(index):
    query=session.get("current_query", '')
    articles=articles_store.get(query,[])
    
    if index<len(articles):
        article=articles[index]
        session['article_index']=index
        session['article_query']=query
        answer = None
        question = None
        if request.method == "POST":
            question = request.form.get("question", "")
            context = f"Answer based on this article only:\n{article['content'][:3000]}"  # ← [:3000] fix
            answer = generate_response(question, fetch_doc, context)
        
        return render_template("article.html", article=article, article_index=index, question=question, answer=answer)
    return "Not found"
       
    
@app.route("/suggestions", methods=["GET"])
def suggestions():
    q=request.args.get('q')
    if not q:
        return jsonify([])

    q_lower = q.lower()
    default_suggestions = [
        "AI Technology",
        "Iran War",
    ]

    results = collection.get()
    suggestions = set()

    for metadata in results.get("metadatas", []):
        query = metadata.get("query")
        if not query:
            continue
        if q_lower in query.lower():
            suggestions.add(query)
            
    if not suggestions:
        for item in default_suggestions:
            if q_lower in item.lower():
                suggestions.add(item)

    return jsonify(sorted(suggestions))       
            
            
        
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')

@app.route("/qa_page", methods=["GET" , "POST"])
def qa_page():
    if request.method=="POST":
        question=request.form.get("question")
        last_query=session.get('last_query','')
        article_index=session.get("article_index")
        article_query=session.get("article_query" , '')
        article_context=''
        if  article_index is not None:
            arts=articles_store.get(article_query,[])
            if article_index<len(arts):
                article_context=arts[article_index]['content']
            
        if article_context:
            context = f"Answer based on this article only:\n{article_context}"
        else:
            context = f"User recently searched about: {last_query}"
        answer=generate_response(question, fetch_doc, context)
        return render_template("qa.html", question=question, answer=answer  )
    return render_template("qa.html")



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

        
        
    


    
    
