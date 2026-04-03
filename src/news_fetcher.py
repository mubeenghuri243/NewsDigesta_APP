from src.config import NEWS_API_KEY 
from newsapi import NewsApiClient
newsapi = NewsApiClient(api_key=NEWS_API_KEY )
from langchain_core.documents import Document
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def fetch_news(query):
    
    all_articles=newsapi.get_everything(q=query,language='en')
    result_list=[]
    for article in all_articles["articles"]:
        if article["description"] is None or article["content"] is None:
            continue
        text=article['title'] + " " +article['description']
        sentiment=sentiment_score(text)
        doc=Document(
            page_content=article['title'] + " " +article['description']+ " " +article['content'],
            metadata={
                "source_name":article['source']['name'],
                "author":article['author'],
                "publishedAt":article['publishedAt'],
                "url":article['url'],
                "sentiment":sentiment
            }
        )
        result_list.append(doc)
    return result_list
        
def sentiment_score(article):
    sid_obj=SentimentIntensityAnalyzer()
    sentiment_dic=sid_obj.polarity_scores(article)
    print(f"Score:{sentiment_dic}")
    if sentiment_dic['compound'] >= 0.05:
        return "Positive"
    elif sentiment_dic["compound"] <=- 0.05:
        return "Negative"
    else:
        return "Neutral"
    
      
        
   