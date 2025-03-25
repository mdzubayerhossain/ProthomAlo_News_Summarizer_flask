import os
import requests
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from bs4 import BeautifulSoup
import re
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

class ProthomAloSummarizer:
    def __init__(self):
        # Load multilingual BERT model for summarization
        model_name = "bert-base-multilingual-cased"
        print("Loading BERT model...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        print("BERT model loaded successfully")

        # Scraping configuration
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,bn;q=0.8",
        }
        self.base_url = "https://www.prothomalo.com"

    def clean_text(self, text):
        """Clean text by removing extra whitespace and newlines"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def split_into_sentences(self, text):
        """Split Bengali text into sentences"""
        # Bengali uses '।' as sentence delimiter
        sentences = re.split(r'[।!?]', text)
        return [s.strip() for s in sentences if s.strip()]

    def get_sentence_embeddings(self, sentences):
        """Get embeddings for a list of sentences using BERT"""
        embeddings = []
        
        for sentence in sentences:
            # Tokenize and convert to tensor
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use CLS token as sentence embedding
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(embedding[0])
        
        return np.array(embeddings)

    def summarize_with_rag(self, text, num_sentences=3):
        """
        Summarize text using RAG-inspired approach
        """
        if not text or len(text.strip()) == 0:
            return ""
        
        # Clean text
        text = self.clean_text(text)
        
        # Split into sentences
        sentences = self.split_into_sentences(text)
        
        # If text is already short, return as is
        if len(sentences) <= num_sentences:
            return text
        
        try:
            # Get sentence embeddings
            embeddings = self.get_sentence_embeddings(sentences)
            
            # Calculate sentence centrality (similarity to other sentences)
            similarity_matrix = cosine_similarity(embeddings)
            centrality_scores = np.sum(similarity_matrix, axis=1)
            
            # Get indices of top sentences by centrality
            top_indices = np.argsort(centrality_scores)[-num_sentences:]
            
            # Sort indices to maintain original order
            top_indices = sorted(top_indices)
            
            # Construct summary from selected sentences
            selected_sentences = [sentences[i] for i in top_indices]
            summary = '। '.join(selected_sentences) + '।'
            
            return summary
        except Exception as e:
            print(f"Error in summarization: {e}")
            return text[:200] + '...'  # Fallback to first 200 characters

    def scrape_news(self, max_articles=10):
        """Scrape latest news articles from Prothom Alo"""
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting news scrape...")
        
        try:
            # Fetch homepage
            response = requests.get(self.base_url, headers=self.headers)
            response.raise_for_status()
            
            # Parse homepage HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Categories to scrape
            categories = ['bangladesh', 'world', 'economy', 'sports', 'entertainment']
            
            # Collect articles
            scraped_articles = []
            
            for category in categories:
                # Find article links in each category
                category_selector = f'a[href*="/{category}/"]'
                category_links = soup.select(category_selector)
                
                for element in category_links:
                    href = element.get('href')
                    
                    # Skip video and gallery links
                    if href and '/video/' not in href and '/gallery/' not in href:
                        # Ensure full URL
                        if not href.startswith('http'):
                            href = self.base_url + href
                        
                        # Scrape individual article
                        article_details = self.scrape_article(href)
                        if article_details:
                            scraped_articles.append(article_details)
                    
                    # Limit to specified max articles
                    if len(scraped_articles) >= max_articles:
                        break
                
                if len(scraped_articles) >= max_articles:
                    break
            
            return scraped_articles
        
        except Exception as e:
            print(f"Error during scraping: {e}")
            return []

    def scrape_article(self, article_url):
        """Scrape details of a single article"""
        try:
            # Fetch article page
            article_response = requests.get(article_url, headers=self.headers)
            article_response.raise_for_status()
            
            # Parse article HTML
            article_soup = BeautifulSoup(article_response.text, 'html.parser')
            
            # Extract title
            title_element = article_soup.select_one('h1')
            title = self.clean_text(title_element.text) if title_element else "No title found"
            
            # Extract content
            content_elements = article_soup.select('.story-element-text')
            full_content = "\n\n".join([self.clean_text(element.get_text()) for element in content_elements if element.get_text()])
            
            # Generate RAG summary
            summary = self.summarize_with_rag(full_content) if full_content else "No summary available"
            
            # Extract image - Handle multiple patterns
            image_url = ""
            
            # Check for updated image structure (Prothom Alo image classes)
            image_element = article_soup.select_one('.qt-image.zoom-desktop img') or \
                            article_soup.select_one('figure img') or \
                            article_soup.select_one('img')
            
            if image_element:
                image_url = image_element.get('src') or image_element.get('data-src', '')
                
                # Ensure full URL
                if image_url and not image_url.startswith('http'):
                    image_url = self.base_url + image_url
            
            # Extract publication date
            date_element = article_soup.select_one('time')
            published_at = date_element.get('datetime') if date_element else ""
            
            # Construct article dictionary
            return {
                'title': title,
                'summary': summary,
                'image_url': image_url,
                'article_url': article_url,
                'published_at': published_at
            }
        
        except Exception as e:
            print(f"Error scraping article {article_url}: {e}")
            return None

# Create Flask App
app = Flask(__name__)
CORS(app)

# Initialize summarizer
summarizer = ProthomAloSummarizer()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/news')
def get_news():
    """API endpoint to fetch news articles"""
    try:
        # Scrape news articles
        articles = summarizer.scrape_news(max_articles=10)
        
        return jsonify({
            'status': 'success',
            'articles': articles,
            'total_articles': len(articles)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)