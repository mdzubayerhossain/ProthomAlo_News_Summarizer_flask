# Prothom Alo News Summarizer

## Project Structure
```
news_app/
│
├── app.py               # Flask backend
├── requirements.txt     # Python dependencies
│
└── templates/
    └── index.html       # Frontend HTML
```

## Setup Instructions

1. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

2. Install dependencies
```bash
pip install flask flask-cors requests beautifulsoup4 torch transformers scikit-learn
```

3. Run the application
```bash
python app.py
```

## Notes
- Application will be available at `http://localhost:5000`
- Automatically scrapes and summarizes news every 10 minutes
- Requires an active internet connection
```
