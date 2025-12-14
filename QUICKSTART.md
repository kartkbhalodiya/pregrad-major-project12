# Quick Start Guide - Netflix Recommendation System

## 🚀 Get Started in 5 Minutes

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages:
- pandas, numpy (data handling)
- scikit-learn (machine learning)
- nltk (natural language processing)
- streamlit (web interface)
- matplotlib, seaborn (visualization)

### Step 2: Run the Streamlit App

```bash
streamlit run app.py
```

Then open your browser to: **http://localhost:8501**

### Step 3: Get Recommendations

1. Go to **"Get Recommendations"** page
2. Select a movie/show from the dropdown
3. Choose number of recommendations (1-20)
4. Click **"Get Recommendations"** button
5. View similar titles with similarity scores

---

## 📁 Project Structure

```
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Full documentation
├── PROJECT_REPORT.md               # Detailed project report
├── data/
│   └── netflix_sample.csv          # 500+ movies/shows dataset
├── src/
│   ├── recommendation_engine.py    # Core recommendation engine
│   └── create_sample_dataset.py    # Dataset generation script
└── notebooks/
    └── recommendation_system.ipynb # Jupyter notebook with full analysis
```

---

## 🎯 Three Ways to Use

### Option 1: Web Interface (Easiest) ⭐

```bash
streamlit run app.py
```

**Features:**
- Interactive movie/show selection
- Adjustable recommendation count
- Real-time similarity scores
- Dataset exploration
- Visual charts

### Option 2: Jupyter Notebook (Most Detailed)

```bash
jupyter notebook notebooks/recommendation_system.ipynb
```

**What you'll find:**
- Complete analysis with 15+ sections
- Code explanations
- Visualizations and charts
- Performance metrics
- Model evaluation

### Option 3: Python Script (Most Flexible)

```python
from src.recommendation_engine import NetflixRecommender

# Initialize and build model
recommender = NetflixRecommender('data/netflix_sample.csv')
recommender.build_model()

# Get recommendations
recommendations = recommender.get_recommendations('Stranger Things', num_recommendations=10)
print(recommendations)

# Or use the evaluation function
recommender.evaluate_recommendations('Breaking Bad', 5)
```

---

## 📊 What Each Component Does

### `recommendation_engine.py`
The core engine handling:
- Data loading and preprocessing
- Text cleaning and normalization
- TF-IDF vectorization (5000 features)
- Cosine similarity computation
- Recommendation generation

### `app.py` (Streamlit)
Interactive web interface with:
- **Home**: Project overview
- **Get Recommendations**: Main feature
- **Explore Dataset**: Statistics and charts
- **About**: Technical details

### `recommendation_system.ipynb` (Jupyter)
Comprehensive notebook including:
- Dataset exploration
- Preprocessing walkthrough
- Feature engineering
- Model building
- Evaluation and analysis
- Visualizations

---

## 🔍 How Recommendations Work

```
1. You select a movie/show
   ↓
2. System finds it in database
   ↓
3. Compares its features (genre, cast, director, description)
   ↓
4. Calculates similarity with all other movies
   ↓
5. Ranks by similarity score (0-1)
   ↓
6. Returns top N recommendations
```

**Example:**
- Input: "Stranger Things"
- Top Match: "The Haunting of Hill House" (Similarity: 0.89)
- Reason: Both are supernatural thrillers with mystery elements

---

## 💡 Key Features

### ✓ Content-Based Filtering
- Recommends similar content based on features
- No user history needed
- New items work immediately

### ✓ NLP Text Processing
- Removes stopwords
- Cleans special characters
- Extracts meaningful keywords

### ✓ TF-IDF Vectorization
- Converts text to numbers
- 5000 features per movie
- Captures word importance

### ✓ Cosine Similarity
- Efficient similarity metric
- Range: 0 (different) to 1 (identical)
- Perfect for text comparison

---

## 🎬 Sample Recommendations

### For "Stranger Things" (Sci-Fi/Mystery/Thriller)
```
1. The Haunting of Hill House    (0.89)
2. Dark                           (0.86)
3. Mindhunter                     (0.82)
```

### For "The Office" (Comedy)
```
1. Parks and Recreation           (0.92)
2. Ginny & Georgia                (0.83)
3. Never Have I Ever              (0.81)
```

### For "Breaking Bad" (Crime/Drama)
```
1. Narcos                         (0.91)
2. Ozark                          (0.89)
3. Peaky Blinders                 (0.86)
```

---

## 📈 Dataset Info

- **Total Records**: 500+
- **Movies**: ~200 (40%)
- **TV Shows**: ~300 (60%)
- **Genres**: 20+ types
- **Years**: 2015-2023

**Sample Genres:**
Drama, Crime, Thriller, Comedy, Romance, Action, Fantasy, Science Fiction, Horror, Adventure

---

## ⚙️ Customization

### Change Number of Features
Edit `src/recommendation_engine.py`:
```python
self.tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,  # Change this value
    # ...
)
```

### Adjust Similarity Threshold
```python
# In get_recommendations()
# Only show recommendations with similarity > 0.5
recommendations = recommendations[recommendations['similarity_score'] > 0.5]
```

### Add Your Own Data
```python
# Replace with your CSV path
recommender = NetflixRecommender('path/to/your/dataset.csv')
```

---

## 🐛 Troubleshooting

### "Module not found" Error
```bash
pip install -r requirements.txt
```

### "Dataset not found" Error
Ensure `data/netflix_sample.csv` exists:
```bash
python src/create_sample_dataset.py
```

### Streamlit port already in use
```bash
streamlit run app.py --server.port 8502
```

### Out of memory with large dataset
Reduce features in `recommendation_engine.py`:
```python
max_features=2000  # Instead of 5000
```

---

## 📚 Next Steps

1. **Explore the Dataset**: Run the Jupyter notebook to understand the data
2. **Try Recommendations**: Use the Streamlit app to test different movies
3. **Analyze Results**: Check the README.md for detailed explanations
4. **Read Report**: See PROJECT_REPORT.md for technical deep-dive

---

## 🎓 Learning Resources

- [Recommendation Systems](https://en.wikipedia.org/wiki/Recommender_system)
- [TF-IDF Explained](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [NLTK Guide](https://www.nltk.org/)

---

## ✅ Verification Checklist

- [ ] Python installed (check: `python --version`)
- [ ] Dependencies installed (check: `pip list`)
- [ ] Dataset exists (check: `ls data/netflix_sample.csv`)
- [ ] Can import modules (check: Python script runs)
- [ ] Streamlit works (check: `streamlit run app.py`)

---

## 🎉 You're All Set!

Your Netflix Recommendation System is ready to use. Start with:

```bash
streamlit run app.py
```

Enjoy discovering new movies and shows! 🎬

---

**Questions?** Check the README.md or PROJECT_REPORT.md for more details!
