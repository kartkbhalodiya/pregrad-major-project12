import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords', quiet=True)


class NetflixRecommender:
    def __init__(self, data_path):
        """
        Initialize the Netflix Recommender system.
        
        Args:
            data_path (str): Path to the Netflix dataset CSV file
        """
        self.df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.load_data(data_path)
        
    def load_data(self, data_path):
        """Load and display basic information about the dataset."""
        try:
            self.df = pd.read_csv(data_path)
            print(f"Dataset loaded successfully!")
            print(f"Shape: {self.df.shape}")
            print(f"Columns: {self.df.columns.tolist()}")
        except FileNotFoundError:
            print(f"Error: Dataset not found at {data_path}")
            
    def clean_text(self, text):
        """
        Clean text data: lowercase, remove special characters, remove stopwords.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def preprocess_data(self):
        """
        Preprocess the dataset by handling missing values and combining features.
        """
        print("\n--- Data Preprocessing ---")
        
        print(f"Initial missing values:\n{self.df.isnull().sum()}")
        
        self.df['listed_in'] = self.df['listed_in'].fillna('Unknown')
        self.df['cast'] = self.df['cast'].fillna('Unknown')
        self.df['director'] = self.df['director'].fillna('Unknown')
        self.df['description'] = self.df['description'].fillna('Unknown')
        
        print("\nAfter filling missing values:")
        print(f"Missing values:\n{self.df.isnull().sum()}")
        
    def create_metadata_soup(self):
        """
        Create a 'metadata soup' by combining relevant features.
        This combines genre, cast, director, and description into a single string.
        """
        print("\n--- Feature Engineering ---")
        print("Creating metadata soup from: genre, cast, director, and description")
        
        self.df['metadata_soup'] = (
            self.df['listed_in'].fillna('') + ' ' +
            self.df['cast'].fillna('') + ' ' +
            self.df['director'].fillna('') + ' ' +
            self.df['description'].fillna('')
        )
        
        print("Cleaning metadata soup...")
        self.df['metadata_soup'] = self.df['metadata_soup'].apply(self.clean_text)
        
        print(f"Sample metadata soup:\n{self.df['metadata_soup'].iloc[0][:200]}...")
        
    def vectorize_features(self, max_features=5000):
        """
        Convert text data to TF-IDF vectors.
        
        Args:
            max_features (int): Maximum number of features to use
        """
        print(f"\n--- TF-IDF Vectorization ---")
        print(f"Vectorizing with max_features={max_features}")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['metadata_soup'])
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
    def compute_similarity(self):
        """Compute cosine similarity matrix between all movies."""
        print(f"\n--- Computing Similarity Matrix ---")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        print(f"Similarity matrix shape: {self.similarity_matrix.shape}")
        
    def get_recommendations(self, title, num_recommendations=10):
        """
        Get top N recommendations for a given movie title.
        
        Args:
            title (str): Movie/Show title to get recommendations for
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            pd.DataFrame: DataFrame with recommended movies
        """
        if self.similarity_matrix is None:
            print("Error: Similarity matrix not computed. Run compute_similarity() first.")
            return None
        
        movie_list = self.df[self.df['title'].str.contains(title, case=False, na=False)]
        
        if len(movie_list) == 0:
            print(f"No movies found matching '{title}'")
            return None
        
        movie_index = movie_list.index[0]
        
        similarity_scores = self.similarity_matrix[movie_index]
        similar_indices = similarity_scores.argsort()[::-1][1:num_recommendations + 1]
        
        recommendations = self.df.iloc[similar_indices][['title', 'type', 'listed_in', 'description', 'release_year']].copy()
        recommendations['similarity_score'] = similarity_scores[similar_indices]
        recommendations = recommendations.reset_index(drop=True)
        
        return recommendations
    
    def evaluate_recommendations(self, test_title, num_recommendations=5):
        """
        Evaluate and display recommendations for a given title.
        
        Args:
            test_title (str): Title to get recommendations for
            num_recommendations (int): Number of recommendations to show
        """
        print(f"\n--- Recommendation Results for '{test_title}' ---")
        
        recommendations = self.get_recommendations(test_title, num_recommendations)
        
        if recommendations is not None:
            print(f"\nTop {num_recommendations} Recommendations:")
            print(recommendations.to_string(index=True))
        else:
            print(f"Could not get recommendations for '{test_title}'")
    
    def build_model(self):
        """Build the complete recommendation model."""
        if self.df is None:
            print("Error: No data loaded. Please load data first.")
            return
        
        self.preprocess_data()
        self.create_metadata_soup()
        self.vectorize_features()
        self.compute_similarity()
        print("\n✓ Model built successfully!")
