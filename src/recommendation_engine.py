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
            # Try UTF-8 first, fallback to latin-1 if encoding error
            try:
                self.df = pd.read_csv(data_path, encoding='utf-8')
            except UnicodeDecodeError:
                self.df = pd.read_csv(data_path, encoding='latin-1')
            
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
        
        # Handle different column names based on dataset
        if 'Genre' in self.df.columns:
            self.df['listed_in'] = self.df['Genre'].fillna('Unknown')
        elif 'listed_in' not in self.df.columns:
            self.df['listed_in'] = 'Unknown'
        else:
            self.df['listed_in'] = self.df['listed_in'].fillna('Unknown')
        
        if 'Actors' in self.df.columns:
            self.df['cast'] = self.df['Actors'].fillna('Unknown')
        elif 'cast' not in self.df.columns:
            self.df['cast'] = 'Unknown'
        else:
            self.df['cast'] = self.df['cast'].fillna('Unknown')
        
        if 'Director' in self.df.columns:
            self.df['director'] = self.df['Director'].fillna('Unknown')
        elif 'director' not in self.df.columns:
            self.df['director'] = 'Unknown'
        else:
            self.df['director'] = self.df['director'].fillna('Unknown')
        
        if 'Summary' in self.df.columns:
            self.df['description'] = self.df['Summary'].fillna('Unknown')
        elif 'description' not in self.df.columns:
            self.df['description'] = 'Unknown'
        else:
            self.df['description'] = self.df['description'].fillna('Unknown')
        
        # Handle title
        if 'Title' in self.df.columns and 'title' not in self.df.columns:
            self.df['title'] = self.df['Title']
        elif 'title' not in self.df.columns:
            self.df['title'] = 'Unknown'
        
        # Handle release year
        if 'Release Date' in self.df.columns and 'release_year' not in self.df.columns:
            try:
                self.df['release_year'] = pd.to_datetime(self.df['Release Date'], errors='coerce').dt.year
                self.df['release_year'] = self.df['release_year'].fillna(2020).astype(int)
            except:
                self.df['release_year'] = 2020
        elif 'release_year' not in self.df.columns:
            self.df['release_year'] = 2020
        
        # Handle type
        if 'Series or Movie' in self.df.columns and 'type' not in self.df.columns:
            self.df['type'] = self.df['Series or Movie']
        elif 'type' not in self.df.columns:
            self.df['type'] = 'Unknown'
        
        # Handle poster URL
        if 'Image' in self.df.columns and 'poster_url' not in self.df.columns:
            self.df['poster_url'] = self.df['Image'].fillna('')
        elif 'poster_url' not in self.df.columns:
            self.df['poster_url'] = ''
        
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
        
        # Use the standardized column names created in preprocessing
        output_cols = ['title', 'type', 'listed_in', 'description', 'release_year', 'poster_url']
        available_cols = [col for col in output_cols if col in self.df.columns]
        
        recommendations = self.df.iloc[similar_indices][available_cols].copy()
        recommendations['similarity_score'] = similarity_scores[similar_indices]
        recommendations['similarity_score'] = recommendations['similarity_score'].fillna(0)
        recommendations = recommendations.reset_index(drop=True)
        
        return recommendations
    
    def get_recommendations_by_tag(self, tag, num_recommendations=10):
        """
        Get top N recommendations for a given tag/genre.
        
        Args:
            tag (str): Genre/tag to get recommendations for
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            pd.DataFrame: DataFrame with recommended movies matching the tag
        """
        if self.similarity_matrix is None:
            print("Error: Similarity matrix not computed. Run compute_similarity() first.")
            return None
        
        movies_with_tag = self.df[self.df['listed_in'].str.contains(tag, case=False, na=False)]
        
        if len(movies_with_tag) == 0:
            print(f"No movies found with tag '{tag}'")
            return None
        
        movies_with_tag_indices = movies_with_tag.index.tolist()
        
        similarity_scores_dict = {}
        for idx in movies_with_tag_indices:
            similarity_scores = self.similarity_matrix[idx]
            similar_indices = similarity_scores.argsort()[::-1][1:num_recommendations + 1]
            for sim_idx in similar_indices:
                if sim_idx not in similarity_scores_dict:
                    similarity_scores_dict[sim_idx] = similarity_scores[sim_idx]
                else:
                    similarity_scores_dict[sim_idx] = max(similarity_scores_dict[sim_idx], similarity_scores[sim_idx])
        
        sorted_indices = sorted(similarity_scores_dict.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
        similar_indices = [idx for idx, _ in sorted_indices]
        
        output_cols = ['title', 'type', 'listed_in', 'description', 'release_year', 'poster_url']
        available_cols = [col for col in output_cols if col in self.df.columns]
        
        recommendations = self.df.iloc[similar_indices][available_cols].copy()
        recommendations['similarity_score'] = [score for _, score in sorted_indices]
        recommendations['similarity_score'] = recommendations['similarity_score'].fillna(0)
        recommendations = recommendations.reset_index(drop=True)
        
        return recommendations
    
    def get_recommendations_by_multiple_tags(self, tags_list, num_recommendations=10):
        """
        Get top N recommendations for multiple tags/genres.
        
        Args:
            tags_list (list): List of genres/tags to get recommendations for
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            pd.DataFrame: DataFrame with recommended movies matching any of the tags
        """
        if self.similarity_matrix is None:
            print("Error: Similarity matrix not computed. Run compute_similarity() first.")
            return None
        
        if not tags_list or len(tags_list) == 0:
            print("Error: No tags provided")
            return None
        
        all_movies_with_tags = pd.DataFrame()
        for tag in tags_list:
            movies_with_tag = self.df[self.df['listed_in'].str.contains(tag, case=False, na=False)]
            all_movies_with_tags = pd.concat([all_movies_with_tags, movies_with_tag]).drop_duplicates()
        
        if len(all_movies_with_tags) == 0:
            print(f"No movies found with tags {tags_list}")
            return None
        
        movies_with_tags_indices = all_movies_with_tags.index.tolist()
        
        similarity_scores_dict = {}
        for idx in movies_with_tags_indices:
            similarity_scores = self.similarity_matrix[idx]
            similar_indices = similarity_scores.argsort()[::-1][1:num_recommendations + 1]
            for sim_idx in similar_indices:
                if sim_idx not in similarity_scores_dict:
                    similarity_scores_dict[sim_idx] = similarity_scores[sim_idx]
                else:
                    similarity_scores_dict[sim_idx] = max(similarity_scores_dict[sim_idx], similarity_scores[sim_idx])
        
        sorted_indices = sorted(similarity_scores_dict.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
        similar_indices = [idx for idx, _ in sorted_indices]
        
        output_cols = ['title', 'type', 'listed_in', 'description', 'release_year', 'poster_url']
        available_cols = [col for col in output_cols if col in self.df.columns]
        
        recommendations = self.df.iloc[similar_indices][available_cols].copy()
        recommendations['similarity_score'] = [score for _, score in sorted_indices]
        recommendations['similarity_score'] = recommendations['similarity_score'].fillna(0)
        recommendations = recommendations.reset_index(drop=True)
        
        return recommendations
    
    def get_all_tags(self):
        """
        Get all unique tags/genres from the dataset.
        
        Returns:
            list: Sorted list of unique tags
        """
        all_tags = set()
        for tags_str in self.df['listed_in'].dropna():
            tags = [tag.strip() for tag in str(tags_str).split(',')]
            all_tags.update(tags)
        
        return sorted(list(all_tags))
    
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
