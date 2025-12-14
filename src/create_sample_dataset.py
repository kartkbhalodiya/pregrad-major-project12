import pandas as pd
import numpy as np

def create_sample_netflix_dataset(output_path, num_records=500):
    """
    Create a sample Netflix dataset for demonstration purposes.
    
    Args:
        output_path (str): Path where to save the CSV file
        num_records (int): Number of records to generate
    """
    
    np.random.seed(42)
    
    titles = [
        "The Crown", "Stranger Things", "The Witcher", "Ozark", "Breaking Bad",
        "Narcos", "House of Cards", "Peaky Blinders", "The Office", "Parks and Recreation",
        "The Boys", "Squid Game", "Euphoria", "Mindhunter", "Dark",
        "The Mandalorian", "Wednesday", "Dahmer", "You", "Money Heist",
        "Chernobyl", "The Last of Us", "The White Lotus", "Succession", "Severance",
        "Stranger Things", "The Midnight Club", "Wednesday", "Monster", "Fear Street",
        "Ginny & Georgia", "Never Have I Ever", "Heartstopper", "Bridgerton", "Sweet Magnolias",
        "Grey's Anatomy", "The Good Doctor", "Outer Banks", "Manifest", "Elite",
        "Locke & Key", "The Haunting of Hill House", "13 Reasons Why", "All of Us Are Dead",
        "Hellbound", "It's Okay to Not Be Okay", "Crash Landing on You", "Descendants of the Sun",
    ]
    
    genres_list = [
        "Crime, Drama, Thriller", "Science Fiction, Drama, Thriller", "Fantasy, Adventure, Drama",
        "Crime, Drama, Thriller", "Crime, Drama, Thriller", "Crime, Drama, Thriller",
        "Politics, Drama", "Crime, Drama, History", "Comedy", "Comedy, Politics",
        "Action, Comedy, Crime", "Drama, Thriller, Horror", "Drama, Thriller", "Crime, Drama",
        "Science Fiction, Mystery, Thriller", "Action, Adventure, Science Fiction", "Comedy, Crime, Drama",
        "Crime, Drama", "Drama, Romance, Thriller", "Crime, Drama", "Drama, History, Thriller",
        "Drama, History", "Drama", "Comedy, Drama", "Drama, Thriller", "Drama, Science Fiction",
        "Horror, Science Fiction, Thriller", "Horror, Mystery, Thriller", "Drama, Horror, Thriller",
        "Horror, Mystery, Thriller", "Drama, Mystery, Thriller", "Comedy, Drama", "Comedy, Drama",
        "Comedy, Drama, Romance", "Drama, Romance", "Drama, Romance", "Drama, Medical",
        "Drama, Medical", "Adventure, Drama, Mystery", "Drama, Mystery, Thriller", "Crime, Drama",
        "Horror, Thriller", "Horror, Thriller", "Drama, Thriller", "Horror, Thriller",
        "Horror, Drama, Thriller", "Drama, Fantasy", "Drama, Romance, Fantasy", "Drama, Romance",
    ]
    
    directors = [
        "Peter Morgan", "Duffer Brothers", "Lauren Schmidt Hissrich", "Bill Dubuque", "Vince Gilligan",
        "Joe Sorrentino", "Beau Willimon", "Steven Knight", "Greg Daniels", "Greg Daniels",
        "Eric Kripke", "Hwang Dong-hyuk", "Sam Levinson", "David Fincher", "Baran bo Odar",
        "Jon Favreau", "Tim Burton", "Ryan Murphy", "Greg Berlanti", "Álex de la Iglesia",
        "Johan Renck", "Craig Mazin", "Mike White", "Jesse Armstrong", "Dan Erickson",
        "Duffer Brothers", "Mike Flanagan", "Tim Burton", "Ryô Murakawa", "Lee Jae-kyoo",
        "Deborah Cahn", "Mindy Kaling", "Alice Oseman", "Chris Van Dusen", "Aina Clotet",
        "Shonda Rhimes", "David Shore", "Jonas Pate", "Jeff Rake", "Carlos Montero",
        "Joe Hill", "Mike Flanagan", "Brian Yorkey", "Lee Jae-kyoo", "Yeon Sang-ho",
        "Park Jae-kyoo", "Lee Jae-kyoo", "Park Jae-kyoo", "Lee Eun-jin", "Lee Jae-kyoo",
    ]
    
    cast_list = [
        "Gillian Anderson, Tobias Menzies, Claire Foy", "Winona Ryder, David Harbour, Finn Wolfhard",
        "Henry Cavill, Anya Chalotra, Freya Allan", "Jason Bateman, Laura Linney, Julia Garner",
        "Bryan Cranston, Aaron Paul, Anna Gunn", "Wagner Moura, Pedro Pascal, Boyd Holbrook",
        "Kevin Spacey, Robin Wright, Michael Kelly", "Cillian Murphy, Paul Anderson, Sophie Rundle",
        "Steve Carell, Rainn Wilson, Jenna Fischer", "Amy Poehler, Nick Offerman, Aubrey Plaza",
        "Karl Urban, Jack Quaid, Antony Starr", "Lee Jung-jae, Park Hae-soo, Sung Bum",
        "Zendaya, Jacob Elordi, Hunter Schafer", "Jonathan Groff, Holt McCallany, Anna Torv",
        "Louis Hofmann, Ulrich Nielsen, Charlotte Doppler", "Pedro Pascal, Grogu, Gina Carano",
        "Jenna Ortega, Catherine Zeta-Jones, Luis Guzmán", "Evan Peters, Lionel Dahmer, Molly Ringwald",
        "Penn Badgley, Victoria Pedretti, Ambyr Childers", "Álvaro Morte, Úrsula Corberó, Itziar Ituño",
        "Paul Ritter, Stellan Skarsgård, Emily Watson", "Gabriel Luna, Bella Ramsey, Pedro Pascal",
        "Andie MacDowell, Murray Bartlett, Jennifer Coolidge", "Brian Cox, Jeremy Strong, Sarah Snook",
        "Adam Scott, Britt Lower, Patricia Arquette", "Winona Ryder, David Harbour, Finn Wolfhard",
        "Iman Marson, Igby Rigney, Mike Flanagan", "Jenna Ortega, Catherine Zeta-Jones, Luis Guzmán",
        "Ryô Murakawa, Sakurako Konishi, Toru Kandou", "Vera Farmiga, Patrick Wilson, Lili Taylor",
    ]
    
    descriptions = [
        "Follows the political rivalries and romance within the reign of Queen Elizabeth II.",
        "When a young boy disappears, his friends uncover a mystery involving secret government experiments.",
        "A witcher hunts monsters for coin across a turbulent world filled with dark magic.",
        "A financial advisor and his wife launder drug money for a Mexican cartel.",
        "A high school chemistry teacher turned meth manufacturer builds a drug empire.",
        "An undercover DEA agent infiltrates the cocaine trade in Colombia during the 1980s.",
        "A ruthless politician and his wife scheme and manipulate their way to the White House.",
        "A gangster family epic spanning several generations following their rise in the 1920s.",
        "A mockumentary about the everyday lives of office workers at a paper company.",
        "The everyday lives of employees in a fictional government department filled with odd situations.",
        "Superheroes are treated as celebrities by a powerful corporation, but the truth is darker.",
        "Hundreds of people compete in deadly games for a massive cash prize.",
        "A troubled teen explores her identity during the pandemic with friends navigating trauma.",
        "Two FBI agents examine the psychology of serial killers to catch a killer on the loose.",
        "A group of people must solve the mystery of a small town where time loops repeatedly.",
        "After the events of The Empire Strikes Back, a lone bounty hunter operates in the outer reaches.",
        "Wednesday Addams explores her psychic powers and attends Nevermore Academy.",
        "A limited series about serial killer Jeffrey Dahmer and his horrifying crimes.",
        "A charming con man develops a dangerous obsession with a woman and will do anything for her.",
        "A group of thieves plans the greatest heist in history: stealing gold from the Bank of Spain.",
    ]
    
    release_years = np.random.choice(range(2015, 2024), num_records)
    
    data = {
        'show_id': [f's{i:04d}' for i in range(num_records)],
        'title': np.random.choice(titles, num_records),
        'type': np.random.choice(['Movie', 'TV Show'], num_records, p=[0.4, 0.6]),
        'director': np.random.choice(directors, num_records),
        'cast': np.random.choice(cast_list, num_records),
        'country': np.random.choice(['United States', 'United Kingdom', 'South Korea', 'Germany', 'Spain', 'Japan'], num_records),
        'date_added': pd.date_range(start='2015-01-01', periods=num_records, freq='D').strftime('%B %d, %Y'),
        'release_year': release_years,
        'rating': np.random.choice(['G', 'PG', 'PG-13', 'R', 'NC-17', 'TV-14', 'TV-MA'], num_records),
        'duration': [f'{np.random.randint(60, 180)} min' if t == 'Movie' else f'{np.random.randint(1, 5)} Seasons' 
                     for t in np.random.choice(['Movie', 'TV Show'], num_records, p=[0.4, 0.6])],
        'listed_in': np.random.choice(genres_list, num_records),
        'description': np.random.choice(descriptions, num_records),
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Sample dataset created with {num_records} records at {output_path}")
    
    return df

if __name__ == '__main__':
    import os
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'netflix_sample.csv')
    create_sample_netflix_dataset(output_path, num_records=500)
