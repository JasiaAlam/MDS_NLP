from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import os


# Read data
def read_data(folder_name, zip):
    data_path = os.path.join(folder_name, zip)
    data = pd.read_csv(data_path)

    return data

# Tokenize the column
def tokenize_column(data, text_col, new_col):
    data[new_col] = data[text_col].apply(lambda x: x.split())
    
    return data


# Train a Word2Vec model on the genres
def train_word2vec_model(genres_data, col_name, vector_size=150, window=5, min_count=1, workers=4):
    # Train a Word2Vec model on the genres
    model = Word2Vec(sentences=genres_data[col_name], 
                     vector_size=vector_size, 
                     window=window, 
                     min_count=min_count, 
                     workers=workers)
    model.save("narrative_word2vec.model")

# Average genre vectore and store as a new column
def add_column_average_genre_vector(genres_data, model_path, col_name, vector_size=150):
    # Load the trained model
    model = Word2Vec.load(model_path)
    # Create a dictionary with genre embeddings
    genre_embeddings = {genre: model.wv[genre] for genre in model.wv.index_to_key}

    # Average genre vector
    def average_genre_vector(genre_list, genre_embeddings, vector_size):
        # Filter out genres not present in the embedding model
        valid_embeddings = [genre_embeddings[genre] for genre in genre_list if genre in genre_embeddings]
        if not valid_embeddings:
            return np.zeros(vector_size)  # Return a zero vector if no valid genres
        # Compute the mean vector
        mean_vector = np.mean(valid_embeddings, axis=0)
        return mean_vector

    # Apply this function to dataset
    genres_data[col_name + '_vector'] = genres_data[col_name].apply(
        lambda genres: average_genre_vector(genres, genre_embeddings, vector_size))

    return genres_data

# Store the data as a new CSV file
def store_data(genres_data, path):
    genres_data.to_csv(path, index=False)

# Main
def main():
    col_name = "narrative_tokenized"
    # Read data
    data = read_data("data", "data_eda.zip")

    # Tokenize the column
    data = tokenize_column(data, "narrative_prep", col_name)

    # Train a Word2Vec model on the genres
    train_word2vec_model(data, col_name=col_name)

    # Add a new column with the average genre vector
    data = add_column_average_genre_vector(data, 
                                           model_path="narrative_word2vec.model", 
                                           col_name=col_name)

    # Store the data as a new CSV file
    store_data(data, "data/data_narrative_vector.csv")

    print("Done!")

if __name__ == "__main__":
    main()