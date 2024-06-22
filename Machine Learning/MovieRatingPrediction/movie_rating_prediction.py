import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import zipfile
import os

# Step 1: Download and Load the Data
# Assuming the dataset is downloaded and extracted in the current directory
dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
dataset_path = 'ml-100k.zip'
data_dir = 'ml-100k'

if not os.path.exists(data_dir):
    # Download the dataset
    !wget {dataset_url}
    # Extract the dataset
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall()

# Load the datasets
ratings = pd.read_csv(os.path.join(data_dir, 'u.data'), sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
movies = pd.read_csv(os.path.join(data_dir, 'u.item'), sep='|', encoding='latin-1', header=None,
                     names=['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action',
                            'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                            'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])

# Merge datasets
data = pd.merge(ratings, movies, on='movie_id')

# Step 2: Data Preprocessing
# Select relevant features and one-hot encode genres
genres = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

data = data[['movie_id', 'title', 'rating'] + genres]

# Step 3: Feature Engineering
# One-hot encode genres
encoder = OneHotEncoder(sparse=False)

genre_features = data[genres].values

# Prepare the feature matrix
X = genre_features
y = data['rating']

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Optional: Display predictions vs actual ratings
for i in range(len(y_test)):
    print(f'Predicted: {y_pred[i]:.2f}, Actual: {y_test.iloc[i]:.2f}')
