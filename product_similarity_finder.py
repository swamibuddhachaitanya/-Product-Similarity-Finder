# product_similarity_finder.py

import os
import json
import requests
import zipfile
import shutil
import pandas as pd
import numpy as np
import torch
from PIL import Image
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from transformers import ViTFeatureExtractor, ViTForImageClassification, AutoTokenizer, AutoModel, SentenceTransformer
import faiss
import re

# Mount Google Drive
def mount_drive():
    from google.colab import drive
    drive.mount('/content/drive')

# Load DataFrame from Google Drive
def load_dataframe(file_path):
    return pd.read_csv(file_path)

# Drop NaN values
def drop_na(df):
    df.dropna(inplace=True)
    return df

# Display DataFrame columns
def display_columns(df):
    print(df.columns)

# Install required packages
def install_packages():
    !pip install langid requests pillow transformers faiss-gpu sentence_transformers

# Detect language of descriptions
def detect_language(df):
    import langid
    df['language'] = df['DESCRIPTION'].apply(lambda x: langid.classify(x)[0])
    return df[df['language'] == 'en']

# Extract image URLs
def extract_image_url(optional_field):
    try:
        data = json.loads(optional_field)
        return data.get('primaryImageUrl', data.get('imageUrl', None))
    except:
        return None

def filter_images(df):
    df['IMAGE_URL'] = df['OPTIONAL_FIELDS'].apply(extract_image_url)
    df = df[df['IMAGE_URL'].notna()]
    return df[['COMP_SKU', 'DESCRIPTION', 'CATEGORY_LEVEL1', 'IMAGE_URL']]

# Fetch images from URLs
def fetch_image(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except:
        return None

def add_product_images(df):
    df['PRODUCT_IMAGES'] = df['IMAGE_URL'].apply(fetch_image)
    df = df.dropna(subset=['PRODUCT_IMAGES'])
    df = df[df['PRODUCT_IMAGES'] != '']
    return df[['COMP_SKU', 'DESCRIPTION', 'CATEGORY_LEVEL1', 'PRODUCT_IMAGES']]

# Save images to a directory
def save_images(df, image_dir, batch_size=6):
    os.makedirs(image_dir, exist_ok=True)
    all_image_paths = []
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size else 0)
    
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        batch_df = df.iloc[batch_start:batch_end]
        image_paths = []
        for index, row in batch_df.iterrows():
            image = row['PRODUCT_IMAGES']
            if image:
                if image.mode == 'P':
                    image = image.convert('RGB')
                image_path = os.path.join(image_dir, f'image_{batch_start + index}.jpg')
                image.save(image_path)
                image_paths.append(image_path)
            else:
                image_paths.append(None)
        all_image_paths.extend(image_paths)
    
    df['IMAGE_PATH'] = all_image_paths
    df.drop('PRODUCT_IMAGES', axis=1, inplace=True)
    return df

# Load images from paths
def load_images(df):
    df['LOADED_IMAGES'] = df['IMAGE_PATH'].apply(lambda x: Image.open(x) if os.path.exists(x) else None)
    return df

# Remove duplicate entries
def remove_duplicates(df):
    return df.drop_duplicates(subset='COMP_SKU')

# Extract image features using ViT
def extract_image_features(df, model, feature_extractor, batch_size=16):
    image_data = df['LOADED_IMAGES'].tolist()
    image_features = []

    for i in range(0, len(image_data), batch_size):
        batch = image_data[i:i + batch_size]
        inputs = feature_extractor(images=batch, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        image_features.extend(outputs.logits)

    df['IMAGE_FEATURES'] = np.array(image_features)
    return df

# Save DataFrame to CSV
def save_dataframe(df, file_name):
    df.to_csv(file_name, index=False)

# Create FAISS index
def create_faiss_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

# Calculate similarity using FAISS
def calculate_similarity(query_vector, index):
    D, I = index.search(np.array([query_vector]), index.ntotal)
    return D, I

# Get similar products
def get_similar_products(df, text_index, image_index, text_embeddings, image_embeddings, query_sku, k=10, text_weight=0.6, image_weight=0.4):
    sku_to_index = {sku: idx for idx, sku in enumerate(df['COMP_SKU'])}
    
    if query_sku not in sku_to_index:
        raise ValueError(f"SKU {query_sku} not found in dataset")

    query_idx = sku_to_index[query_sku]
    query_text_embedding = text_embeddings[query_idx]
    query_image_feature = image_embeddings[query_idx]

    text_cosine_sim, _ = calculate_similarity(query_text_embedding, text_index)
    image_cosine_sim, _ = calculate_similarity(query_image_feature, image_index)

    aggregated_similarity_scores = (text_cosine_sim * text_weight + image_cosine_sim * image_weight).flatten()
    top_k_indices = np.argsort(aggregated_similarity_scores)[::-1][1:k+1]

    similar_products = []
    for index in top_k_indices:
        sku = df.iloc[index]['COMP_SKU']
        text_similarity_score = text_cosine_sim[0][index]
        image_similarity_score = image_cosine_sim[0][index]
        total_similarity_score = aggregated_similarity_scores[index]
        similar_products.append({
            'COMP_SKU': sku,
            'TEXT_SIMILARITY': text_similarity_score,
            'IMAGE_SIMILARITY': image_similarity_score,
            'TOTAL_SIMILARITY': total_similarity_score
        })

    return similar_products

# Example usage
if __name__ == "__main__":
    # Mount Google Drive
    mount_drive()

    # Load data
    df = load_dataframe('/content/drive/MyDrive/Colab Notebooks/twenty_df.csv')

    # Clean data
    df = drop_na(df)
    df = detect_language(df)
    df = filter_images(df)
    df = add_product_images(df)
    df = save_images(df, 'image_data_twenty')
    df = load_images(df)

    # Remove duplicates
    df = remove_duplicates(df)

    # Extract image features using ViT
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-large-patch16-224-in21k")
    model = ViTForImageClassification.from_pretrained("google/vit-large-patch16-224-in21k")
    model.eval()
    df = extract_image_features(df, model, feature_extractor)

    # Save DataFrame
    save_dataframe(df, 'ImgSixty.csv')

    # Create FAISS indexes
    text_embeddings = np.stack(df['TEXT_EMBEDDINGS'].values)
    image_embeddings = np.stack(df['IMAGE_FEATURES'].values)
    text_embeddings /= np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    image_embeddings /= np.linalg.norm(image_embeddings, axis=1, keepdims=True)

    text_index = create_faiss_index(text_embeddings)
    image_index = create_faiss_index(image_embeddings)

    # Get similar products
    query_sku = 'B00CALXTDM'
    similar_products = get_similar_products(df, text_index, image_index, text_embeddings, image_embeddings, query_sku)

    # Print similar products
    for product in similar_products:
        print(f"COMP_SKU: {product['COMP_SKU']}")
        print(f"Text Similarity: {product['TEXT_SIMILARITY']:.4f}")
        print(f"Image Similarity: {product['IMAGE_SIMILARITY']:.4f}")
        print(f"Total Similarity: {product['TOTAL_SIMILARITY']:.4f}")
        print("-" * 30)
