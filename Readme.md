# Product Similarity Finder

## Overview

This repository contains a product similarity finder that leverages both textual and image features to recommend similar products. Built using advanced machine learning techniques and FAISS (Facebook AI Similarity Search), this model combines text and image similarity scores to deliver accurate and relevant product recommendations.

## Features

- **Text and Image Similarity**: Uses a combination of text embeddings and image features to find similar products.
- **Efficient Search**: Employs FAISS for fast and scalable similarity search.
- **Multimodal Approach**: Aggregates similarity scores from both textual and visual data.

## Tech Stack

- **Python**: Core programming language used.
- **Pandas**: For data manipulation and analysis.
- **Transformers (Hugging Face)**: For text and image feature extraction.
- **FAISS**: For efficient similarity search.
- **PIL (Python Imaging Library)**: For image processing.
- **Google Colab**: For running the code and leveraging GPU resources.

## Getting Started

### Prerequisites

- Google Colab (Recommended) or a local Python environment with necessary libraries installed.
- Basic understanding of Python and machine learning concepts.

### Sample Data

You can download the sample data used in this project from the following link:
[Sample Data Download](https://drive.google.com/file/d/1EHgwx96LyLQQeEx2qyeXNpBhnrkl5r0O/view?usp=drive_link)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/product-similarity-finder.git
   cd product-similarity-finder
   ```

2. **Install dependencies:**

   If you're using Google Colab, install the necessary Python packages:

   ```python
   !pip install pandas requests pillow transformers faiss-gpu sentence_transformers
   ```

   For a local environment, install the dependencies using pip:

   ```bash
   pip install pandas requests pillow transformers faiss-gpu sentence_transformers
   ```

### Running the Code

1. **Mount Google Drive (if using Google Colab):**

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Load the sample data:**

   ```python
   import pandas as pd

   file_path = '/content/drive/MyDrive/path-to-your-sample-data.csv'
   df = pd.read_csv(file_path)
   ```

3. **Run the product similarity finder:**

   The main functionality is encapsulated in the `product_similarity_finder.py` script. You can execute this script to process the data, extract features, and find similar products.

   ```python
   from product_similarity_finder import (
       mount_drive, load_dataframe, drop_na, detect_language, 
       filter_images, add_product_images, save_images, load_images, 
       remove_duplicates, extract_image_features, create_faiss_index, 
       get_similar_products, save_dataframe
   )

   # Mount Google Drive (only if running on Google Colab)
   mount_drive()

   # Load the data
   df = load_dataframe('/content/drive/MyDrive/path-to-your-sample-data.csv')

   # Process the data
   df = drop_na(df)
   df = detect_language(df)
   df = filter_images(df)
   df = add_product_images(df)
   df = save_images(df, 'image_data')
   df = load_images(df)
   df = remove_duplicates(df)

   # Extract features and create FAISS indexes
   feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-large-patch16-224-in21k")
   model = ViTForImageClassification.from_pretrained("google/vit-large-patch16-224-in21k")
   model.eval()
   df = extract_image_features(df, model, feature_extractor)
   save_dataframe(df, 'ImgSixty.csv')

   text_embeddings = np.stack(df['TEXT_EMBEDDINGS'].values)
   image_embeddings = np.stack(df['IMAGE_FEATURES'].values)
   text_index = create_faiss_index(text_embeddings)
   image_index = create_faiss_index(image_embeddings)

   # Example: Find similar products
   query_sku = 'B00CALXTDM'
   similar_products = get_similar_products(df, text_index, image_index, text_embeddings, image_embeddings, query_sku)

   for product in similar_products:
       print(f"COMP_SKU: {product['COMP_SKU']}")
       print(f"Text Similarity: {product['TEXT_SIMILARITY']:.4f}")
       print(f"Image Similarity: {product['IMAGE_SIMILARITY']:.4f}")
       print(f"Total Similarity: {product['TOTAL_SIMILARITY']:.4f}")
       print("-" * 30)
   ```

### Saving and Accessing Results

After processing, you can save the DataFrame with the extracted image paths and features:

```python
save_dataframe(df, 'ImgSixty.csv')
```

You can also create a ZIP archive of the images or any other processed data and download it:

```python
import shutil

shutil.make_archive('image_data', 'zip', 'image_data')
from google.colab import files
files.download('image_data.zip')
```

## Contributing

Contributions are welcome! If you would like to contribute, please fork the repository, create a feature branch, and submit a pull request.

---

Happy coding! If you have any questions or issues, feel free to open an issue in the repository.
