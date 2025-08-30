# Visual Product Matcher

A simple web application that allows users to upload an image (or provide an image URL) and find visually similar products from a small database. Built to be fast, accurate, and easy to use, with a focus on combining visual features, color, and category information for better matches.

---

## **Live Demo**

[Insert your live application URL here]

---

## **Overview**

The goal of this project is to create a lightweight system that can find products visually and feature-wise similar to any uploaded image. Users can upload a shoe, bag, or any other product, and see the closest matches from a curated database.

---

## **Features**

- **Image Upload**:  
  - Upload an image directly from your device.  
  - Or provide an image URL to fetch it online.  

- **Search & Display**:  
  - Preview the uploaded image.  
  - See a list of up to 20 similar products.  
  - Filter results by dominant color and similarity score.  

- **Product Database**:  
  - Contains 50+ products with images.  
  - Each product has metadata like name, category, and dominant colors.  

- **User Experience Enhancements**:  
  - Basic error handling for invalid files or URLs.  
  - Loading indicators while processing the image.  
  - Mobile responsive design for access on any device.  

---

## **My Approach**

Finding matching products can be done in several ways, but machine learning is one of the go-to methods. Even within ML, there are multiple approaches—like color matching, feature extraction, and more—but the fastest and most effective method I chose is **embeddings**.  

Embeddings are essentially a compact representation of an image, like a “hash” that captures the important visual features. These embeddings can then be compared using algorithms like **cosine similarity** to find similar items.  

To improve results and find items that are visually **and** feature-wise similar, I combined multiple techniques:  

1. **Pretrained ResNet50 Model**:  
   - Reads the uploaded image.  
   - Predicts the **category** of the item.  

2. **Dominant Color Extraction**:  
   - Identifies the most dominant colors in the image.  
   - Helps match products that are visually similar in color.  

3. **Embedding Generation**:  
   - Creates embeddings for the uploaded image.  
   - Stores embeddings in the database for future searches if the item is new.  

4. **Matching & Filtering**:  
   - First filters products by **category**.  
   - Then filters by at least **2 dominant color matches**.  
   - Finally, ranks and filters based on **embedding similarity**.  

5. **Results Display**:  
   - Top 20 matching products are displayed to the user, combining visual and feature-based similarity.  

This hybrid approach ensures that matches are not just visually similar but also belong to the correct category and have similar dominant colors, making the search results more relevant and accurate.

---

## **Tech Stack**

- **Frontend**: HTML, CSS  
- **Backend**: Python (Flask)  
- **AI/ML**: Pretrained ResNet50 for feature extraction and category prediction  
- **Database**: JSON or simple file-based storage for product metadata and embeddings  

---
