# Personalized Course Recommender System

This repository contains the complete implementation of a **Personalized Course Recommender App**, built as part of the Machine Learning Capstone Project (e.g., for the IBM Data Science Professional Certificate).

The application is built using Python and Streamlit, providing an interactive web interface where users can select various machine learning models, tune hyperparameters, and receive dynamic course recommendations.

## 🚀 Features

The application implements and evaluates multiple recommendation strategies, ranging from simple content-based filtering to advanced collaborative filtering and deep learning models:

1. **Course Similarity (Content-Based):** Recommends courses based on textual similarity (BoW, TF-IDF) of course descriptions and titles.
2. **User Profile (Content-Based):** Extracts a user's domain interests by weighting course genres they have interacted with and recommends courses that align with that domain.
3. **Clustering (PCA & K-Means):** Groups similar users into learning personas and recommends popular courses within the user's assigned cluster.
4. **KNN (K-Nearest Neighbors):** A standard collaborative filtering approach based on the sparse user-item interaction matrix.
5. **NMF (Non-Negative Matrix Factorization):** Decomposes the interaction matrix to discover latent features and predict unseen ratings.
6. **Neural Networks (Embedding-Based):** Uses a Keras-based deep learning architecture to learn dense representations of users and courses.
7. **Regression with Embedding Features:** Uses embeddings extracted from the Neural Network as features for a Ridge/Linear Regression model.
8. **Classification with Embedding Features:** Uses embeddings extracted from the Neural Network as features for a Random Forest classifier.

## 🛠️ Tech Stack

* **Frontend UI:** Streamlit, Streamlit-AgGrid
* **Machine Learning / AI:** Scikit-Learn, TensorFlow, Keras
* **Data Processing:** Pandas, NumPy
* **Data Visualization (Report):** Matplotlib, WordCloud, FPDF

## 📂 Project Structure

* `recommender_app.py`: The main entry point for the Streamlit web application. Contains the UI logic and sidebar parameter tuning.
* `backend.py`: The core machine learning backend. Contains the logic for training models, generating embeddings, computing similarities, and returning predictions.
* `generate_report.py`: A utility script to generate the `Machine Learning Capstone Project Report.pdf` containing EDA visualizations and architectural flowcharts.
* `data/` (or root `.csv` files): Contains the pre-processed datasets, including `ratings.csv`, `course_processed.csv`, `course_genre.csv`, and precomputed embeddings for the neural networks.
* `requirements.txt`: Python dependencies required to run the app.

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hirendra84/course_recommender.git
   cd course_recommender
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App:**
   ```bash
   streamlit run recommender_app.py
   ```
   The application will automatically launch in your default web browser at `http://localhost:8501`.

## 📄 PDF Report Generation
To generate the multi-page Capstone PDF report (which includes EDA charts and system flowcharts), run:
```bash
pip install matplotlib wordcloud fpdf2
python3 generate_report.py
```

## 🤝 Acknowledgements
This project was developed following the architectural guidelines provided in the IBM Machine Learning Capstone course on Coursera.
