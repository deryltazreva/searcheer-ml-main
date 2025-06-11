from neural_model import JobCompatibilityNeuralNetwork
from utils.similarity_utils import calculate_text_similarity
from utils.skills_utils import extract_comprehensive_skills_match
from utils.experience_utils import analyze_experience_match, analyze_education_match, analyze_industry_match
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import re

class EnhancedJobAnalyzer:
    def __init__(self):
        self.nn_model = JobCompatibilityNeuralNetwork()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.is_nn_trained = False

    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return ' '.join(text.split())

    def create_synthetic_training_data(self, job_data, cv_samples):
        print("Creating synthetic training data...")
        training_data = []
        for cv_text in cv_samples:
            cv_processed = self.preprocess_text(cv_text)
            sample_jobs = job_data.sample(min(100, len(job_data)))
            for _, job in sample_jobs.iterrows():
                job_text = self.preprocess_text(f"{job['title']} {job['description']}")
                if len(job_text.strip()) < 30:
                    continue
                similarity = calculate_text_similarity(cv_processed, job_text, self.tfidf_vectorizer)
                label = 1 if similarity > 0.1 else 0
                training_data.append({
                    'cv_text': cv_processed,
                    'job_text': job_text,
                    'label': label
                })
        return pd.DataFrame(training_data)

    def train_neural_network(self, job_data):
        cv_samples = self.generate_cv_samples()
        training_data = self.create_synthetic_training_data(job_data, cv_samples)
        if len(training_data) < 50:
            print("Not enough training data. Skipping training.")
            return
        self.nn_model.train(training_data)
        self.is_nn_trained = self.nn_model.is_trained

    def predict_compatibility_nn(self, cv_text, job_text):
        return self.nn_model.predict(cv_text, job_text)

    def generate_cv_samples(self):
        return [
            "Experienced software engineer with 5 years in Python, JavaScript, and machine learning.",
            "Data scientist with experience in R, SQL, and deep learning.",
            "Frontend developer with React and Angular experience.",
            "DevOps engineer with Docker, Kubernetes, and AWS expertise.",
            "Marketing specialist with SEO and content strategy background.",
            "Financial analyst with Excel, SQL, and modeling skills.",
            "Product manager with agile, scrum, and product strategy experience.",
            "UI/UX designer with Figma and prototyping experience."
        ]
