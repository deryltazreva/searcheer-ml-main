from analyzer import EnhancedJobAnalyzer
from utils.pdf_utils import upload_cv
from utils.language_utils import validate_english_text
from utils.experience_utils import analyze_experience_match, analyze_education_match, analyze_industry_match
from utils.skills_utils import extract_comprehensive_skills_match
from utils.similarity_utils import generate_detailed_analysis_report, find_alternative_jobs_for_cv
import pandas as pd
import os
import time

def main():
    print("Enhanced Job Compatibility Analyzer")
    print("-" * 50)

    # upload CV
    print("\n" + "-"*50)
    print("UPLOAD YOUR CV")
    print("-"*50)
    cv_text = upload_cv()
    if not cv_text:
        print("CV upload failed. Exiting...")
        return

    # input job description
    print("\n" + "-"*50)
    print("INPUT JOB DESCRIPTION")
    print("-"*50)
    job_title = input("Job Title: ").strip()
    job_description = input("Job Description (paste full text): \n").strip()

    if len(job_title) < 5 or len(job_description) < 50:
        print("Job title or description too short. Exiting...")
        return

    job_lang_check = validate_english_text(job_description, "Job Description")
    if not job_lang_check['is_english']:
        print(job_lang_check['message'])
        return

    #load dataset
    print("\n" + "-"*50)
    print("LOADING JOB DATASET")
    print("-"*50)
    dataset_path = "D:/dbs_coding_camp/ml-capstone/data/cleaned_fake_job_postings.csv"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return

    job_data = pd.read_csv(dataset_path)
    print(job_data.columns)
    job_data = job_data[job_data.get("fraudulent", 0) == 0]
    job_data = job_data.dropna(subset=['job_id','title', 'description'])

    #analyzer
    print("\n" + "-"*50)
    print("ANALYZING")
    print("-"*50)
    analyzer = EnhancedJobAnalyzer()
    analyzer.train_neural_network(job_data)

    #men-generate report
    results = generate_detailed_analysis_report(cv_text, job_title, job_description, analyzer)
    if not results:
        print("Analysis failed. Exiting...")
        return

    # recommendasi
    print("\n" + "-"*50)
    print("RECOMMENDATIONS")
    print("-"*50)
    find_alternative_jobs_for_cv(job_data, cv_text, results)

if __name__ == "__main__":
    main()
