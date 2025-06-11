from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.skills_utils import extract_comprehensive_skills_match
from utils.experience_utils import analyze_experience_match, analyze_education_match, analyze_industry_match

def calculate_text_similarity(text1, text2, vectorizer):
    try:
        texts = [text1.strip(), text2.strip()]
        tfidf_matrix = vectorizer.fit_transform(texts)
        if tfidf_matrix.shape[1] == 0:
            return 0.0
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except Exception as e:
        print(f"Similarity calculation error: {e}")
        return 0.0

def generate_detailed_analysis_report(cv_text, job_title, job_description, analyzer):
    
    print("\nDETAILED BREAKDOWN:")
    
    similarity = calculate_text_similarity(analyzer.preprocess_text(cv_text), analyzer.preprocess_text(job_description), analyzer.tfidf_vectorizer) * 100
    print(f"Text Similarity: {similarity:.1f}%")
    
    skills = extract_comprehensive_skills_match(cv_text, job_description)
    print(f"Skill Match: {skills['skill_match_percentage']:.1f}%")
    
    experience = analyze_experience_match(cv_text, job_description)
    print(f"Experience Match: {experience:.1f}%")
    
    education = analyze_education_match(cv_text, job_description)
    print(f"Education Match: {education:.1f}%")
    
    industry = analyze_industry_match(cv_text, job_description)
    print(f"Industry Match: {industry:.1f}%")

    total_score = (
        similarity * 0.15 +
        skills['skill_match_percentage'] * 0.40 +
        experience * 0.20 +
        education * 0.15 +
        industry * 0.10
    )
    print(f"\nCompatibility Score: {total_score:.1f}%")

    # if analyzer.is_nn_trained:
    #     nn_score = analyzer.predict_compatibility_nn(cv_text, f"{job_title} {job_description}")
    #     if nn_score is not None:
    #         print(f"\nNeural Network Prediction: {nn_score:.1f}%")

    print("\nSKILL ANALYSIS")
    print("\nMatched Skills:")
    for skill, _ in skills['matched_skills']:
        print(f"   {skill}")
    
    print("\nMissing Skills (High Priority):")
    for skill, _ in skills['missing_skills']:
        print(f"   {skill}")
    
    print("\nAdditional Skills in Your CV:")

    cv_skills = skills.get('total_cv_skills', None)

    if isinstance(cv_skills, list):
        if len(cv_skills) > 0:
            for skill in cv_skills[:10]:
                print(f"   {skill}")
        else:
            print("   No additional skills found in your CV.")
    elif cv_skills is not None:
        print(f"   Total skills in CV: {cv_skills}")

    if total_score < 45:
        print("LOW MATCH. Consider alternative positions or significant upskilling.")
        print("Tips to improve:")
        print("• Focus on building fundamental skills first")
        print("• Look for entry-level positions in this field")
        print("• Consider career transition planning")
    elif total_score < 60:
        print("MODERATE MATCH. Focus on skill development.")
        print("Tips to improve:")
        print("• Develop missing high-priority skills")
        print("• Gain relevant experience through projects")
        print("• Consider applying for similar junior roles")
    else:
        print("GOOD MATCH. You're a competitive candidate.")
        print("Tips to improve:")
        print("• Highlight your matching skills in application")
        print("• Prepare examples demonstrating relevant experience")
        print("• Consider applying with confidence")
    return {
        'overall_score': total_score,
        'text_similarity': similarity,
        'skill_match': skills['skill_match_percentage'],
        'experience_match': experience,
        'education_match': education,
        'industry_match': industry,
        'skills_analysis': skills
    }


def find_alternative_jobs_for_cv(job_data, cv_text, results, top_n=6):
    if not results.get('skills_analysis'):
        return []
    
    cv_text = cv_text.lower()
    matched_skills = [s for s, _ in results['skills_analysis']['matched_skills']]
    matches = []

    #ini sengaja ku tambahin untuk debug aja
    print(f"Available columns in job_data: {job_data.columns.tolist()}")
    
    id_column = None
    if 'job_id' in job_data.columns:
        id_column = 'job_id'
        print("Using 'job_id' column")
    elif 'id' in job_data.columns:
        id_column = 'id'
        print("Using 'id' column")
    else:
        id_column = 'index'
        print("Using index as job_id fallback")
    
    for idx, job in job_data.iterrows():
        try:
            title = str(job.get('title', 'Unknown Title'))
            description = str(job.get('description', ''))
            
            text = f"{title} {description}".lower()
            if len(text) < 30:
                continue
            
            skill_match = sum([1 for s in matched_skills if s in text]) / max(len(matched_skills), 1)
            text_sim = calculate_text_similarity(cv_text, text, TfidfVectorizer())
            score = 0.6 * text_sim + 0.4 * skill_match
            
            if id_column == 'job_id':
                job_id = job.get('job_id', f"job_{idx}")
            elif id_column == 'id':
                job_id = job.get('id', f"job_{idx}")
            else:  
                job_id = f"job_{idx}"
            
            matches.append((score, title, description, str(job_id)))
            
        except Exception as e:
            print(f"Error processing job at index {idx}: {e}")
            continue
    
    matches = sorted(matches, key=lambda x: x[0], reverse=True)[:top_n]
    
    recommended_jobs = []
    for i, (score, title, desc, job_id) in enumerate(matches, 1):
        desc_preview = desc[:100] + "..." if len(desc) > 100 else desc
        
        job_recommendation = {
            'rank': i,
            'job_id': str(job_id),  
            'job_title': title,
            'score': round(float(score), 4),  
            'description': desc_preview
        }
        
        recommended_jobs.append(job_recommendation)
        
        #debug hasil report
        print(f"Adding job {i}: ID={job_id}, Title={title[:50]}..., Score={score:.4f}")
    
    return recommended_jobs
