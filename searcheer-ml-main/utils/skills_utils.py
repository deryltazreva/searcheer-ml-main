def extract_comprehensive_skills_match(cv_text, job_text):
    skill_keywords = {
        'python': 1.0, 'sql': 1.0, 'excel': 0.9, 'power bi': 1.0,
        'tableau': 1.0, 'communication': 0.6, 'teamwork': 0.6,
        'machine learning': 1.0, 'deep learning': 1.0, 'ai': 1.0,
        'data analysis': 1.0, 'data science': 1.0
    }
    cv_lower = cv_text.lower()
    job_lower = job_text.lower()
    matched = []
    missing = []
    for skill, weight in skill_keywords.items():
        in_cv = skill in cv_lower
        in_job = skill in job_lower
        if in_job and in_cv:
            matched.append((skill, weight))
        elif in_job:
            missing.append((skill, weight))
    skill_match_pct = (len(matched) / max(len(matched) + len(missing), 1)) * 100
    return {
        'matched_skills': matched,
        'missing_skills': missing,
        'skill_match_percentage': skill_match_pct,
        'total_cv_skills': len([s for s, _ in skill_keywords.items() if s in cv_lower]),
        'total_required_skills': len([s for s, _ in skill_keywords.items() if s in job_lower]),
        'total_matched_skills': len(matched),
        'skill_coverage': (len(matched) / max(len(skill_keywords), 1)) * 100
    }
