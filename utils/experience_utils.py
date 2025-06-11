import re

def analyze_experience_match(cv_text, job_text):
    patterns = [r'(\d+)\s*(?:years?|yrs?)', r'minimum\s*(\d+)', r'over\s*(\d+)']
    def extract_years(text):
        years = []
        for p in patterns:
            matches = re.findall(p, text.lower())
            years += [int(m) for m in matches if m.isdigit()]
        return years

    cv_years = extract_years(cv_text)
    job_years = extract_years(job_text)
    if not cv_years or not job_years:
        return 50.0
    cv_max = max(cv_years)
    job_min = min(job_years)
    if cv_max >= job_min:
        return 100.0
    elif job_min - cv_max == 1:
        return 80.0
    elif job_min - cv_max == 2:
        return 60.0
    else:
        return 40.0

def analyze_education_match(cv_text, job_text):
    levels = {'phd': 4, 'master': 3, 'bachelor': 2, 'diploma': 1, 'high school': 0}
    def get_level(text):
        text = text.lower()
        return max([lvl for key, lvl in levels.items() if key in text], default=0)
    cv_level = get_level(cv_text)
    job_level = get_level(job_text)
    if cv_level >= job_level:
        return 100.0
    elif cv_level == job_level - 1:
        return 80.0
    elif cv_level == job_level - 2:
        return 60.0
    else:
        return 30.0

def analyze_industry_match(cv_text, job_text):
    keywords = {
        'tech': ['developer', 'engineer', 'software'],
        'finance': ['finance', 'accounting', 'analyst'],
        'marketing': ['marketing', 'seo', 'campaign'],
        'design': ['ux', 'ui', 'design', 'figma']
    }
    def find_industries(text):
        text = text.lower()
        return {ind for ind, keys in keywords.items() if any(k in text for k in keys)}
    cv_ind = find_industries(cv_text)
    job_ind = find_industries(job_text)
    if not job_ind:
        return 100.0
    return (len(cv_ind.intersection(job_ind)) / len(job_ind)) * 100 if cv_ind else 50.0
