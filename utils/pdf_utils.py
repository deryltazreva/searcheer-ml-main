import PyPDF2
import re
import io
import os
from utils.language_utils import validate_english_text

def extract_text_from_pdf(file_content):
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {e}")

def is_ats_friendly(text):
    issues = []
    score = 0
    
    #essential sections 30 poing
    essential_sections = [
        r'(?i)\b(contact|personal|info|details)\b',
        r'(?i)\b(experience|employment|work|career)\b',
        r'(?i)\b(education|academic|qualification)\b',
        r'(?i)\b(skills|competenc|technical)\b'
    ]
    
    sections_found = 0
    for section in essential_sections:
        if re.search(section, text):
            sections_found += 1
    
    section_score = (sections_found / len(essential_sections)) * 30
    score += section_score
    
    if sections_found < 3:
        issues.append("Missing essential sections (Contact, Experience, Education, Skills)")
    
    #proper contact information 20 poin
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'(?:\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}'
    
    contact_score = 0
    if re.search(email_pattern, text):
        contact_score += 10
    else:
        issues.append("No valid email address found")
    
    if re.search(phone_pattern, text):
        contact_score += 10
    else:
        issues.append("No valid phone number found")
    
    score += contact_score
    
    #panjang teks dan readability 15 poin
    word_count = len(text.split())
    if word_count >= 200:
        score += 15
    elif word_count >= 100:
        score += 10
        issues.append("CV content seems too brief")
    else:
        issues.append("CV content is too short for proper ATS parsing")
    
    #structured format indicators 15 poin
    structure_indicators = [
        r'(?i)\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
        r'\b\d{4}\b',  #tahun
        r'(?i)\b(present|current|now)\b',  #current position
        r'(?i)\b(bachelor|master|phd|degree|diploma|certificate)\b'  #jenjang pendidikan
    ]
    
    structure_found = sum(1 for indicator in structure_indicators if re.search(indicator, text))
    structure_score = min(structure_found * 4, 15)
    score += structure_score
    
    if structure_score < 8:
        issues.append("Lacks proper date formatting or educational qualifications")
    
    #professional keyword(?) 10 poin
    professional_keywords = [
        r'(?i)\b(managed|developed|implemented|achieved|led|created|designed|coordinated)\b',
        r'(?i)\b(responsible|experience|project|team|client|customer)\b'
    ]
    
    keyword_found = sum(1 for keyword in professional_keywords if re.search(keyword, text))
    keyword_score = min(keyword_found * 2, 10)
    score += keyword_score
    
    if keyword_score < 4:
        issues.append("Limited professional action words or experience descriptions")
    
    #cek karakter aneh 10 poin
    problematic_chars = re.findall(r'[^\w\s\-.,@+()#/]', text)
    if len(problematic_chars) > 20:
        score += 5
        issues.append("Contains many special characters that may confuse ATS systems")
    else:
        score += 10
    
    #termasuk ats-friendly jika score total > 70 dan jumlah issues maks 2
    is_compatible = score >= 70 and len(issues) <= 2
    
    return is_compatible, issues, score

def upload_cv():
    print("Please enter the path to your CV (PDF format only):")
    path = input("Path to PDF: ").strip()

    if not path.lower().endswith('.pdf'):
        print("❌ Only PDF files are supported.")
        return None

    if not os.path.exists(path):
        print("❌ File not found.")
        return None

    with open(path, "rb") as f:
        file_content = f.read()
    
    text = extract_text_from_pdf(file_content)

    language_result = validate_english_text(text, text_type="CV")
    if language_result['detected_language'] != 'en':
        print(f"{language_result['message']}")
        return None
    
    ats_compatible, issues, score = is_ats_friendly(text)  
    
    if not ats_compatible:
        
        print("\nTips for creating an ATS-friendly CV:")
        print("   ✅ Use standard section headings (Contact, Experience, Education, Skills)")
        print("   ✅ Include clear contact information (email, phone)")
        print("   ✅ Use consistent date formats (MM/YYYY or Month YYYY)")
        print("   ✅ Include relevant keywords for your industry")
        print("   ✅ Use simple formatting without complex graphics")
        print("   ✅ Avoid special characters, images, or fancy fonts")
        print("   ✅ Save as PDF with selectable text")
        print("   ✅ Use bullet points for achievements and responsibilities")
        return None
    
    # if len(text) < 50:
    if len(text.split()) < 100: 
        print("\n❌ Your CV is poorly readable or uninformative")
        print("The system can only read partial information from your CV. This may be due to unclear structure or poor text quality.")
        print("\nImprovement suggestions:")
        print("   ✅ Ensure text is not blurry or distorted")
        print("   ✅ Use good color contrast between text and background")
        print("   ✅ Organize information with clear structure")
        print("   ✅ Use separate sections for each part of your CV")
        print("   ✅ Avoid overlapping text or elements")
        print("   ✅ Provide adequate margins for each section")
        print("   ✅ Re-scan if CV is from a physical document scan")
        return None
    print(f"✅ Extracted {len(text)} characters from PDF.")
    return text

if __name__ == "__main__":
    cv_text = upload_cv()
    