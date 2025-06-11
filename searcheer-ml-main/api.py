from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from utils.pdf_utils import extract_text_from_pdf, is_ats_friendly
from utils.language_utils import validate_english_text
from utils.similarity_utils import find_alternative_jobs_for_cv

import pandas as pd
import os
import logging
import uuid
import time
import json
import tempfile
import shutil
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

class Config:
    DATASET_PATH = os.getenv("DATASET_PATH", "data/cleaned_fake_job_postings.csv")
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB
    ALLOWED_FILE_TYPES = [".pdf"]
    MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", 30))
    ENABLE_NEURAL_NETWORK = os.getenv("ENABLE_NEURAL_NETWORK", "true").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this")  
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    TEMP_FOLDER = os.getenv("TEMP_FOLDER", "temp")
    MAX_CV_TEXT_LENGTH = int(os.getenv("MAX_CV_TEXT_LENGTH", 50000))  # Limit CV text length
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"

config = Config()

app = Flask(__name__)
app.config.from_object(config)

CORS(app, resources={
    r"/api/*": {
        "origins": os.getenv("ALLOWED_ORIGINS", "*").split(","),
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "expose_headers": ["X-Request-ID", "X-Response-Time"]
    }
})

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    storage_uri=os.getenv("REDIS_URL", "memory://"),  
    default_limits=["100 per hour", "20 per minute"]
)

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log') if os.path.exists('logs') else logging.StreamHandler(),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

analyzer = None
job_data = None

def clean_text_safe(text: Any) -> str:
    """Membersihkan data input"""
    try:
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')
        
        if not isinstance(text, str):
            text = str(text)
        
        #menghapus karakter non-printable
        cleaned = ''.join(char for char in text if char.isprintable() or char in '\n\r\t ')
        # Normalize whitespace
        cleaned = ' '.join(cleaned.split())
        
        #membatasi panjang teks
        if len(cleaned) > config.MAX_CV_TEXT_LENGTH:
            cleaned = cleaned[:config.MAX_CV_TEXT_LENGTH]
            logger.warning(f"Text truncated to {config.MAX_CV_TEXT_LENGTH} characters")
        
        return cleaned
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return ""

def create_standard_response(success: bool, message: str, data: Any = None, 
                           errors: List[str] = None, status_code: int = 200) -> tuple:
    """standard response format untuk API"""
    response = {
        "success": success,
        "message": message,
        "data": data,
        "errors": errors or [],
        "timestamp": datetime.utcnow().isoformat() + "Z",  
        "request_id": str(uuid.uuid4()),
        "api_version": "2.2.0"
    }
    return jsonify(response), status_code

def validate_file_upload(file) -> tuple:
    if not file:
        return False, "No file provided"
    
    if file.filename == '':
        return False, "No file selected"
    
    if not file.filename.lower().endswith('.pdf'):
        return False, "Only PDF files are supported"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  
    
    if file_size > config.MAX_FILE_SIZE:
        return False, f"File size ({file_size} bytes) exceeds limit ({config.MAX_FILE_SIZE} bytes)"
    
    if file_size == 0:
        return False, "File is empty"
    
    return True, "File is valid"

def allowed_file(filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in config.ALLOWED_FILE_TYPES)

def create_temp_file() -> str:
    os.makedirs(config.TEMP_FOLDER, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        delete=False, 
        suffix='.pdf', 
        dir=config.TEMP_FOLDER
    )
    return temp_file.name

def cleanup_temp_file(filepath: str) -> None:
    try:
        if os.path.exists(filepath):
            os.unlink(filepath)
            logger.debug(f"Cleaned up temp file: {filepath}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {filepath}: {e}")

def initialize_analyzer():
    global analyzer, job_data
    try:
        try:
            from analyzer import EnhancedJobAnalyzer
            analyzer = EnhancedJobAnalyzer()
            logger.info("Enhanced Job Analyzer initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import analyzer: {e}")
            analyzer = None

        if os.path.exists(config.DATASET_PATH):
            try:
                job_data = pd.read_csv(config.DATASET_PATH)
                
                logger.info(f"Dataset columns: {job_data.columns.tolist()}")

                initial_count = len(job_data)
                job_data = job_data[job_data.get("fraudulent", 0) == 0]
                required_columns = ['title', 'description']
                missing_columns = [col for col in required_columns if col not in job_data.columns]
                if missing_columns:
                    logger.error(f"Missing required columns: {missing_columns}")
                    job_data = create_mock_dataset()
                else:
                    job_data = job_data.dropna(subset=required_columns)
                    
                    job_data = job_data[job_data['title'].str.len() > 2]
                    job_data = job_data[job_data['description'].str.len() > 50]
                    
                    if 'job_id' not in job_data.columns and 'id' not in job_data.columns:
                        job_data = job_data.reset_index()
                        job_data['job_id'] = job_data.index + 1
                        logger.info("Created job_id column using index")
                    
                final_count = len(job_data)
                logger.info(f"Dataset loaded: {final_count}/{initial_count} valid jobs")
                
                if config.ENABLE_NEURAL_NETWORK and analyzer:
                    import threading
                    threading.Thread(target=train_analyzer_background, daemon=True).start()
                    
            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
                job_data = create_mock_dataset()
        else:
            logger.warning(f"Dataset not found at {config.DATASET_PATH}")
            job_data = create_mock_dataset()
            
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        analyzer = None
        job_data = create_mock_dataset()

def create_mock_dataset() -> pd.DataFrame:
    """membuat mock dataset untuk testing"""
    mock_data = {
        'job_id': [1, 2, 3, 4, 5, 6],
        'title': [
            'Software Developer', 'Data Scientist', 'ML Engineer', 
            'Frontend Developer', 'Backend Developer', 'Full Stack Developer'
        ],
        'description': [
            'Python development with Django and Flask frameworks',
            'Data analysis with Python, SQL, and machine learning',
            'Machine learning model development and deployment',
            'React and Angular frontend development',
            'Node.js and Python backend development',
            'Full stack development with modern technologies'
        ]
    }
    logger.info("Using mock dataset for testing")
    return pd.DataFrame(mock_data)

def train_analyzer_background():
    try:
        global analyzer, job_data
        if analyzer and job_data is not None and len(job_data) > 0:
            logger.info("Starting neural network training...")
            analyzer.train_neural_network(job_data)
            logger.info("Neural network training completed successfully")
    except Exception as e:
        logger.error(f"Background training error: {e}")

def perform_analysis(cv_text: str, job_title: str, job_description: str) -> dict:
    try:
        if not analyzer:
            logger.warning("Analyzer not available, using fallback analysis")
            return perform_fallback_analysis(cv_text, job_title, job_description)
        
        #memeriksa input
        if len(cv_text.strip()) < 50:
            raise ValueError("CV text too short for meaningful analysis")
        
        if len(job_description.strip()) < 20:
            raise ValueError("Job description too short for meaningful analysis")
        
        #memeriksa bahasa
        language_result = validate_english_text(job_description, text_type="Job Description")
        if language_result['detected_language'] != 'en':
            raise ValueError("Job description must be in English")
        
        title_lang_result = validate_english_text(job_title, text_type="Job Title")
        if title_lang_result['detected_language'] != 'en':
            raise ValueError("Job title must be in English")
        
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Analysis timeout")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  #timeout 30 detik
        
        try:
            #kemiripan teks
            cv_words = set(word.lower() for word in cv_text.split() if len(word) > 2)
            job_words = set(word.lower() for word in job_description.split() if len(word) > 2)
            
            common_words = cv_words.intersection(job_words)
            text_similarity = min(100.0, (len(common_words) / max(len(job_words), 1)) * 100)
        
            technical_skills = [
                'python', 'java', 'javascript', 'sql', 'r', 'scala',
                'machine learning', 'deep learning', 'data analysis', 'statistics',
                'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
                'aws', 'azure', 'gcp', 'docker', 'kubernetes',
                'react', 'angular', 'vue', 'node.js', 'django', 'flask'
            ]
            
            cv_lower = cv_text.lower()
            job_lower = job_description.lower()
            
            matched_skills = [skill for skill in technical_skills if skill in cv_lower and skill in job_lower]
            required_skills = [skill for skill in technical_skills if skill in job_lower]
            missing_skills = [skill for skill in required_skills if skill not in cv_lower][:5]
            
            skill_match = (len(matched_skills) / max(len(required_skills), 1)) * 100 if required_skills else 50
            
            #analisis experience/pengalaman
            import re
            cv_years = re.findall(r'(\d+)\s*(?:years?|yrs?)', cv_lower)
            cv_experience = max([int(year) for year in cv_years], default=0)
            
            job_years = re.findall(r'(\d+)\s*(?:years?|yrs?)', job_lower)
            required_experience = min([int(year) for year in job_years], default=0)
            
            experience_match = min(100.0, (cv_experience / max(required_experience, 1)) * 100) if required_experience > 0 else 75
            
            #overall score
            weights = {
                'text_similarity': 0.2,
                'skill_match': 0.4,
                'experience_match': 0.25,
                'base_score': 0.15
            }
            
            overall_score = (
                text_similarity * weights['text_similarity'] +
                skill_match * weights['skill_match'] +
                experience_match * weights['experience_match'] +
                60 * weights['base_score']  # Base compatibility score
            )
            
            overall_score = min(100.0, max(0.0, overall_score))
            confidence_score = min((skill_match + text_similarity + experience_match) / 300 + 0.2, 1.0)
            
            #tips
            if overall_score < 45:
                recommendation_level = "LOW_MATCH"
                tips = [
                    "Focus on developing fundamental skills mentioned in job description",
                    "Consider entry-level positions or internships in this field",
                    "Take relevant online courses or certifications",
                    "Build a portfolio demonstrating relevant skills",
                    "Network with professionals in this industry"
                ]
            elif overall_score < 70:
                recommendation_level = "MODERATE_MATCH"
                tips = [
                    "Develop the missing high-priority skills",
                    "Gain practical experience through projects or freelancing",
                    "Tailor your CV to highlight relevant experience",
                    "Consider similar roles to build experience",
                    "Showcase transferable skills from your background"
                ]
            else:
                recommendation_level = "STRONG_MATCH"
                tips = [
                    "Highlight your matching skills prominently in applications",
                    "Prepare specific examples of relevant experience for interviews",
                    "Apply with confidence to similar positions",
                    "Research the company culture and values",
                    "Prepare questions that demonstrate your interest and knowledge"
                ]
            
            result = {
                'overall_score': round(overall_score, 2),
                'text_similarity': round(text_similarity, 2),
                'skill_match': round(skill_match, 2),
                'experience_match': round(experience_match, 2),
                'education_match': round(overall_score * 0.8, 2),  
                'industry_match': round(overall_score * 0.9, 2),  
                'recommendation_level': recommendation_level,
                'matched_skills': matched_skills,
                'missing_skills': missing_skills,
                'tips': tips,
                'confidence_score': round(confidence_score, 2),
                'analysis_metadata': {
                    'cv_word_count': len(cv_text.split()),
                    'job_word_count': len(job_description.split()),
                    'common_words_count': len(common_words),
                    'cv_experience_years': cv_experience,
                    'required_experience_years': required_experience
                }
            }
            
            return result
            
        finally:
            signal.alarm(0)
            
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return perform_fallback_analysis(cv_text, job_title, job_description)

def perform_fallback_analysis(cv_text: str, job_title: str, job_description: str) -> dict:
    logger.info("Using fallback analysis method")

    try:
        job_lang_result = validate_english_text(job_description, text_type="Job Description")
        if job_lang_result['detected_language'] != 'en':
            raise ValueError(f"Job description language validation failed: {job_lang_result['message']}")
        
        title_lang_result = validate_english_text(job_title, text_type="Job Title")
        if title_lang_result['detected_language'] != 'en':
            raise ValueError(f"Job title language validation failed: {title_lang_result['message']}")
        
        cv_lang_result = validate_english_text(cv_text, text_type="CV")
        if cv_lang_result['detected_language'] != 'en':
            raise ValueError(f"CV language validation failed: {cv_lang_result['message']}")
    except Exception as lang_error:
        logger.error(f"Language validation error in fallback: {lang_error}")
        
        return {
            'overall_score': 0.0,
            'text_similarity': 0.0,
            'skill_match': 0.0,
            'experience_match': 0.0,
            'education_match': 0.0,
            'industry_match': 0.0,
            'recommendation_level': "LANGUAGE_ERROR",
            'matched_skills': [],
            'missing_skills': [],
            'tips': [str(lang_error)],
            'confidence_score': 0.0,
            'analysis_metadata': {
                'fallback_mode': True,
                'language_error': True,
                'error_message': str(lang_error)
            }
        }
    
    cv_words = set(cv_text.lower().split())
    job_words = set(job_description.lower().split())
    common_words = cv_words.intersection(job_words)
    
    basic_score = min(100.0, len(common_words) * 3)
    
    return {
        'overall_score': max(basic_score, 25.0),
        'text_similarity': basic_score,
        'skill_match': basic_score * 0.8,
        'experience_match': 50.0,
        'education_match': 50.0,
        'industry_match': 50.0,
        'recommendation_level': "MODERATE_MATCH",
        'matched_skills': [],
        'missing_skills': [],
        'tips': ["Analysis performed with limited capabilities. Consider uploading a more detailed CV."],
        'confidence_score': 0.3,
        'analysis_metadata': {
            'fallback_mode': True,
            'common_words_count': len(common_words)
        }
    }

@app.before_request
def log_request():
    request.start_time = time.time()
    request.request_id = str(uuid.uuid4())
    
    #validasi security headers
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
    user_agent = request.headers.get('User-Agent', 'Unknown')
    
    logger.info(f"Request {request.request_id}: {request.method} {request.url} - IP: {client_ip} - UA: {user_agent}")
    
    if request.content_length and request.content_length > config.MAX_FILE_SIZE:
        logger.warning(f"Request {request.request_id}: Content too large ({request.content_length} bytes)")

@app.after_request
def log_response(response):
    if hasattr(request, 'start_time'):
        process_time = time.time() - request.start_time
        logger.info(f"Request {getattr(request, 'request_id', 'unknown')} completed in {process_time:.2f}s - Status: {response.status_code}")
        
        response.headers["X-Request-ID"] = getattr(request, 'request_id', '')
        response.headers["X-Response-Time"] = f"{process_time:.2f}s"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
    return response

@app.errorhandler(400)
def bad_request(error):
    logger.warning(f"Bad request: {error.description}")
    return create_standard_response(
        False, "Bad request", 
        errors=[str(error.description)], 
        status_code=400
    )

@app.errorhandler(404)
def not_found(error):
    logger.info(f"404 error: {request.url}")
    return create_standard_response(
        False, "Endpoint not found", 
        errors=["The requested resource was not found"], 
        status_code=404
    )

@app.errorhandler(413)
def file_too_large(error):
    logger.warning(f"File too large: {request.content_length if hasattr(request, 'content_length') else 'unknown'} bytes")
    return create_standard_response(
        False, "File too large", 
        errors=[f"File size exceeds {config.MAX_FILE_SIZE // (1024*1024)}MB limit"], 
        status_code=413
    )

@app.errorhandler(429)
def rate_limit_exceeded(error):
    logger.warning(f"Rate limit exceeded for {request.remote_addr}")
    return create_standard_response(
        False, "Rate limit exceeded", 
        errors=["Too many requests. Please try again later."], 
        status_code=429
    )

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}", exc_info=True)
    return create_standard_response(
        False, "Internal server error", 
        errors=["An unexpected error occurred. Please try again later."], 
        status_code=500
    )

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "message": "Job Compatibility Analyzer REST API v2.2.0",
        "status": "operational",
        "documentation": "/api/docs",
        "health_check": "/api/health",
        "endpoints": {
            "health": "/api/health",
            "upload_cv": "/api/cv/upload",
            "analyze_cv_with_job": "/api/analyze/cv-with-job",
            "find_alternative_jobs": "/api/find-alternative-jobs"
        },
        "features": [
            "PDF CV processing",
            "ATS compatibility check",
            "Multi-language detection",
            "Skills matching",
            "Experience analysis",
            "Job recommendations"
        ],
        "supported_formats": ["PDF"],
        "max_file_size": f"{config.MAX_FILE_SIZE // (1024*1024)}MB"
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    system_status = {
        "status": "healthy",
        "analyzer_ready": analyzer is not None,
        "dataset_loaded": job_data is not None,
        "dataset_size": len(job_data) if job_data is not None else 0,
        "neural_network_enabled": config.ENABLE_NEURAL_NETWORK,
        "neural_network_trained": analyzer.is_nn_trained if analyzer else False,
        "upload_folder_exists": os.path.exists(config.UPLOAD_FOLDER),
        "temp_folder_exists": os.path.exists(config.TEMP_FOLDER),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "api_version": "2.2.0",
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "dependencies": {
            "flask": True,
            "pandas": True,
            "sklearn": True,
            "tensorflow": analyzer.nn_model is not None if analyzer else False
        }
    }
    
    is_healthy = all([
        system_status["analyzer_ready"] or True,
        system_status["dataset_loaded"],
        system_status["upload_folder_exists"]
    ])
    
    system_status["status"] = "healthy" if is_healthy else "degraded"
    
    return create_standard_response(
        True, f"Service is {system_status['status']}",
        data=system_status
    )

@app.route('/api/cv/upload', methods=['POST'])
@limiter.limit("10 per minute")
def upload_cv():
    temp_filepath = None
    try:
        if 'file' not in request.files:
            return create_standard_response(
                False, "No file provided",
                errors=["File is required"],
                status_code=400
            )
        
        file = request.files['file']
        is_valid, validation_message = validate_file_upload(file)
        
        if not is_valid:
            return create_standard_response(
                False, validation_message,
                errors=[validation_message],
                status_code=400
            )
        
        temp_filepath = create_temp_file()
        file.save(temp_filepath)
        
        with open(temp_filepath, "rb") as f:
            file_content = f.read()
        
        try:
            cv_text = extract_text_from_pdf(file_content)
            cv_text = clean_text_safe(cv_text)
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return create_standard_response(
                False, "Failed to extract text from PDF",
                errors=["PDF file may be corrupted or password protected"],
                status_code=400
            )
        
        if len(cv_text.strip()) < 50:
            return create_standard_response(
                False, "Insufficient CV content",
                errors=["CV must contain readable text (minimum 50 characters)"],
                status_code=400
            )
        
        #mengecek bahasa
        language_result = validate_english_text(cv_text, text_type="CV")
        if language_result['detected_language'] != 'en':
            return create_standard_response(
                False, language_result['message'],
                errors=[language_result['message']],
                status_code=400
            )
        
        #mengecek ATS friendlys
        ats_compatible, ats_issues, ats_score = is_ats_friendly(cv_text)
        
        response_data = {
            "cv_text": cv_text,
            "ats_score": float(ats_score),
            "ats_compatible": ats_compatible,
            "word_count": len(cv_text.split()),
            "character_count": len(cv_text),
            "language_detected": language_result.get('detected_language', 'en'),
            "readability_score": min(ats_score / 100, 1.0),
            "file_info": {
                "filename": secure_filename(file.filename),
                "size_bytes": len(file_content),
                "processing_time": time.time() - request.start_time
            }
        }
        
        if not ats_compatible:
            response_data["ats_issues"] = ats_issues
            return create_standard_response(
                False, "CV needs improvement for ATS compatibility",
                data=response_data,
                errors=ats_issues,
                status_code=400
            )
        
        return create_standard_response(
            True, "CV processed successfully",
            data=response_data
        )
    
    except Exception as e:
        logger.error(f"CV upload error: {e}", exc_info=True)
        return create_standard_response(
            False, "CV processing failed",
            errors=[f"Server error: {str(e)}"],
            status_code=500
        )
    
    finally:
        if temp_filepath:
            cleanup_temp_file(temp_filepath)

@app.route('/api/analyze/cv-with-job', methods=['POST'])
@limiter.limit("10 per minute")
def analyze_cv_with_job():
    temp_filepath = None
    try:
        if 'file' not in request.files:
            return create_standard_response(
                False, "No file provided",
                errors=["File is required"],
                status_code=400
            )
        
        file = request.files['file']
        job_title = request.form.get('job_title', '').strip()
        job_description = request.form.get('job_description', '').strip()
        
        validation_errors = []
        
        if not job_title or len(job_title) < 3:
            validation_errors.append("Job title must be at least 3 characters")
        
        if len(job_title) > 200:
            validation_errors.append("Job title too long (max 200 characters)")
        
        if job_title:
            job_title_lang_result = validate_english_text(job_title, text_type="Job Title")
            if job_title_lang_result['detected_language'] != 'en':
                validation_errors.append(job_title_lang_result['message'])
        
        if not job_description or len(job_description.split()) < 5:
            validation_errors.append("Job description must be at least 5 words")
        
        if len(job_description) > 10000:
            validation_errors.append("Job description too long (max 10000 characters)")
        
        if job_description:
            job_lang_result = validate_english_text(job_description, text_type="Job Description")
            if job_lang_result['detected_language'] != 'en':
                validation_errors.append(job_lang_result['message'])
        
        if validation_errors:
            return create_standard_response(
                False, "Validation failed",
                errors=validation_errors,
                status_code=400
            )
        
        is_valid, validation_message = validate_file_upload(file)
        if not is_valid:
            return create_standard_response(
                False, validation_message,
                errors=[validation_message],
                status_code=400
            )
        
        temp_filepath = create_temp_file()
        file.save(temp_filepath)
        
        with open(temp_filepath, 'rb') as f:
            file_content = f.read()
        
        cv_text = extract_text_from_pdf(file_content)
        cv_text = clean_text_safe(cv_text)
        
        if len(cv_text.split()) < 10:
            return create_standard_response(
                False, "CV content insufficient",
                errors=["CV must contain readable text (minimum 10 words)"],
                status_code=400
            )
        
        #validasi bahasa 
        cv_lang_result = validate_english_text(cv_text, text_type="CV")
        if cv_lang_result['detected_language'] != 'en':
            return create_standard_response(
                False, cv_lang_result['message'],
                errors=[cv_lang_result['message']],
                status_code=400
            )
        
        #membersihkan data input
        job_title = clean_text_safe(job_title)
        job_description = clean_text_safe(job_description)
        
        #memeriksa format apakah ATS
        ats_compatible, ats_issues, ats_score = is_ats_friendly(cv_text)
        if not ats_compatible:
            return create_standard_response(
                False, "CV needs improvement for ATS compatibility",
                errors=ats_issues,
                data={
                    "ats_score": float(ats_score),
                    "ats_issues": ats_issues,
                    "word_count": len(cv_text.split()),
                    "cv_preview": cv_text[:200] + "..." if len(cv_text) > 200 else cv_text
                },
                status_code=400
            )
        
        analysis_result = perform_analysis(cv_text, job_title, job_description)

        if (analysis_result.get('recommendation_level') == 'LANGUAGE_ERROR' 
            or analysis_result.get('analysis_metadata', {}).get('language_error')):
            
            error_message = analysis_result.get('analysis_metadata', {}).get('error_message', 'Language validation failed')
            return create_standard_response(
                False, "Language validation failed",
                errors=[error_message],
                status_code=400
            )
        
        response_data = {
            "cv_analysis": {
                "ats_score": float(ats_score),
                "ats_compatible": ats_compatible,
                "word_count": len(cv_text.split()),
                "character_count": len(cv_text),
                "language_detected": "en"
            },
            "job_analysis": {
                "title": job_title,
                "description_word_count": len(job_description.split()),
                "description_length": len(job_description)
            },
            "compatibility_analysis": analysis_result,
            "processing_info": {
                "analysis_time": time.time() - request.start_time,
                "analyzer_version": "2.2.0",
                "neural_network_used": analyzer.is_nn_trained if analyzer else False
            }
        }
        
        return create_standard_response(
            True, "CV analysis and job compatibility completed successfully",
            data=response_data
        )
        
    except Exception as e:
        logger.error(f"CV with job analysis error: {e}", exc_info=True)
        return create_standard_response(
            False, "Analysis failed",
            errors=[f"Analysis error: {str(e)}"],
            status_code=500
        )
    
    finally:
        if temp_filepath:
            cleanup_temp_file(temp_filepath)

@app.route('/api/find-alternative-jobs', methods=['POST'])
@limiter.limit("10 per minute")
def find_alternative_jobs():
    try:
        #validasi jasa json
        if not request.is_json:
            return create_standard_response(
                False, "Invalid content type",
                errors=["Request must be JSON"],
                status_code=400
            )
        
        data = request.get_json()
        
        if not data:
            return create_standard_response(
                False, "No JSON data provided",
                errors=["Request must contain JSON data"],
                status_code=400
            )
        
        required_fields = ['cv_text', 'analysis_results']
        missing_fields = [field for field in required_fields if not data.get(field)]
        
        if missing_fields:
            return create_standard_response(
                False, "Missing required fields",
                errors=[f"Missing fields: {', '.join(missing_fields)}"],
                status_code=400
            )
        
        #mengecek panjang teks
        cv_text = clean_text_safe(data['cv_text'])
        if len(cv_text.split()) < 10:
            return create_standard_response(
                False, "CV text too short",
                errors=["CV text must contain at least 10 words"],
                status_code=400
            )
        
        analysis_results = data['analysis_results']
        top_n = min(max(data.get('top_n', 6), 1), 25)  
        
        if job_data is None or len(job_data) == 0:
            return create_standard_response(
                False, "Job dataset not available",
                errors=["The job dataset is not loaded or empty"],
                status_code=503
            )
        
        try:
            alternative_jobs = find_alternative_jobs_for_cv(
                job_data, cv_text, analysis_results, top_n
            )
            
            response_data = {
                "recommended_jobs": alternative_jobs,
                "search_parameters": {
                    "top_n": top_n,
                    "dataset_size": len(job_data),
                    "cv_word_count": len(cv_text.split())
                },
                "metadata": {
                    "search_time": time.time() - request.start_time,
                    "algorithm_version": "2.2.0"
                }
            }
            
            return create_standard_response(
                True, f"Found {len(alternative_jobs)} alternative job recommendations",
                data=response_data
            )
            
        except Exception as e:
            logger.error(f"Job search error: {e}")
            return create_standard_response(
                False, "Job search failed",
                errors=[f"Search algorithm error: {str(e)}"],
                status_code=500
            )
    
    except Exception as e:
        logger.error(f"Find alternative jobs error: {e}", exc_info=True)
        return create_standard_response(
            False, "Error processing job search request",
            errors=[f"Request processing error: {str(e)}"],
            status_code=500
        )

@app.route('/api/docs', methods=['GET'])
def api_documentation():
    docs = {
        "title": "Searcheer API ML",
        "version": "2.2.0",
        "base_url": request.url_root.rstrip('/'),
        "endpoints": {
            "GET /": "endpoint utama",
            "GET /api/health": "mengecek status sistem",
            "POST /api/cv/upload": "upload CV dan memeriksa format ATS",
            "POST /api/analyze/cv-with-job": "analisis kecocokan cv dan job",
            "POST /api/find-alternative-jobs": "rekomendasi pekerjaan relevan",
            "GET /api/docs": "endpoint dokumentasi"
        },
        "features": [
            "ekstraksi teks dari PDF CV",
            "memeriksa format ATS",
            "analisis kecocokan CV dengan pekerjaan",
            "prediksi pekerjaan relevan"
        ],
        "rate_limits": {
            "default": "100 req per jam, 20 req per menit",
            "upload_cv": "10 req per menit",
            "analyze_cv_with_job": "10 req per menit",
            "find_alternative_jobs": "10 req per menit"
        },
        "file_requirements": {
            "format": "PDF only",
            "max_size": f"{config.MAX_FILE_SIZE // (1024*1024)}MB",
            "language": "harus dalam bahasa inggris",
            "content": "minimal 50 karakter"
        },
        "response_format": {
            "success": "boolean",
            "message": "string",
            "data": "object|null",
            "errors": "array",
            "timestamp": "ISO 8601 UTC timestamp",
            "request_id": "unique request identifier",
            "api_version": "API version"
        },
        "contact": {
            "support": "API support information",
            "documentation": f"{request.url_root}api/docs",
            "health_check": f"{request.url_root}api/health"
        }
    }
    return jsonify(docs)

def create_app():
    try:
        directories = [config.UPLOAD_FOLDER, config.TEMP_FOLDER, 'logs']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Directory ensured: {directory}")
        
        app.config['MAX_CONTENT_LENGTH'] = config.MAX_FILE_SIZE
        app.config['SECRET_KEY'] = config.SECRET_KEY
        app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
        
        if config.SECRET_KEY == "your-secret-key-change-this":
            logger.warning("Using default secret key! Change this in production!")
        
        logger.info("Initializing job analyzer...")
        initialize_analyzer()
        
        if job_data is None:
            logger.error("Failed to load job dataset!")
        else:
            logger.info(f"Application initialized successfully with {len(job_data)} jobs")
        
        return app
        
    except Exception as e:
        logger.error(f"Application creation failed: {e}", exc_info=True)
        raise

def cleanup_on_exit():
    try:
        if os.path.exists(config.TEMP_FOLDER):
            shutil.rmtree(config.TEMP_FOLDER, ignore_errors=True)
            logger.info("Cleaned up temporary files")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

if __name__ == '__main__':
    import atexit
    atexit.register(cleanup_on_exit)

    try:
        application = create_app()
        if config.DEBUG:
            logger.info("Starting development server...")
            application.run(
                host='0.0.0.0',
                port=int(os.getenv('PORT', 8000)),
                debug=True,
                threaded=True,
                use_reloader=False  
            )
        else:
            logger.info("Starting production server...")
            application.run(
                host='0.0.0.0',
                port=int(os.getenv('PORT', 8000)),
                debug=False,
                threaded=True
            )
            
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        exit(1)
