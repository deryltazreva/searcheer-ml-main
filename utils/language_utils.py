import re
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0


def detect_language(text):
    if not text or len(text.strip()) < 10:
        return "unknown"
    try:
        cleaned_text = re.sub(r'[^\w\s]', ' ', text)
        cleaned_text = ' '.join(cleaned_text.split())
        if len(cleaned_text) < 10:
            return "unknown"
        return detect(cleaned_text)
    except:
        return "unknown"

def validate_english_text(text, text_type="text", min_confidence=0.8):
    min_length = 50 if text_type.lower() in ["cv", "job description"] else 5
    if not text or len(text.strip()) < min_length:
        return {
            'is_english': False,
            'detected_language': 'unknown',
            'confidence': 0.0,
            'message': f"❌ {text_type} too short (min {min_length} chars)"
        }
    detected_lang = detect_language(text)
    is_english = detected_lang == 'en'
    confidence = 0.9 if is_english else 0.1
    return {
        'is_english': is_english,
        'detected_language': detected_lang,
        'confidence': confidence,
        'message': (
            f"{'✅' if is_english else '❌'} {text_type} detected as {detected_lang.upper()}"
            if is_english else
            (
                "❌ {0} detected as {1}\n\n"
                "Why English is important:\n"
                "   ✅ Most ATS systems are optimized for English language\n"
                "   ✅ Keyword matching will be more accurate\n"
                "   ✅ Increases compatibility with various platforms\n"
                "   ✅ International standard for professionals CVs\n"
                "   ✅ Expands job opportunities globally\n"
                "   ✅ Better parsing of technical terms and skills"
            ).format(text_type, detected_lang.upper())
        )
    }


