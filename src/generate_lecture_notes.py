"""
Smart Notes Generator - Lecture Notes Generation Module
Generates comprehensive lecture notes from slides and audio using LLM
Supports: Hindi/English audio → English notes
Uses: Free Google Gemini API + PaddleOCR + Faster-Whisper
Author: Smart Notes Generator Team
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import logging
import argparse
import time
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration settings for notes generation"""
    # API Keys (set as environment variable or paste here)
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyB67rUPGmWis56fWs1D3-wAIigKRddTv_0')
    
    # Model settings
    GEMINI_MODEL = 'gemini-2.5-flash'  # January 2026 free tier (250 req/day)
    WHISPER_MODEL = 'tiny'  # tiny/base/small/medium/large - tiny for testing
    
    # Language settings
    AUDIO_LANGUAGE = 'auto'  # 'auto', 'hi', 'en'
    OUTPUT_LANGUAGE = 'en'   # Always English notes
    
    # Processing settings
    MAX_RETRIES = 3
    RATE_LIMIT_DELAY = 2  # seconds between API calls


class IntelligentOCR:
    """
    Advanced OCR with preprocessing for lecture slides
    """
    def __init__(self, lang='en'):
        logger.info("Initializing OCR engine...")
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            logger.error("PaddleOCR not installed. Run: pip install paddleocr")
            raise
        
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang
        )
        logger.info("✓ OCR engine ready!")
    
    def preprocess_image(self, image_path):
        """Enhance image for better OCR"""
        img = cv2.imread(str(image_path))
        
        if img is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Increase contrast (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        return binary
    
    def extract_text(self, image_path):
        """Extract text from image with spatial information"""
        processed = self.preprocess_image(image_path)
        
        if processed is None:
            return {"text": "", "raw_text": "", "confidence": 0.0, "blocks": []}
        
        try:
            # Note: Removed 'cls=True' parameter - not supported in newer PaddleOCR versions
            result = self.ocr.ocr(processed)
            
            if not result or not result[0]:
                return {"text": "", "raw_text": "", "confidence": 0.0, "blocks": []}
            
            blocks = []
            full_text = []
            confidences = []
            
            for line in result[0]:
                text = line[1][0]
                conf = line[1][1]
                bbox = line[0]
                
                # Y-position for sorting
                y_pos = (bbox[0][1] + bbox[2][1]) / 2
                
                blocks.append({
                    'text': text,
                    'confidence': conf,
                    'y_position': y_pos
                })
                
                full_text.append(text)
                confidences.append(conf)
            
            # Sort by vertical position
            blocks.sort(key=lambda x: x['y_position'])
            
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                'text': '\n'.join([b['text'] for b in blocks]),
                'raw_text': ' '.join(full_text),
                'confidence': avg_confidence,
                'blocks': blocks
            }
        
        except Exception as e:
            logger.warning(f"OCR error for {image_path}: {e}")
            return {"text": "", "raw_text": "", "confidence": 0.0, "blocks": []}
    
    def process_slides_folder(self, slides_dir):
        """Process all slides in folder"""
        slides_dir = Path(slides_dir)
        
        if not slides_dir.exists():
            logger.error(f"Slides folder not found: {slides_dir}")
            return []
        
        # Get all image files
        image_files = sorted(
            list(slides_dir.glob('*.jpg')) + 
            list(slides_dir.glob('*.png')) +
            list(slides_dir.glob('*.jpeg'))
        )
        
        if not image_files:
            logger.error(f"No images found in {slides_dir}")
            return []
        
        logger.info(f"Processing {len(image_files)} slides with OCR...")
        
        results = []
        for img_path in tqdm(image_files, desc="Extracting text"):
            ocr_result = self.extract_text(img_path)
            
            results.append({
                'image_path': str(img_path),
                'image_name': img_path.name,
                'ocr_text': ocr_result['text'],
                'ocr_raw': ocr_result['raw_text'],
                'confidence': ocr_result['confidence'],
                'blocks': ocr_result['blocks']
            })
        
        return results


class AudioTranscriber:
    """
    Transcribe audio with multi-language support
    """
    def __init__(self, model_size='large-v3'):
        logger.info(f"Loading Whisper model ({model_size})...")
        try:
            import whisper
        except ImportError:
            logger.error("openai-whisper not installed. Run: pip install openai-whisper")
            raise
        
        # Map model size (large-v3 -> large for openai-whisper)
        if 'v3' in model_size or 'v2' in model_size:
            model_size = model_size.split('-')[0]  # 'large-v3' -> 'large'
        
        self.model = whisper.load_model(model_size)
        logger.info("✓ Whisper model ready!")
    
    def transcribe(self, audio_path, language='auto'):
        """
        Transcribe audio to text with timestamps
        Supports: auto-detect, Hindi, English
        """
        logger.info(f"Transcribing audio (language: {language})...")
        
        # Prepare whisper parameters
        transcribe_params = {
            'verbose': False,
            'word_timestamps': True
        }
        
        if language != 'auto':
            transcribe_params['language'] = language
        
        try:
            result = self.model.transcribe(
                str(audio_path),
                **transcribe_params
            )
            
            detected_lang = result.get('language', 'unknown')
            logger.info(f"Detected language: {detected_lang}")
            
            transcript_data = []
            
            for segment in result['segments']:
                # Extract words with timestamps if available
                words = []
                if 'words' in segment and segment['words']:
                    words = [
                        {
                            'word': word.get('word', ''),
                            'start': word.get('start', 0),
                            'end': word.get('end', 0)
                        }
                        for word in segment['words']
                    ]
                
                transcript_data.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip(),
                    'words': words
                })
            
            return {
                'segments': transcript_data,
                'language': detected_lang,
                'full_text': ' '.join([s['text'] for s in transcript_data])
            }
        
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {
                'segments': [],
                'language': 'unknown',
                'full_text': ''
            }
    
    def align_to_slides(self, transcript, transitions_data):
        """Align transcript segments to slide timestamps"""
        
        if not transcript['segments']:
            logger.warning("No transcript segments to align")
            return []
        
        logger.info("Aligning transcript to slides...")
        
        aligned = []
        
        for i, transition in enumerate(transitions_data):
            start_time = transition.get('timestamp', 0)
            
            # Get end time (next slide or end of video)
            if i < len(transitions_data) - 1:
                end_time = transitions_data[i + 1].get('timestamp', float('inf'))
            else:
                # Last slide - use last transcript segment
                end_time = transcript['segments'][-1]['end'] if transcript['segments'] else start_time + 60
            
            # Find segments in this time range
            slide_segments = [
                seg for seg in transcript['segments']
                if seg['start'] >= start_time and seg['end'] <= end_time
            ]
            
            # Combine text
            combined_text = ' '.join([s['text'] for s in slide_segments])
            
            aligned.append({
                'slide_number': i + 1,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'transcript': combined_text,
                'segments': slide_segments,
                'num_segments': len(slide_segments)
            })
        
        logger.info(f"Aligned {len(aligned)} slides with transcripts")
        return aligned


class FreeGeminiLLM:
    """
    Free Google Gemini LLM for notes generation
    """
    def __init__(self, api_key=None):
        api_key = api_key or Config.GOOGLE_API_KEY
        
        if api_key == 'YOUR_API_KEY_HERE':
            logger.warning("="*60)
            logger.warning("GOOGLE API KEY NOT SET!")
            logger.warning("Get free key at: https://aistudio.google.com/app/apikey")
            logger.warning("Set as environment variable: export GOOGLE_API_KEY='your-key'")
            logger.warning("Or paste directly in Config.GOOGLE_API_KEY")
            logger.warning("="*60)
            raise ValueError("Google API key required")
        
        try:
            import google.generativeai as genai
        except ImportError:
            logger.error("google-generativeai not installed. Run: pip install google-generativeai")
            raise
        
        genai.configure(api_key=api_key)
        
        # Try different model names to find one that works (Jan 2026 free tier models)
        models_to_try = [
            Config.GEMINI_MODEL,
            'gemini-2.5-flash',      # 250 req/day - best for notes
            'gemini-2.5-flash-lite', # 1000 req/day - faster, lighter
            'gemini-1.5-pro'         # Legacy free tier
        ]
        
        self.model = None
        for model_name in models_to_try:
            try:
                self.model = genai.GenerativeModel(model_name)
                # Test the model with a simple prompt
                test_response = self.model.generate_content("test")
                logger.info(f"✓ Gemini LLM ready! (Model: {model_name})")
                break
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        if self.model is None:
            raise RuntimeError("Could not initialize any Gemini model. Check your API key and internet connection.")
    
    def clean_ocr(self, raw_ocr, audio_context, detected_language='unknown'):
        """Clean OCR text using audio context"""
        
        if not raw_ocr.strip():
            return raw_ocr
        
        prompt = f"""You are an OCR text cleaner for educational content.

**Raw OCR Text:**
{raw_ocr}

**Audio Context (teacher's words):**
{audio_context}

**Audio Language:** {detected_language}

**Task:**
Fix OCR errors by:
1. Joining fragmented words (e.g., "P h y s i c s" → "Physics")
2. Correcting character misidentifications (0/O, 1/l, 5/S)
3. Using audio context to resolve ambiguities
4. If audio is in Hindi/other language, use it to understand context but output in ENGLISH
5. Preserve mathematical notation carefully
6. Keep original structure and line breaks

**Constraints:**
- Output ONLY the cleaned text
- Do NOT add information not in OCR/audio
- If uncertain, keep original text
- Preserve technical terms exactly

**Output the cleaned text:**"""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning(f"OCR cleaning error: {e}")
            return raw_ocr
    
    def generate_notes(self, slide_data, detected_language='unknown'):
        """Generate structured notes from slide + audio"""
        
        prompt = f"""You are an expert educational note-taker creating lecture notes.

**SLIDE {slide_data['slide_number']}**

**Slide Text (OCR):**
{slide_data['cleaned_ocr']}

**Teacher's Explanation:**
{slide_data['transcript']}

**Audio Language:** {detected_language}
**Timestamp:** {slide_data['start_time']:.1f}s - {slide_data['end_time']:.1f}s

**Task:** Create structured lecture notes in ENGLISH following these rules:

1. **STRUCTURE:**
   - Start with clear heading (## format)
   - Use bullet points for key concepts
   - Use numbered lists for processes/steps
   - Create subsections if content is complex

2. **CONTENT RULES:**
   - Use ONLY information from slide and audio
   - If audio is Hindi/other language, translate concepts to English
   - Do NOT add external facts
   - Preserve technical terms
   - If teacher emphasizes "important" or "exam", highlight with ⚠️
   - Format math as LaTeX: $E=mc^2$

3. **VISUAL ENHANCEMENT:**
   - **Bold** for key terms (first mention)
   - `code` for technical syntax
   - Definition boxes: > **Definition:** [term] - [explanation]
   - If describing a process, add: "📊 *[Process described]*"

4. **LANGUAGE HANDLING:**
   - If audio is Hindi, translate to English naturally
   - Keep original Hindi technical terms in parentheses if needed
   - Example: "Teacher explained velocity (वेग) as rate of change..."

5. **ORIGINALITY:**
   - NO external examples
   - NO elaboration beyond teacher's words
   - If unclear: "(unclear in recording)"

**Output pure Markdown, start with heading:**"""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Notes generation error for slide {slide_data['slide_number']}: {e}")
            return f"## Slide {slide_data['slide_number']}\n\n*Error generating notes*"
    
    def generate_summary(self, all_notes):
        """Generate executive summary"""
        
        prompt = f"""Create a brief executive summary (2-3 paragraphs) of this lecture.

**All Lecture Notes:**
{all_notes}

**Include:**
1. Main topic
2. Key concepts covered (3-5 points)
3. Important takeaways

**Format:** Pure text, no markdown headers. Keep it concise.

**Summary:**"""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Summary generation error: {e}")
            return "*Summary generation failed*"


class NotesGenerator:
    """
    Main notes generation orchestrator
    """
    def __init__(self, lecture_folder, google_api_key=None):
        self.lecture_folder = Path(lecture_folder)
        self.slides_dir = self.lecture_folder / 'slides'
        self.audio_dir = self.lecture_folder / 'audio'
        
        # Validate folders
        if not self.lecture_folder.exists():
            raise FileNotFoundError(f"Lecture folder not found: {lecture_folder}")
        if not self.slides_dir.exists():
            raise FileNotFoundError(f"Slides folder not found: {self.slides_dir}")
        
        # Load metadata
        metadata_path = self.lecture_folder / 'metadata.json'
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    self.metadata = json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Error loading metadata.json: {e}, using empty metadata")
                self.metadata = {}
        else:
            self.metadata = {}
        
        # Load transitions
        transitions_path = self.lecture_folder / 'transitions.json'
        if transitions_path.exists():
            try:
                with open(transitions_path) as f:
                    content = f.read().strip()
                    if content:
                        data = json.loads(content)
                        # Handle both formats: direct array or wrapped in 'transitions' key
                        if isinstance(data, dict) and 'transitions' in data:
                            self.transitions = data['transitions']
                        elif isinstance(data, list):
                            self.transitions = data
                        else:
                            logger.warning("Unexpected transitions.json format, will create dummy transitions")
                            self.transitions = []
                    else:
                        logger.warning("transitions.json is empty, will create dummy transitions")
                        self.transitions = []
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Error loading transitions.json: {e}, will create dummy transitions")
                self.transitions = []
        else:
            self.transitions = []
            logger.warning("No transitions.json found, will create dummy transitions")
        
        # Initialize components
        self.ocr = IntelligentOCR()
        self.transcriber = AudioTranscriber(Config.WHISPER_MODEL)
        self.llm = FreeGeminiLLM(google_api_key)
    
    def find_audio_file(self):
        """Find audio file in audio folder"""
        if not self.audio_dir.exists():
            logger.warning(f"Audio folder not found: {self.audio_dir}")
            return None
        
        audio_formats = ['.mp3', '.wav', '.m4a', '.mp4', '.avi']
        
        for ext in audio_formats:
            files = list(self.audio_dir.glob(f'*{ext}'))
            if files:
                return files[0]
        
        return None
    
    def process(self):
        """Main processing pipeline"""
        
        logger.info("="*70)
        logger.info("LECTURE NOTES GENERATION SYSTEM")
        logger.info("="*70)
        
        # Define checkpoint files
        ocr_cache = self.lecture_folder / 'ocr_cache.json'
        transcript_cache = self.lecture_folder / 'transcript_cache.json'
        
        # Step 1: OCR from slides
        logger.info("\n[STEP 1/5] OCR Extraction")
        logger.info("-" * 70)
        
        if ocr_cache.exists():
            logger.info("Loading OCR results from cache...")
            with open(ocr_cache, 'r', encoding='utf-8') as f:
                ocr_results = json.load(f)
            logger.info(f"✓ Loaded {len(ocr_results)} cached OCR results")
        else:
            ocr_results = self.ocr.process_slides_folder(self.slides_dir)
            
            if not ocr_results:
                logger.error("No slides found. Exiting.")
                return None
            
            # Save OCR cache
            with open(ocr_cache, 'w', encoding='utf-8') as f:
                json.dump(ocr_results, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Saved OCR results to cache")
        
        # Step 2: Audio transcription
        logger.info("\n[STEP 2/5] Audio Transcription")
        logger.info("-" * 70)
        
        if transcript_cache.exists():
            logger.info("Loading transcript from cache...")
            with open(transcript_cache, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
            detected_language = transcript.get('language', 'unknown')
            logger.info(f"✓ Loaded cached transcript (language: {detected_language})")
        else:
            audio_file = self.find_audio_file()
            
            if audio_file:
                logger.info(f"Audio file: {audio_file.name}")
                transcript = self.transcriber.transcribe(
                    audio_file,
                    language=Config.AUDIO_LANGUAGE
                )
                detected_language = transcript.get('language', 'unknown')
                
                # Save transcript cache
                with open(transcript_cache, 'w', encoding='utf-8') as f:
                    json.dump(transcript, f, indent=2, ensure_ascii=False)
                logger.info(f"✓ Saved transcript to cache")
            else:
                logger.warning("No audio file found, proceeding without transcription")
                transcript = {'segments': [], 'language': 'unknown', 'full_text': ''}
                detected_language = 'unknown'
        
        # Step 3: Align transcript to slides
        logger.info("\n[STEP 3/5] Aligning Transcript to Slides")
        logger.info("-" * 70)
        
        if self.transitions:
            aligned_transcript = self.transcriber.align_to_slides(
                transcript,
                self.transitions
            )
        else:
            # Create dummy transitions from slide count
            aligned_transcript = []
            duration_per_slide = 60  # Default 1 minute
            
            for i, ocr_result in enumerate(ocr_results):
                aligned_transcript.append({
                    'slide_number': i + 1,
                    'start_time': i * duration_per_slide,
                    'end_time': (i + 1) * duration_per_slide,
                    'duration': duration_per_slide,
                    'transcript': '',
                    'segments': []
                })
        
        # Step 4: Clean OCR and generate notes
        logger.info("\n[STEP 4/5] Generating Notes with LLM")
        logger.info("-" * 70)
        
        all_slide_data = []
        
        for i, (ocr_res, trans_res) in enumerate(zip(ocr_results, aligned_transcript)):
            logger.info(f"Processing Slide {i+1}/{len(ocr_results)}...")
            
            # Clean OCR
            cleaned_ocr = self.llm.clean_ocr(
                ocr_res['ocr_text'],
                trans_res['transcript'],
                detected_language
            )
            
            # Generate notes
            slide_data = {
                'slide_number': i + 1,
                'image_path': ocr_res['image_path'],
                'image_name': ocr_res['image_name'],
                'raw_ocr': ocr_res['ocr_text'],
                'cleaned_ocr': cleaned_ocr,
                'transcript': trans_res['transcript'],
                'start_time': trans_res['start_time'],
                'end_time': trans_res['end_time'],
                'duration': trans_res['duration']
            }
            
            notes = self.llm.generate_notes(slide_data, detected_language)
            slide_data['notes'] = notes
            
            all_slide_data.append(slide_data)
            
            # Rate limiting for free tier
            time.sleep(Config.RATE_LIMIT_DELAY)
        
        # Step 5: Generate summary and format
        logger.info("\n[STEP 5/5] Creating Final Document")
        logger.info("-" * 70)
        
        # Combine all notes
        all_notes_text = '\n\n'.join([s['notes'] for s in all_slide_data])
        
        # Generate summary
        summary = self.llm.generate_summary(all_notes_text)
        
        # Create final output
        output = self.create_markdown_output(all_slide_data, summary, detected_language)
        
        # Save outputs
        self.save_outputs(output, all_slide_data)
        
        logger.info("\n" + "="*70)
        logger.info("PROCESSING COMPLETE!")
        logger.info("="*70)
        
        return output
    
    def create_markdown_output(self, slides_data, summary, detected_language):
        """Create formatted Markdown document"""
        
        md = []
        
        # Title
        lecture_title = self.metadata.get('title', self.lecture_folder.name.replace('_', ' ').title())
        md.append(f"# {lecture_title}\n")
        
        # Metadata
        md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        md.append(f"**Source Language:** {detected_language.upper()}")
        md.append(f"**Notes Language:** English")
        md.append(f"**Total Slides:** {len(slides_data)}")
        md.append("\n---\n")
        
        # Summary
        md.append("## 📋 Executive Summary\n")
        md.append(summary)
        md.append("\n---\n")
        
        # Table of Contents
        md.append("## 📑 Table of Contents\n")
        for slide in slides_data:
            # Extract first heading
            first_heading = slide['notes'].split('\n')[0].replace('## ', '').replace('#', '')
            md.append(
                f"{slide['slide_number']}. [{first_heading}]"
                f"(#slide-{slide['slide_number']}) "
                f"(⏱️ {slide['start_time']:.0f}s)"
            )
        md.append("\n---\n")
        
        # Detailed notes
        for slide in slides_data:
            # Anchor
            md.append(f"<a id='slide-{slide['slide_number']}'></a>\n")
            
            # Slide header
            md.append(f"### Slide {slide['slide_number']}")
            md.append(f"**Time:** {slide['start_time']:.1f}s - {slide['end_time']:.1f}s")
            md.append(f"**Duration:** {slide['duration']:.1f}s\n")
            
            # Image
            rel_path = Path(slide['image_path']).relative_to(self.lecture_folder)
            md.append(f"![Slide {slide['slide_number']}]({rel_path})\n")
            
            # Notes
            md.append(slide['notes'])
            md.append("\n---\n")
        
        # Footer
        md.append("\n## 📝 Generation Details\n")
        md.append("- **OCR:** PaddleOCR")
        md.append("- **Transcription:** Faster-Whisper (Large-v3)")
        md.append("- **LLM:** Google Gemini 1.5 Flash (Free)")
        md.append(f"- **Project:** {self.lecture_folder.name}")
        
        return '\n'.join(md)
    
    def save_outputs(self, markdown_content, slides_data):
        """Save all output files"""
        
        # Save Markdown
        md_path = self.lecture_folder / 'lecture_notes.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        logger.info(f"✓ Markdown saved: {md_path.name}")
        
        # Save JSON
        json_path = self.lecture_folder / 'notes_data.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(slides_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ JSON data saved: {json_path.name}")
        
        # Try to generate PDF
        try:
            pdf_path = self.create_pdf(markdown_content)
            logger.info(f"✓ PDF saved: {pdf_path.name}")
        except Exception as e:
            logger.warning(f"PDF generation skipped: {e}")
            logger.info("   (Optional: Install weasyprint for PDF support)")
    
    def create_pdf(self, markdown_content):
        """Convert Markdown to PDF"""
        try:
            import markdown
            from weasyprint import HTML, CSS
        except ImportError:
            raise ImportError("Install: pip install markdown weasyprint")
        
        # Convert to HTML
        html_content = markdown.markdown(
            markdown_content,
            extensions=['extra', 'codehilite', 'toc']
        )
        
        # CSS styling
        css = CSS(string="""
            @page { margin: 2cm; }
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                line-height: 1.8;
                color: #333;
            }
            h1 {
                color: #1a73e8;
                border-bottom: 3px solid #1a73e8;
                padding-bottom: 10px;
            }
            h2 { color: #34495e; margin-top: 25px; }
            h3 { color: #7f8c8d; }
            img {
                max-width: 100%;
                border: 1px solid #ddd;
                padding: 5px;
                margin: 10px 0;
            }
            code {
                background: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
            }
            blockquote {
                border-left: 4px solid #1a73e8;
                padding-left: 15px;
                background: #f9f9f9;
                margin: 10px 0;
            }
        """)
        
        # Generate PDF
        html_full = f"<!DOCTYPE html><html><body>{html_content}</body></html>"
        pdf_path = self.lecture_folder / 'lecture_notes.pdf'
        HTML(string=html_full).write_pdf(pdf_path, stylesheets=[css])
        
        return pdf_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate lecture notes from processed slides and audio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/generate_lecture_notes.py data/lectures/dsa_pankaj_sir
  python src/generate_lecture_notes.py data/lectures/algo_1 --api-key YOUR_KEY
  
Environment:
  GOOGLE_API_KEY - Google Gemini API key (get free at https://aistudio.google.com/app/apikey)
        """
    )
    
    parser.add_argument(
        'lecture_folder',
        type=str,
        help='Path to processed lecture folder (e.g., data/lectures/dsa_pankaj_sir)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='Google Gemini API key (or set GOOGLE_API_KEY env variable)'
    )
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        os.environ['GOOGLE_API_KEY'] = args.api_key
        Config.GOOGLE_API_KEY = args.api_key
    
    try:
        # Process
        generator = NotesGenerator(args.lecture_folder, args.api_key)
        result = generator.process()
        
        if result:
            logger.info("\n" + "="*70)
            logger.info("Generated files:")
            logger.info(f"  - lecture_notes.md")
            logger.info(f"  - notes_data.json")
            logger.info(f"  - lecture_notes.pdf (if weasyprint installed)")
            logger.info("="*70)
            logger.info("\n✨ Done! Happy studying! ✨\n")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
