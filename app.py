import streamlit as st
import requests
import json
import time
import base64
import os
from gtts import gTTS
from io import BytesIO
import tempfile
import numpy as np
# Modified imports to handle potential dependency issues
try:
    from PIL import Image
    from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
except ImportError:
    st.error("Some required libraries couldn't be imported. Using fallback options.")
    Image = None

# Set page configuration
st.set_page_config(
    page_title="ROH-Ads: AI Marketing Strategy Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
GEMINI_API_KEY = "AIzaSyBFjG6kQWfrpg0Q7tcvxxQHNDl3DVW8-gA"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Initialize session state
if 'business_data' not in st.session_state:
    st.session_state.business_data = {
        "business_name": "",
        "industry": "",
        "target_audience": "",
        "marketing_goals": "",
        "budget_range": "",
        "current_challenges": "",
        "five_year_traction": "",  # Added field for 5-year traction plan
    }
if 'marketing_strategy' not in st.session_state:
    st.session_state.marketing_strategy = None
if 'tts_active' not in st.session_state:
    st.session_state.tts_active = False
if 'current_tts_text' not in st.session_state:
    st.session_state.current_tts_text = ""
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {
        'images': [],
        'videos': [],
        'audio': [],
        'documents': []  # Added documents category
    }
# Voice gender removed as we're only using female voice
if 'voice_speed' not in st.session_state:
    st.session_state.voice_speed = "Normal"
if 'language' not in st.session_state:
    st.session_state.language = "en"  # Default language is English
if 'ml_model' not in st.session_state:
    st.session_state.ml_model = None

# Language mapping for UI and TTS (simplified without male voice options)
LANGUAGES = {
    "en": {"name": "English", "code": "en-us"},
    "es": {"name": "Spanish", "code": "es-es"},
    "fr": {"name": "French", "code": "fr-fr"},
    "de": {"name": "German", "code": "de-de"},
    "it": {"name": "Italian", "code": "it-it"},
    "ja": {"name": "Japanese", "code": "ja-jp"},
    "ko": {"name": "Korean", "code": "ko-kr"},
    "pt": {"name": "Portuguese", "code": "pt-br"},
    "ru": {"name": "Russian", "code": "ru-ru"},
    "zh": {"name": "Chinese", "code": "zh-cn"}
}

# Initialize ML models for multilingual support with error handling
@st.cache_resource
def load_ml_models():
    """Load and cache ML models for multilingual support"""
    try:
        # Set Hugging Face token for authentication
        hf_token = "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"
        
        try:
            tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased", token=hf_token)
            model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-multilingual-cased", token=hf_token)
            fill_mask_pipeline = pipeline("fill-mask", model="google-bert/bert-base-multilingual-cased", token=hf_token)
            return {"tokenizer": tokenizer, "model": model, "fill_mask": fill_mask_pipeline}
        except Exception as e:
            st.warning(f"Could not load BERT model: {str(e)}. Using simplified functionality.")
            return {"error": str(e)}
    except Exception as e:
        st.error(f"Error initializing ML models: {str(e)}")
        return None

# TTS function with simplified voice options (female only)
def text_to_speech(text, speed="Normal", language="en"):
    """Convert text to speech in multiple languages and create an audio player"""
    if not text:
        return None
    
    try:
        # Configure TTS based on speed
        slow_option = False
        if speed == "Slow":
            slow_option = True
        
        # Use the language mapping for language support (female voice only)
        lang_data = LANGUAGES.get(language, LANGUAGES["en"])
        lang_code = lang_data["code"]
        
        # Create the TTS object
        tts = gTTS(text=text, lang=lang_code[:2], slow=slow_option)  # Use first 2 chars for language code
        
        # Save to a temporary file (better audio quality than BytesIO for some browsers)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            with open(fp.name, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            os.unlink(fp.name)  # Remove the temp file
        
        # Convert to base64 for the audio player
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_player = f'<audio autoplay controls><source src="data:audio/mp3;base64,{audio_base64}"></audio>'
        
        return audio_player
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None

def encode_image(image_file):
    """Convert an image file to base64 for Gemini API"""
    return base64.b64encode(image_file.read()).decode("utf-8")

def encode_media(media_file):
    """Convert a media file to base64 for Gemini API"""
    return base64.b64encode(media_file.read()).decode("utf-8")

def generate_with_gemini(prompt, media_files=None, language="en"):
    """Generate content using Gemini model with multimodal support and error handling"""
    try:
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }
        
        # Translate prompt to English for better results if not already in English
        if language != "en":
            try:
                translate_prompt = f"Translate the following text to English: {prompt}"
                # We'll use Gemini itself for translation
                translation_data = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": translate_prompt}]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.1,  # Lower temperature for more accurate translation
                        "topP": 0.95,
                        "topK": 40,
                        "maxOutputTokens": 4096
                    }
                }
                
                translation_response = requests.post(
                    GEMINI_API_URL,
                    headers=headers,
                    data=json.dumps(translation_data),
                    timeout=30  # Add timeout to prevent hanging
                )
                
                if translation_response.status_code == 200:
                    translation_result = translation_response.json()
                    prompt = translation_result["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    st.warning(f"Translation failed, using original prompt: {translation_response.status_code}")
            except Exception as e:
                st.warning(f"Translation error: {str(e)}. Using original prompt.")
        
        # Prepare the data structure for the API call with multimodal support
        parts = []
        
        # Add text prompt
        parts.append({"text": prompt})
        
        # Add media files if provided
        if media_files:
            try:
                for media_file in media_files:
                    try:
                        media_type = media_file.type
                        
                        if media_type.startswith('image'):
                            # Handle image
                            parts.append({
                                "inlineData": {
                                    "mimeType": media_file.type,
                                    "data": encode_media(media_file)
                                }
                            })
                        elif media_type.startswith('video'):
                            # Future support for video once Gemini supports it
                            st.info("Video analysis functionality is coming soon.")
                        elif media_type.startswith('audio'):
                            # Future support for audio once Gemini supports it
                            st.info("Audio analysis functionality is coming soon.")
                        elif media_type == 'application/pdf' or media_type.startswith('text'):
                            # Handle documents
                            try:
                                document_content = media_file.read().decode('utf-8', errors='replace')
                                # Limit document content length to avoid API limits
                                max_content_length = 10000  # Adjust based on Gemini's limits
                                if len(document_content) > max_content_length:
                                    document_content = document_content[:max_content_length] + "... [truncated]"
                                parts.append({"text": f"\nDocument content from {media_file.name}:\n{document_content}"})
                            except Exception as doc_err:
                                st.warning(f"Error processing document {media_file.name}: {str(doc_err)}")
                    except Exception as media_err:
                        st.warning(f"Error processing media file: {str(media_err)}")
            except Exception as media_list_err:
                st.warning(f"Error processing media files: {str(media_list_err)}")
        
        data = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 4096
            }
        }
        
        response = requests.post(
            GEMINI_API_URL,
            headers=headers,
            data=json.dumps(data),
            timeout=60  # Increased timeout for multimodal requests
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result["candidates"][0]["content"]["parts"][0]["text"]
            
            # Translate back to selected language if not English
            if language != "en":
                try:
                    back_translate_prompt = f"Translate the following text to {LANGUAGES[language]['name']}: {response_text}"
                    back_translation_data = {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [{"text": back_translate_prompt}]
                            }
                        ],
                        "generationConfig": {
                            "temperature": 0.1,
                            "topP": 0.95,
                            "topK": 40,
                            "maxOutputTokens": 4096
                        }
                    }
                    
                    back_translation_response = requests.post(
                        GEMINI_API_URL,
                        headers=headers,
                        data=json.dumps(back_translation_data),
                        timeout=30
                    )
                    
                    if back_translation_response.status_code == 200:
                        back_translation_result = back_translation_response.json()
                        response_text = back_translation_result["candidates"][0]["content"]["parts"][0]["text"]
                except Exception as back_translate_err:
                    st.warning(f"Back-translation error: {str(back_translate_err)}. Using English response.")
            
            return response_text
        else:
            error_message = f"API Error: {response.status_code}"
            try:
                error_details = response.json()
                error_message += f" - {error_details.get('error', {}).get('message', 'Unknown error')}"
            except:
                pass
            st.error(error_message)
            return f"Error: Could not generate content. Please try again later."
            
    except requests.exceptions.Timeout:
        st.error("Request to Gemini API timed out. Please try again.")
        return "The request timed out. Please try with a simpler query or fewer media files."
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return f"An error occurred: {str(e)}"

def analyze_media_for_autofill(media_files):
    """Analyze uploaded media files to extract business information for autofill"""
    if not media_files:
        return {}
    
    try:
        # Prepare a prompt for Gemini to analyze the media
        analysis_prompt = """
        Analyze the uploaded media and extract the following business information:
        1. Business name
        2. Industry type
        3. Current challenges or problems they might be facing
        4. Budget range (if visible)
        5. Potential 5-year growth trajectory and traction plan
        
        Format the response as a JSON object with these keys: 
        business_name, industry, current_challenges, budget_range, five_year_traction
        
        If you cannot determine any field, leave it as an empty string.
        """
        
        # Call Gemini API with the prompt and media files
        analysis_result = generate_with_gemini(analysis_prompt, media_files)
        
        # Extract the JSON from the response
        try:
            # Look for JSON structure in the response
            import re
            json_match = re.search(r'```json\n(.*?)\n```', analysis_result, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON-like structure without markdown
                json_match = re.search(r'(\{.*\})', analysis_result, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = analysis_result
            
            try:
                extracted_data = json.loads(json_str)
            except json.JSONDecodeError:
                # If JSON parsing fails, create a structured response manually
                st.warning("Could not parse the AI response as JSON. Using simpler extraction.")
                extracted_data = {}
                
                # Simple key-value extraction fallback
                if "business_name" in analysis_result.lower():
                    match = re.search(r'business[_\s]name["\s:]+([^"\n,]+)', analysis_result, re.IGNORECASE)
                    if match:
                        extracted_data["business_name"] = match.group(1).strip()
                
                if "industry" in analysis_result.lower():
                    match = re.search(r'industry["\s:]+([^"\n,]+)', analysis_result, re.IGNORECASE)
                    if match:
                        extracted_data["industry"] = match.group(1).strip()
            
            # Ensure we have all the expected keys
            expected_keys = ['business_name', 'industry', 'current_challenges', 'budget_range', 'five_year_traction']
            for key in expected_keys:
                if key not in extracted_data:
                    extracted_data[key] = ""
                    
            return extracted_data
            
        except Exception as e:
            st.error(f"Error parsing media analysis result: {str(e)}")
            return {}
    except Exception as e:
        st.error(f"Error in media analysis: {str(e)}")
        return {}

def save_uploaded_file(uploaded_file, file_type):
    """Save uploaded file and return the file path"""
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
    
    # Add to session state
    if file_type == "image":
        st.session_state.uploaded_files['images'].append(file_details)
    elif file_type == "video":
        st.session_state.uploaded_files['videos'].append(file_details)
    elif file_type == "audio":
        st.session_state.uploaded_files['audio'].append(file_details)
    elif file_type == "document":
        st.session_state.uploaded_files['documents'].append(file_details)
    
    return file_details

def display_file_preview(file, file_type):
    """Display a preview of the uploaded file"""
    if file_type == "image":
        st.image(file, caption=file.name, use_column_width=True)
    elif file_type == "video":
        st.video(file)
    elif file_type == "audio":
        st.audio(file)
    elif file_type == "document":
        # Display document info
        st.write(f"Document: {file.name}")
        try:
            # Try to show a preview for text documents
            if file.type.startswith('text'):
                text_content = file.read().decode('utf-8')
                st.code(text_content[:500] + '...' if len(text_content) > 500 else text_content)
        except:
            st.write("Preview not available for this document type")

# UI Components
def sidebar():
    with st.sidebar:
        st.title("🚀 ROH-Ads")
        st.subheader("AI Marketing Strategy Assistant")
        
        st.markdown("---")
        
        # Language Selector
        st.subheader("🌐 Language")
        language_options = {code: data["name"] for code, data in LANGUAGES.items()}
        selected_language = st.selectbox(
            "Select Language",
            options=list(language_options.keys()),
            format_func=lambda x: language_options[x],
            index=list(language_options.keys()).index(st.session_state.language)
        )
        
        if selected_language != st.session_state.language:
            st.session_state.language = selected_language
            st.experimental_rerun()
        
        # TTS Controls
        st.subheader("🔊 Text-to-Speech")
        st.session_state.tts_active = st.toggle("Enable AI Voice", value=st.session_state.tts_active)
        
        st.session_state.voice_speed = st.select_slider(
            "Voice Speed",
            options=["Slow", "Normal", "Fast"],
            value=st.session_state.voice_speed,
            disabled=not st.session_state.tts_active
        )
        
        if st.session_state.tts_active and st.button("Speak Current Analysis"):
            if st.session_state.current_tts_text:
                # Create a short summary for TTS to avoid long audio
                summary_prompt = f"""
                Create a 3-4 sentence summary of the key points from this content. Focus only on the most important takeaways:
                
                {st.session_state.current_tts_text[:1000]}...
                """
                with st.spinner("Generating audio summary..."):
                    summary = generate_with_gemini(summary_prompt, language=st.session_state.language)
                    audio_player = text_to_speech(
                        summary, 
                        speed=st.session_state.voice_speed,
                        language=st.session_state.language
                    )
                    if audio_player:
                        st.markdown(audio_player, unsafe_allow_html=True)
            else:
                st.warning("No analysis available to speak yet.")
        
        st.markdown("---")
        
        # File Upload Section
        st.subheader("📁 Media Upload")
        
        # Image Upload
        uploaded_image = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_image:
            for img in uploaded_image:
                save_uploaded_file(img, "image")
                st.success(f"Uploaded image: {img.name}")
                
        # Video Upload
        uploaded_video = st.file_uploader("Upload Videos", type=["mp4", "mov", "avi"], accept_multiple_files=True)
        if uploaded_video:
            for vid in uploaded_video:
                save_uploaded_file(vid, "video")
                st.success(f"Uploaded video: {vid.name}")
                
        # Audio Upload
        uploaded_audio = st.file_uploader("Upload Audio", type=["mp3", "wav", "ogg"], accept_multiple_files=True)
        if uploaded_audio:
            for aud in uploaded_audio:
                save_uploaded_file(aud, "audio")
                st.success(f"Uploaded audio: {aud.name}")
        
        # Document Upload
        uploaded_docs = st.file_uploader("Upload Documents", type=["pdf", "txt", "docx"], accept_multiple_files=True)
        if uploaded_docs:
            for doc in uploaded_docs:
                save_uploaded_file(doc, "document")
                st.success(f"Uploaded document: {doc.name}")
                
        st.markdown("---")
        
        # Navigation
        st.subheader("Navigation")
        page = st.radio("Go to:", ["Business Profile", "Strategy Generator", "Campaign Planning", "Media Gallery"])
        
        st.markdown("---")
        
        # About section
        st.markdown("### About ROH-Ads")
        st.write("""
        ROH-Ads is an AI-powered marketing strategy assistant that helps businesses create effective marketing strategies.
        """)
        
        return page

def business_profile_page():
    st.header("🏢 Business Profile")
    st.write("Let's gather some information about your business to create tailored marketing strategies.")
    
    # Prepare upload columns for media
    st.subheader("Upload Business Media for Auto-Analysis")
    st.write("Upload your business media to automatically extract information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        logo = st.file_uploader("Company Logo", type=["jpg", "jpeg", "png"])
        if logo:
            st.image(logo, width=200)
            save_uploaded_file(logo, "image")
    
    with col2:
        business_docs = st.file_uploader("Business Documents", type=["pdf", "txt", "docx"])
        if business_docs:
            save_uploaded_file(business_docs, "document")
            st.write(f"Uploaded: {business_docs.name}")
    
    with col3:
        promo_media = st.file_uploader("Promotional Media", type=["jpg", "jpeg", "png", "mp4", "mp3"])
        if promo_media:
            if promo_media.type.startswith('image'):
                st.image(promo_media, width=200)
                save_uploaded_file(promo_media, "image")
            elif promo_media.type.startswith('video'):
                st.video(promo_media)
                save_uploaded_file(promo_media, "video")
            elif promo_media.type.startswith('audio'):
                st.audio(promo_media)
                save_uploaded_file(promo_media, "audio")
    
    # Auto-analyze button
    media_files_for_analysis = []
    if logo:
        media_files_for_analysis.append(logo)
    if business_docs:
        media_files_for_analysis.append(business_docs)
    if promo_media:
        media_files_for_analysis.append(promo_media)
    
    if media_files_for_analysis and st.button("Auto-Analyze Media"):
        with st.spinner("Analyzing your business media..."):
            extracted_data = analyze_media_for_autofill(media_files_for_analysis)
            
            # Update session state with extracted data
            for key, value in extracted_data.items():
                if value and key in st.session_state.business_data:
                    st.session_state.business_data[key] = value
            
            st.success("Media analyzed and form auto-filled!")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.business_data["business_name"] = st.text_input(
            "Business Name", 
            value=st.session_state.business_data["business_name"]
        )
        
        st.session_state.business_data["industry"] = st.selectbox(
            "Industry",
            options=[
                "", "E-commerce", "SaaS", "Healthcare", "Education", "Finance", 
                "Retail", "Real Estate", "Food & Beverage", "Manufacturing", "Other"
            ],
            index=0 if st.session_state.business_data["industry"] == "" else 
                  list([
                      "", "E-commerce", "SaaS", "Healthcare", "Education", "Finance", 
                      "Retail", "Real Estate", "Food & Beverage", "Manufacturing", "Other"
                  ]).index(st.session_state.business_data["industry"])
        )
        
        st.session_state.business_data["budget_range"] = st.select_slider(
            "Marketing Budget Range",
            options=["Under $1,000", "$1,000-$5,000", "$5,000-$10,000", 
                     "$10,000-$50,000", "$50,000-$100,000", "$100,000+"],
            value=st.session_state.business_data["budget_range"] if st.session_state.business_data["budget_range"] else "Under $1,000"
        )
    
    with col2:
        # Replace target audience with 5-year traction plan
        st.session_state.business_data["five_year_traction"] = st.text_area(
            "5-Year Traction Plan",
            value=st.session_state.business_data["five_year_traction"],
            height=100,
            help="Describe your business growth expectations for the next 5 years"
        )
        
        # Keep marketing goals for consistency
        st.session_state.business_data["marketing_goals"] = st.text_area(
            "Marketing Goals",
            value=st.session_state.business_data["marketing_goals"],
            height=100
        )
    
    st.session_state.business_data["current_challenges"] = st.text_area(
        "Current Marketing Challenges",
        value=st.session_state.business_data["current_challenges"],
        height=100
    )
    
    if st.button("Save Profile"):
        if st.session_state.business_data["business_name"] and st.session_state.business_data["industry"]:
            st.success("Business profile saved successfully!")
            
            # Mention uploaded media in the prompt
            uploaded_media_text = ""
            if st.session_state.uploaded_files['images']:
                uploaded_media_text += f"They have uploaded {len(st.session_state.uploaded_files['images'])} images. "
            if st.session_state.uploaded_files['videos']:
                uploaded_media_text += f"They have uploaded {len(st.session_state.uploaded_files['videos'])} videos. "
            if st.session_state.uploaded_files['audio']:
                uploaded_media_text += f"They have uploaded {len(st.session_state.uploaded_files['audio'])} audio files. "
            if st.session_state.uploaded_files['documents']:
                uploaded_media_text += f"They have uploaded {len(st.session_state.uploaded_files['documents'])} documents. "
            
            analysis_prompt = f"""
            Analyze this business profile for marketing strategy opportunities:
            Business Name: {st.session_state.business_data['business_name']}
            Industry: {st.session_state.business_data['industry']}
            Marketing Goals: {st.session_state.business_data['marketing_goals']}
            Budget Range: {st.session_state.business_data['budget_range']}
            Challenges: {st.session_state.business_data['current_challenges']}
            5-Year Traction Plan: {st.session_state.business_data['five_year_traction']}
            
            {uploaded_media_text}
            
            Provide a concise summary and 3-5 initial marketing strategy recommendations based on this data.
            Focus particularly on their 5-year traction plan and how marketing can support that growth trajectory.
            """
            
            with st.spinner("Analyzing your business profile..."):
                analysis = generate_with_gemini(analysis_prompt, language=st.session_state.language)
                st.session_state.profile_analysis = analysis
                st.session_state.current_tts_text = analysis
            
            st.subheader("Initial Analysis")
            st.write(st.session_state.profile_analysis)
            
            # Auto-play TTS if enabled
            if st.session_state.tts_active:
                # Create a short summary for TTS to avoid long audio
                summary_prompt = f"""
                Create a 3-4 sentence summary of the following marketing analysis. Keep it very brief but informative:
                
                {analysis}
                """
                with st.spinner("Generating audio summary..."):
                    summary = generate_with_gemini(summary_prompt, language=st.session_state.language)
                    audio_player = text_to_speech(
                        summary, 
                        speed=st.session_state.voice_speed,
                        language=st.session_state.language
                    )
                    if audio_player:
                        st.markdown(audio_player, unsafe_allow_html=True)
        else:
            st.error("Please fill in at least the Business Name and Industry fields.")

def strategy_generator_page():
    st.header("🎯 Marketing Strategy Generator")
    
    if st.session_state.business_data["business_name"] == "":
        st.warning("Please complete your business profile first.")
        return
    
    st.write(f"Generating marketing strategies for **{st.session_state.business_data['business_name']}**")
    
    # Media upload specifically for strategy
    st.subheader("Upload Strategy-Related Media")
    strategy_media = st.file_uploader(
        "Upload relevant market research, competitor analyses, etc.", 
        type=["jpg", "jpeg", "png", "pdf", "docx", "txt", "mp4", "mp3"], 
        accept_multiple_files=True
    )
    
    strategy_media_files = []
    
    if strategy_media:
        for media in strategy_media:
            if media.type.startswith('image'):
                save_uploaded_file(media, "image")
                st.image(media, width=150, caption=media.name)
                strategy_media_files.append(media)
            elif media.type.startswith('video'):
                save_uploaded_file(media, "video")
                st.video(media)
                strategy_media_files.append(media)
            elif media.type.startswith('audio'):
                save_uploaded_file(media, "audio")
                st.audio(media)
                strategy_media_files.append(media)
            else:  # Document
                save_uploaded_file(media, "document")
                st.write(f"Uploaded document: {media.name}")
                strategy_media_files.append(media)
    
    st.markdown("---")
    
    st.subheader("Strategy Focus")
    focus_areas = st.multiselect(
        "Select marketing focus areas",
        options=[
            "Social Media Marketing", "Content Marketing", "Email Marketing", 
            "Search Engine Optimization (SEO)", "Pay-Per-Click Advertising (PPC)",
            "Influencer Marketing", "Video Marketing", "Affiliate Marketing"
        ]
    )
    
    timeframe = st.radio(
        "Strategy Timeframe",
        options=["Short-term (1-3 months)", "Medium-term (3-6 months)", "Long-term (6-12 months)"]
    )
    
    competitors = st.text_area("List main competitors (if any)")
    
    if st.button("Generate Marketing Strategy"):
        if focus_areas:
            # Mention uploaded media in the prompt
            uploaded_media_text = ""
            if st.session_state.uploaded_files['images']:
                uploaded_media_text += f"They have uploaded {len(st.session_state.uploaded_files['images'])} images. "
            if st.session_state.uploaded_files['videos']:
                uploaded_media_text += f"They have uploaded {len(st.session_state.uploaded_files['videos'])} videos. "
            if st.session_state.uploaded_files['audio']:
                uploaded_media_text += f"They have uploaded {len(st.session_state.uploaded_files['audio'])} audio files. "
            if st.session_state.uploaded_files['documents']:
                uploaded_media_text += f"They have uploaded {len(st.session_state.uploaded_files['documents'])} documents. "
                
            strategy_prompt = f"""
            Create a comprehensive marketing strategy for:
            Business: {st.session_state.business_data['business_name']}
            Industry: {st.session_state.business_data['industry']}
            Marketing Goals: {st.session_state.business_data['marketing_goals']}
            Budget: {st.session_state.business_data['budget_range']}
            Challenges: {st.session_state.business_data['current_challenges']}
            5-Year Traction Plan: {st.session_state.business_data['five_year_traction']}
            
            Focus on these marketing areas: {', '.join(focus_areas)}
            Timeframe: {timeframe}
            Competitors: {competitors}
            
            {uploaded_media_text}
            
            Please structure the strategy with these sections:
            1. Executive Summary
            2. Market Analysis
            3. 5-Year Growth Trajectory
            4. Marketing Channels & Tactics
            5. Content Strategy
            6. Budget Allocation
            7. Timeline & Implementation
            8. Success Metrics & KPIs
            
            Make the strategy specific, actionable, and tailored to their business profile.
            Especially focus on their 5-year traction plan and create a marketing roadmap that supports this growth trajectory.
            """
            
            with st.spinner("Generating your marketing strategy..."):
                # Use strategy_media_files if available for multimodal input
                strategy = generate_with_gemini(strategy_prompt, 
                                              media_files=strategy_media_files if strategy_media_files else None,
                                              language=st.session_state.language)
                st.session_state.marketing_strategy = strategy
                st.session_state.current_tts_text = strategy
            
            st.subheader("Your Marketing Strategy")
            st.write(st.session_state.marketing_strategy)
            
            # Download button for the strategy
            st.download_button(
                label="Download Strategy as Text",
                data=st.session_state.marketing_strategy,
                file_name=f"{st.session_state.business_data['business_name']}_marketing_strategy.txt",
                mime="text/plain"
            )
            
            # Auto-play TTS if enabled
            if st.session_state.tts_active:
                # Create a short summary for TTS to avoid long audio
                summary_prompt = f"""
                Create a 3-4 sentence summary of the key points from this marketing strategy. Focus only on the most important takeaways:
                
                {strategy[:1000]}...
                """
                with st.spinner("Generating audio summary..."):
                    summary = generate_with_gemini(summary_prompt, language=st.session_state.language)
                    audio_player = text_to_speech(
                        summary, 
                        gender=st.session_state.voice_gender, 
                        speed=st.session_state.voice_speed,
                        language=st.session_state.language
                    )
                    if audio_player:
                        st.markdown(audio_player, unsafe_allow_html=True)
        else:
            st.error("Please select at least one marketing focus area.")

def campaign_planning_page():
    st.header("📅 Campaign Planning")
    
    if st.session_state.marketing_strategy is None:
        st.warning("Please generate a marketing strategy first.")
        return
    
    st.subheader("Create a Marketing Campaign")
    
    # Media upload specifically for campaign
    st.subheader("Upload Campaign-Related Media")
    campaign_media = st.file_uploader(
        "Upload creative assets, brand guidelines, etc.", 
        type=["jpg", "jpeg", "png", "pdf", "docx", "txt", "mp4", "mp3"], 
        accept_multiple_files=True
    )
    
    campaign_media_files = []
    
    if campaign_media:
        for media in campaign_media:
            if media.type.startswith('image'):
                save_uploaded_file(media, "image")
                st.image(media, width=150, caption=media.name)
                campaign_media_files.append(media)
            elif media.type.startswith('video'):
                save_uploaded_file(media, "video")
                st.video(media)
                campaign_media_files.append(media)
            elif media.type.startswith('audio'):
                save_uploaded_file(media, "audio")
                st.audio(media)
                campaign_media_files.append(media)
            else:  # Document
                save_uploaded_file(media, "document")
                st.write(f"Uploaded document: {media.name}")
                campaign_media_files.append(media)
    
    st.markdown("---")
    
    campaign_name = st.text_input("Campaign Name")
    campaign_goal = st.selectbox(
        "Primary Campaign Objective",
        options=[
            "Brand Awareness", "Lead Generation", "Sales/Conversions", 
            "Customer Retention", "Product Launch", "Event Promotion"
        ]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date")
        campaign_budget = st.number_input("Campaign Budget ($)", min_value=0, step=100)
    
    with col2:
        end_date = st.date_input("End Date")
        primary_channel = st.selectbox(
            "Primary Marketing Channel",
            options=[
                "Social Media", "Email", "Content Marketing", "PPC", 
                "SEO", "Events", "Influencer Marketing"
            ]
        )
    
    campaign_description = st.text_area("Campaign Description")
    
    if st.button("Generate Campaign Plan"):
        if campaign_name and campaign_goal and campaign_description:
            # Mention uploaded media in the prompt
            uploaded_media_text = ""
            if st.session_state.uploaded_files['images']:
                uploaded_media_text += f"They have uploaded {len(st.session_state.uploaded_files['images'])} images. "
            if st.session_state.uploaded_files['videos']:
                uploaded_media_text += f"They have uploaded {len(st.session_state.uploaded_files['videos'])} videos. "
            if st.session_state.uploaded_files['audio']:
                uploaded_media_text += f"They have uploaded {len(st.session_state.uploaded_files['audio'])} audio files. "
            if st.session_state.uploaded_files['documents']:
                uploaded_media_text += f"They have uploaded {len(st.session_state.uploaded_files['documents'])} documents. "
                
            campaign_prompt = f"""
            Create a detailed marketing campaign plan for:
            Business: {st.session_state.business_data['business_name']}
            Campaign Name: {campaign_name}
            Campaign Goal: {campaign_goal}
            Timeframe: {start_date} to {end_date}
            Budget: ${campaign_budget}
            Primary Channel: {primary_channel}
            Description: {campaign_description}
            5-Year Traction Plan: {st.session_state.business_data['five_year_traction']}
            
            {uploaded_media_text}
            
            This campaign should align with the overall marketing strategy for the business.
            
            Please include:
            1. Campaign Brief (summary, goals, KPIs)
            2. Target Audience Segments
            3. Messaging & Creative Direction
            4. Channel Strategy & Content Calendar
            5. Budget Breakdown
            6. Timeline with Key Milestones
            7. Measurement Plan
            8. Contribution to 5-Year Growth Goals
            
            Make the campaign plan specific, actionable, and provide examples of content or messaging where applicable.
            """
            
            with st.spinner("Generating your campaign plan..."):
                campaign_plan = generate_with_gemini(campaign_prompt, 
                                                   media_files=campaign_media_files if campaign_media_files else None,
                                                   language=st.session_state.language)
                st.session_state.current_tts_text = campaign_plan
            
            st.subheader("Your Campaign Plan")
            st.write(campaign_plan)
            
            # Download button for the campaign plan
            st.download_button(
                label="Download Campaign Plan",
                data=campaign_plan,
                file_name=f"{campaign_name}_campaign_plan.txt",
                mime="text/plain"
            )
            
            # Auto-play TTS if enabled
            if st.session_state.tts_active:
                # Create a short summary for TTS to avoid long audio
                summary_prompt = f"""
                Create a brief 3-4 sentence summary of this marketing campaign plan for {campaign_name}. Focus on the key actions and expected outcomes:
                
                {campaign_plan[:1000]}...
                """
                with st.spinner("Generating audio summary..."):
                    summary = generate_with_gemini(summary_prompt, language=st.session_state.language)
                    audio_player = text_to_speech(
                        summary, 
                        gender=st.session_state.voice_gender, 
                        speed=st.session_state.voice_speed,
                        language=st.session_state.language
                    )
                    if audio_player:
                        st.markdown(audio_player, unsafe_allow_html=True)
        else:
            st.error("Please fill in all required fields.")

def media_gallery_page():
    st.header("🖼️ Media Gallery")
    st.write("View and manage your uploaded media files")
    
    # Filter tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Images", "Videos", "Audio", "Documents"])
    
    with tab1:
        st.subheader("Uploaded Images")
        if st.session_state.uploaded_files['images']:
            # Create a grid layout for images
            cols = st.columns(3)
            for i, img_info in enumerate(st.session_state.uploaded_files['images']):
                col_idx = i % 3
                with cols[col_idx]:
                    st.write(f"**{img_info['FileName']}**")
                    st.write(f"Type: {img_info['FileType']}")
                    st.write(f"Size: {img_info['FileSize']/1024:.1f} KB")
        else:
            st.info("No images uploaded yet")
    
    with tab2:
        st.subheader("Uploaded Videos")
        if st.session_state.uploaded_files['videos']:
            for video_info in st.session_state.uploaded_files['videos']:
                st.write(f"**{video_info['FileName']}**")
                st.write(f"Type: {video_info['FileType']}")
                st.write(f"Size: {video_info['FileSize']/1024/1024:.1f} MB")
                st.markdown("---")
        else:
            st.info("No videos uploaded yet")
    
    with tab3:
        st.subheader("Uploaded Audio")
        if st.session_state.uploaded_files['audio']:
            for audio_info in st.session_state.uploaded_files['audio']:
                st.write(f"**{audio_info['FileName']}**")
                st.write(f"Type: {audio_info['FileType']}")
                st.write(f"Size: {audio_info['FileSize']/1024/1024:.1f} MB")
                st.markdown("---")
        else:
            st.info("No audio files uploaded yet")
            
    with tab4:
        st.subheader("Uploaded Documents")
        if st.session_state.uploaded_files['documents']:
            for doc_info in st.session_state.uploaded_files['documents']:
                st.write(f"**{doc_info['FileName']}**")
                st.write(f"Type: {doc_info['FileType']}")
                st.write(f"Size: {doc_info['FileSize']/1024/1024:.1f} MB")
                st.markdown("---")
        else:
            st.info("No documents uploaded yet")
    
    # Media analysis section
    st.subheader("Media Analysis")
    media_to_analyze = st.multiselect(
        "Select media files to analyze",
        options=[file['FileName'] for category in st.session_state.uploaded_files.values() for file in category]
    )
    
    if media_to_analyze and st.button("Analyze Selected Media"):
        st.write("Analyzing media...")
        # This is a placeholder - in a real app we'd need to retrieve the actual file objects
        # but for demo purposes we'll just show the concept
        analysis_result = f"Analysis of {len(media_to_analyze)} items would be performed here using Gemini"
        st.write(analysis_result)

# Main application with error handling
def main():
    # Initialize session state if not already done
    if 'language' not in st.session_state:
        st.session_state.language = "en"
    if 'tts_active' not in st.session_state:
        st.session_state.tts_active = False
    if 'voice_speed' not in st.session_state:
        st.session_state.voice_speed = "Normal"
    
    # Try to load ML models, but continue even if it fails
    try:
        if 'ml_model' not in st.session_state:
            st.session_state.ml_model = load_ml_models()
    except Exception as e:
        st.warning(f"ML model loading failed: {str(e)}. Some features may be limited.")
        st.session_state.ml_model = None
    
    # Continue with the app, even if ML models aren't available
    try:
        page = sidebar()
        
        if page == "Business Profile":
            business_profile_page()
        elif page == "Strategy Generator":
            strategy_generator_page()
        elif page == "Campaign Planning":
            campaign_planning_page()
        elif page == "Media Gallery":
            media_gallery_page()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please try refreshing the page or check your internet connection.")

if __name__ == "__main__":
    main()
