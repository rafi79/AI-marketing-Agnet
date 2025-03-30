import streamlit as st
import requests
import json
import time
import base64
import os
from gtts import gTTS
from io import BytesIO
import tempfile

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
        "current_challenges": ""
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
        'audio': []
    }
if 'voice_gender' not in st.session_state:
    st.session_state.voice_gender = "Female"
if 'voice_speed' not in st.session_state:
    st.session_state.voice_speed = "Normal"

# TTS function with improved male voice support
def text_to_speech(text, gender="Female", speed="Normal"):
    """Convert text to speech and create an audio player"""
    if not text:
        return None
    
    try:
        # Configure TTS based on speed
        slow_option = False
        if speed == "Slow":
            slow_option = True
        
        # Use different language codes for male/female approximation
        # This is a workaround since gTTS doesn't directly support gender selection
        lang_code = "en-us"  # Default female voice
        if gender == "Male":
            lang_code = "en-gb"  # British English tends to have a deeper voice
        
        # Create the TTS object
        tts = gTTS(text=text, lang=lang_code, slow=slow_option)
        
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

def generate_with_gemini(prompt, image_data=None):
    """Generate content using Gemini model via REST API"""
    try:
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }
        
        # Prepare the data structure for the API call
        data = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 4096
            }
        }
        
        # Add image if provided (for future multimodal implementation)
        if image_data:
            # This would need to be implemented with proper multimodal API support
            pass
        
        response = requests.post(
            GEMINI_API_URL,
            headers=headers,
            data=json.dumps(data)
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            st.error(f"API Error: {response.status_code}")
            return f"Error: {response.text}"
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return f"An error occurred: {str(e)}"

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
    
    return file_details

def display_file_preview(file, file_type):
    """Display a preview of the uploaded file"""
    if file_type == "image":
        st.image(file, caption=file.name, use_column_width=True)
    elif file_type == "video":
        st.video(file)
    elif file_type == "audio":
        st.audio(file)

# UI Components
def sidebar():
    with st.sidebar:
        st.title("🚀 ROH-Ads")
        st.subheader("AI Marketing Strategy Assistant")
        
        st.markdown("---")
        
        # TTS Controls
        st.subheader("🔊 Text-to-Speech")
        st.session_state.tts_active = st.toggle("Enable AI Voice", value=st.session_state.tts_active)
        
        st.session_state.voice_speed = st.select_slider(
            "Voice Speed",
            options=["Slow", "Normal", "Fast"],
            value=st.session_state.voice_speed,
            disabled=not st.session_state.tts_active
        )
        
        st.session_state.voice_gender = st.radio(
            "Voice Type",
            options=["Female", "Male"],
            index=0 if st.session_state.voice_gender == "Female" else 1,
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
                    summary = generate_with_gemini(summary_prompt)
                    audio_player = text_to_speech(
                        summary, 
                        gender=st.session_state.voice_gender, 
                        speed=st.session_state.voice_speed
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
    
    # Media upload section specifically for business profile
    st.subheader("Upload Business Media")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        logo = st.file_uploader("Company Logo", type=["jpg", "jpeg", "png"])
        if logo:
            st.image(logo, width=200)
            save_uploaded_file(logo, "image")
    
    with col2:
        product_images = st.file_uploader("Product Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if product_images:
            for img in product_images:
                save_uploaded_file(img, "image")
                st.image(img, width=150, caption=img.name)
    
    with col3:
        promo_video = st.file_uploader("Promotional Video", type=["mp4", "mov"])
        if promo_video:
            st.video(promo_video)
            save_uploaded_file(promo_video, "video")
    
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
        st.session_state.business_data["target_audience"] = st.text_area(
            "Target Audience Description",
            value=st.session_state.business_data["target_audience"],
            height=100
        )
        
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
            
            analysis_prompt = f"""
            Analyze this business profile for marketing strategy opportunities:
            Business Name: {st.session_state.business_data['business_name']}
            Industry: {st.session_state.business_data['industry']}
            Target Audience: {st.session_state.business_data['target_audience']}
            Marketing Goals: {st.session_state.business_data['marketing_goals']}
            Budget Range: {st.session_state.business_data['budget_range']}
            Challenges: {st.session_state.business_data['current_challenges']}
            
            {uploaded_media_text}
            
            Provide a concise summary and 3-5 initial marketing strategy recommendations based on this data.
            """
            
            with st.spinner("Analyzing your business profile..."):
                analysis = generate_with_gemini(analysis_prompt)
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
                    summary = generate_with_gemini(summary_prompt)
                    audio_player = text_to_speech(
                        summary, 
                        gender=st.session_state.voice_gender, 
                        speed=st.session_state.voice_speed
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
        type=["jpg", "jpeg", "png", "pdf", "mp4", "mp3"], 
        accept_multiple_files=True
    )
    
    if strategy_media:
        for media in strategy_media:
            if media.type.startswith('image'):
                save_uploaded_file(media, "image")
                st.image(media, width=150, caption=media.name)
            elif media.type.startswith('video'):
                save_uploaded_file(media, "video")
                st.video(media)
            elif media.type.startswith('audio'):
                save_uploaded_file(media, "audio")
                st.audio(media)
    
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
                
            strategy_prompt = f"""
            Create a comprehensive marketing strategy for:
            Business: {st.session_state.business_data['business_name']}
            Industry: {st.session_state.business_data['industry']}
            Target Audience: {st.session_state.business_data['target_audience']}
            Goals: {st.session_state.business_data['marketing_goals']}
            Budget: {st.session_state.business_data['budget_range']}
            Challenges: {st.session_state.business_data['current_challenges']}
            
            Focus on these marketing areas: {', '.join(focus_areas)}
            Timeframe: {timeframe}
            Competitors: {competitors}
            
            {uploaded_media_text}
            
            Please structure the strategy with these sections:
            1. Executive Summary
            2. Market Analysis
            3. Target Audience Insights
            4. Marketing Channels & Tactics
            5. Content Strategy
            6. Budget Allocation
            7. Timeline & Implementation
            8. Success Metrics & KPIs
            
            Make the strategy specific, actionable, and tailored to their business profile.
            """
            
            with st.spinner("Generating your marketing strategy..."):
                strategy = generate_with_gemini(strategy_prompt)
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
                    summary = generate_with_gemini(summary_prompt)
                    audio_player = text_to_speech(
                        summary, 
                        gender=st.session_state.voice_gender, 
                        speed=st.session_state.voice_speed
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
        type=["jpg", "jpeg", "png", "pdf", "mp4", "mp3"], 
        accept_multiple_files=True
    )
    
    if campaign_media:
        for media in campaign_media:
            if media.type.startswith('image'):
                save_uploaded_file(media, "image")
                st.image(media, width=150, caption=media.name)
            elif media.type.startswith('video'):
                save_uploaded_file(media, "video")
                st.video(media)
            elif media.type.startswith('audio'):
                save_uploaded_file(media, "audio")
                st.audio(media)
    
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
                
            campaign_prompt = f"""
            Create a detailed marketing campaign plan for:
            Business: {st.session_state.business_data['business_name']}
            Campaign Name: {campaign_name}
            Campaign Goal: {campaign_goal}
            Timeframe: {start_date} to {end_date}
            Budget: ${campaign_budget}
            Primary Channel: {primary_channel}
            Description: {campaign_description}
            
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
            
            Make the campaign plan specific, actionable, and provide examples of content or messaging where applicable.
            """
            
            with st.spinner("Generating your campaign plan..."):
                campaign_plan = generate_with_gemini(campaign_prompt)
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
                    summary = generate_with_gemini(summary_prompt)
                    audio_player = text_to_speech(
                        summary, 
                        gender=st.session_state.voice_gender, 
                        speed=st.session_state.voice_speed
                    )
                    if audio_player:
                        st.markdown(audio_player, unsafe_allow_html=True)
        else:
            st.error("Please fill in all required fields.")

def media_gallery_page():
    st.header("🖼️ Media Gallery")
    st.write("View and manage your uploaded media files")
    
    # Filter tabs
    tab1, tab2, tab3 = st.tabs(["Images", "Videos", "Audio"])
    
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
    
    # Upload new media
    st.subheader("Upload New Media")
    
    upload_type = st.radio("Select media type", ["Image", "Video", "Audio"])
    
    if upload_type == "Image":
        new_images = st.file_uploader("Upload new images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if new_images:
            for img in new_images:
                save_uploaded_file(img, "image")
                st.success(f"Uploaded image: {img.name}")
                st.image(img, width=200)
    
    elif upload_type == "Video":
        new_videos = st.file_uploader("Upload new videos", type=["mp4", "mov", "avi"], accept_multiple_files=True)
        if new_videos:
            for vid in new_videos:
                save_uploaded_file(vid, "video")
                st.success(f"Uploaded video: {vid.name}")
                st.video(vid)
    
    elif upload_type == "Audio":
        new_audio = st.file_uploader("Upload new audio", type=["mp3", "wav", "ogg"], accept_multiple_files=True)
        if new_audio:
            for aud in new_audio:
                save_uploaded_file(aud, "audio")
                st.success(f"Uploaded audio: {aud.name}")
                st.audio(aud)

# Main application
def main():
    page = sidebar()
    
    if page == "Business Profile":
        business_profile_page()
    elif page == "Strategy Generator":
        strategy_generator_page()
    elif page == "Campaign Planning":
        campaign_planning_page()
    elif page == "Media Gallery":
        media_gallery_page()

if __name__ == "__main__":
    main()
