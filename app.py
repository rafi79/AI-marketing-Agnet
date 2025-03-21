import streamlit as st
import os
import base64
import mimetypes
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO
from datetime import datetime
import time

# AI Model Imports
from google import genai
from google.genai import types
import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Set page config
st.set_page_config(
    page_title="AI Marketing Agent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #333;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 10px;
        padding: 1.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .tool-icon {
        font-size: 2rem;
        margin-right: 0.5rem;
    }
    .highlight {
        background-color: #f0f7ff;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Load API keys and initialize models
def load_api_keys():
    """Load API keys from Streamlit secrets or environment variables"""
    gemini_api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
    hf_token = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", "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"))
    return gemini_api_key, hf_token

@st.cache_resource
def load_gemma_model():
    """Load Gemma 3 1B-it model from HuggingFace"""
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it", token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", token=HF_TOKEN)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return pipe
    except Exception as e:
        st.error(f"Error loading Gemma model: {e}")
        return None

@st.cache_data
def save_uploaded_file(uploaded_file):
    """Save uploaded file to temp directory and return the path"""
    if uploaded_file is not None:
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None

# Initialize Gemini client
def init_gemini_client(api_key):
    if api_key:
        genai.configure(api_key=api_key)
        return genai.Client(api_key=api_key)
    return None

# Function to generate text with Gemma model
def generate_with_gemma(prompt, max_tokens=1024):
    messages = [{"role": "user", "content": prompt}]
    try:
        response = gemma_pipe(messages, max_new_tokens=max_tokens, temperature=0.7)
        return response[0]["generated_text"].split("assistant: ")[-1]
    except Exception as e:
        st.error(f"Error generating with Gemma: {e}")
        return "Error generating response. Please check your inputs and try again."

# Function to generate text with Gemini model
def generate_with_gemini(prompt, model="gemini-2.0-flash", temperature=0.7, max_tokens=4096):
    try:
        # Create the prompt structure
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]
        
        # Configure generation parameters
        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=0.95,
            top_k=40,
            max_output_tokens=max_tokens,
            response_mime_type="text/plain",
        )
        
        # Generate content
        response = gemini_client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        
        return response.text
    except Exception as e:
        st.error(f"Error generating with Gemini: {e}")
        return "Error generating response. Please check your inputs and try again."

# Function to generate image with Gemini
def generate_image_with_gemini(prompt):
    try:
        # Create the prompt structure
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]
        
        # Configure generation parameters for image
        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            response_modalities=["image", "text"],
        )
        
        # Generate content
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=contents,
            config=generate_content_config,
        )
        
        # Extract and save the image
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            part = response.candidates[0].content.parts[0]
            if hasattr(part, 'inline_data'):
                return part.inline_data.data, part.inline_data.mime_type
        
        return None, None
    except Exception as e:
        st.error(f"Error generating image with Gemini: {e}")
        return None, None

# Save generated image
def save_image(image_data, mime_type):
    file_name = f"generated_image_{int(time.time())}"
    file_extension = mimetypes.guess_extension(mime_type)
    full_path = f"{file_name}{file_extension}"
    
    with open(full_path, "wb") as f:
        f.write(image_data)
    
    return full_path

# Function to display generated image
def display_generated_image(image_data, mime_type):
    if image_data and mime_type:
        image = Image.open(BytesIO(image_data))
        st.image(image, caption="Generated Image", use_column_width=True)
        
        # Create download button
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        href = f'<a href="data:image/png;base64,{img_str}" download="generated_image.png">Download Image</a>'
        st.markdown(href, unsafe_allow_html=True)

# Market & Product Analysis function
def market_analysis(product_description, target_audience, industry):
    prompt = f"""
    # Market & Product Analysis Task
    
    ## Product Information:
    {product_description}
    
    ## Target Audience:
    {target_audience}
    
    ## Industry/Niche:
    {industry}
    
    ## Task:
    As a marketing expert, please provide a comprehensive market analysis including:
    
    1. Target audience demographics and psychographics
    2. Top 3 trending themes relevant to this product
    3. Recommended social media platforms with reasoning
    4. Optimal product positioning strategy
    5. Key competitors and how to differentiate
    6. Market opportunities and gaps to exploit
    
    Please structure your response with clear headers and actionable insights.
    """
    return generate_with_gemma(prompt)

# Brand Building function
def brand_building(product_description, target_audience, brand_values):
    prompt = f"""
    # Brand Building Task
    
    ## Product Information:
    {product_description}
    
    ## Target Audience:
    {target_audience}
    
    ## Brand Values/Personality:
    {brand_values}
    
    ## Task:
    Create a compelling brand identity including:
    
    1. 5 unique brand name suggestions with rationale
    2. 3 memorable taglines that capture the brand essence
    3. Brand voice and tone guidelines (formal, casual, humorous, etc.)
    4. Brand story/narrative elements
    5. Key brand messaging points for marketing materials
    6. Visual brand direction recommendations (colors, style, imagery)
    
    Please structure your response with clear headers and provide reasoning for each recommendation.
    """
    return generate_with_gemma(prompt)

# Content Generation function
def content_generation(product_description, target_audience, brand_name, platform):
    prompt = f"""
    # Content Generation Task
    
    ## Product Information:
    {product_description}
    
    ## Target Audience:
    {target_audience}
    
    ## Brand Name/Identity:
    {brand_name}
    
    ## Platform:
    {platform}
    
    ## Task:
    Generate engaging content for the specified platform including:
    
    1. 5 static post ideas with captions and hashtags
    2. 3 video content ideas with script outlines
    3. 2 interactive content ideas (polls, questions, contests)
    4. Content calendar suggestion for 2 weeks (post types and timing)
    5. Engagement strategies and call-to-actions
    
    Please structure your response with clear headers and make all content authentic, engaging and tailored to the target audience.
    """
    return generate_with_gemini(prompt)

# Campaign Planning function
def campaign_planning(product_description, target_audience, brand_name, objectives):
    prompt = f"""
    # Campaign Planning Task
    
    ## Product Information:
    {product_description}
    
    ## Target Audience:
    {target_audience}
    
    ## Brand Name/Identity:
    {brand_name}
    
    ## Campaign Objectives:
    {objectives}
    
    ## Task:
    Develop a comprehensive marketing campaign plan including:
    
    1. Campaign theme and core message
    2. Channel strategy with budget allocation recommendations
    3. Content types and creative direction
    4. Timeline and key milestones
    5. A/B testing ideas for optimization
    6. KPIs and success metrics
    7. Potential challenges and contingency plans
    
    Please structure your response with clear headers and provide actionable, specific recommendations.
    """
    return generate_with_gemini(prompt)

# Performance Analysis function
def performance_analysis(campaign_data, campaign_objectives):
    prompt = f"""
    # Performance Analysis Task
    
    ## Campaign Data:
    {campaign_data}
    
    ## Campaign Objectives:
    {campaign_objectives}
    
    ## Task:
    Analyze the campaign performance and provide:
    
    1. Summary of key metrics and performance against objectives
    2. Identification of successful and underperforming elements
    3. Insights on audience response and engagement patterns
    4. Recommendations for improvement and optimization
    5. Actionable next steps and strategy adjustments
    6. Future testing ideas based on learnings
    
    Please structure your response with clear headers and provide data-driven insights with specific recommendations.
    """
    return generate_with_gemini(prompt)

# Image Generation function
def generate_brand_imagery(product_description, brand_style, image_type):
    prompt = f"""
    Create a professional marketing image for:
    
    Product: {product_description}
    Brand Style: {brand_style}
    Image Type: {image_type}
    
    The image should be high quality, visually appealing, and aligned with the brand style. 
    Create an image that would work well for digital marketing campaigns.
    """
    return generate_image_with_gemini(prompt)

# Video Script Generation
def generate_video_script(product_description, brand_name, video_type, duration):
    prompt = f"""
    # Video Script Generation Task
    
    ## Product Information:
    {product_description}
    
    ## Brand Name:
    {brand_name}
    
    ## Video Type:
    {video_type}
    
    ## Duration:
    {duration}
    
    ## Task:
    Create a detailed video script including:
    
    1. Opening hook/attention grabber
    2. Scene-by-scene breakdown with visual descriptions
    3. Complete dialogue/narration text
    4. Key messaging points to emphasize
    5. Call-to-action for closing
    6. Music/sound suggestions
    
    Please format the script professionally with clear scene delineation, timing, and both visual and audio elements.
    """
    return generate_with_gemini(prompt)

# Social Media Response Generator
def generate_social_responses(comments, brand_voice, product_info):
    prompt = f"""
    # Social Media Response Generation Task
    
    ## Comments to Respond to:
    {comments}
    
    ## Brand Voice:
    {brand_voice}
    
    ## Product Information:
    {product_info}
    
    ## Task:
    Generate appropriate responses to each comment that:
    
    1. Maintain consistent brand voice and personality
    2. Address the specific question or feedback
    3. Encourage further engagement when appropriate
    4. Handle any negative comments professionally
    5. Incorporate product information naturally when relevant
    
    Please provide each response separately with the comment it's addressing.
    """
    return generate_with_gemini(prompt)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Market Analysis"
if 'generated_data' not in st.session_state:
    st.session_state.generated_data = {
        "market_analysis": None,
        "brand_identity": None,
        "content_ideas": None,
        "campaign_plan": None,
        "performance_analysis": None,
        "generated_images": []
    }

# Load API keys
GEMINI_API_KEY, HF_TOKEN = load_api_keys()

# Initialize clients
gemini_client = init_gemini_client(GEMINI_API_KEY)
gemma_pipe = load_gemma_model()

# Main app layout
st.markdown('<h1 class="main-header">🚀 AI Marketing Agent</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/marketing.png", width=80)
    st.markdown("## Marketing Toolbox")
    
    tool_selection = st.radio(
        "Select Marketing Tool:",
        ["Market Analysis", "Brand Builder", "Content Creator", 
         "Campaign Planner", "Performance Analyzer", "Image Generator"]
    )
    
    # Save current tab
    st.session_state.current_tab = tool_selection
    
    st.markdown("---")
    st.markdown("### Product Information")
    
    # Common inputs for all tools
    product_description = st.text_area("Product Description", placeholder="Describe your product in detail...")
    target_audience = st.text_area("Target Audience", placeholder="Describe your target audience...")
    
    st.markdown("---")
    st.markdown("### Project History")
    
    # Display history
    if st.session_state.history:
        for i, item in enumerate(st.session_state.history):
            st.markdown(f"**{i+1}. {item['tool']}** - {item['timestamp']}")
    else:
        st.markdown("No history yet.")

# Main content area based on selected tool
if st.session_state.current_tab == "Market Analysis":
    st.markdown('<h2 class="sub-header">🧐 Market & Product Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Product & Market Analysis")
        
        industry = st.text_input("Industry/Niche", placeholder="E.g., Fashion, Technology, Health...")
        
        if st.button("Generate Market Analysis"):
            if product_description and target_audience and industry:
                with st.spinner("Analyzing market and product..."):
                    analysis_result = market_analysis(product_description, target_audience, industry)
                    st.session_state.generated_data["market_analysis"] = analysis_result
                    
                    # Add to history
                    st.session_state.history.append({
                        "tool": "Market Analysis",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "inputs": {
                            "product": product_description,
                            "audience": target_audience,
                            "industry": industry
                        }
                    })
                    
                st.success("Market analysis completed!")
            else:
                st.warning("Please fill all required fields.")
        
        if st.session_state.generated_data["market_analysis"]:
            st.markdown("### Analysis Results")
            st.markdown(st.session_state.generated_data["market_analysis"])
            
            # Export options
            if st.button("Export Analysis"):
                # Create downloadable report
                report = f"""
                # Market & Product Analysis Report
                
                ## Product Information
                {product_description}
                
                ## Target Audience
                {target_audience}
                
                ## Industry/Niche
                {industry}
                
                ## Analysis Results
                {st.session_state.generated_data["market_analysis"]}
                
                ## Generated on
                {datetime.now().strftime("%Y-%m-%d %H:%M")}
                """
                
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name="market_analysis_report.md",
                    mime="text/markdown"
                )
                
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_tab == "Performance Analyzer":
    st.markdown('<h2 class="sub-header">📈 Performance Analyzer</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Campaign Performance Analysis")
        
        # Options for data input
        data_input_method = st.radio(
            "Input Campaign Data",
            ["Upload Data File", "Manual Input", "Sample Data"]
        )
        
        campaign_data = ""
        
        if data_input_method == "Upload Data File":
            uploaded_file = st.file_uploader("Upload campaign data (CSV, Excel)", type=["csv", "xlsx"])
            if uploaded_file is not None:
                try:
                    # Save the file
                    file_path = save_uploaded_file(uploaded_file)
                    
                    # Preview the data
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        campaign_data = df.to_string()
                    else:  # Excel
                        df = pd.read_excel(uploaded_file)
                        campaign_data = df.to_string()
                        
                    st.dataframe(df)
                except Exception as e:
                    st.error(f"Error processing file: {e}")
        
        elif data_input_method == "Manual Input":
            campaign_data = st.text_area(
                "Enter Campaign Performance Data",
                placeholder="Enter your campaign data in a structured format..."
            )
        
        else:  # Sample Data
            sample_data = pd.DataFrame({
                'Date': pd.date_range(start='2023-01-01', periods=10),
                'Channel': ['Instagram', 'Facebook', 'Email', 'Instagram', 'TikTok', 
                           'Facebook', 'Instagram', 'Email', 'TikTok', 'Facebook'],
                'Impressions': [12500, 8700, 5400, 13200, 18900, 9200, 14500, 5800, 21300, 9800],
                'Clicks': [420, 310, 180, 460, 630, 340, 490, 210, 710, 360],
                'Conversions': [23, 18, 12, 25, 34, 19, 27, 14, 38, 21],
                'Cost': [150, 120, 80, 160, 200, 130, 170, 90, 220, 140]
            })
            
            campaign_data = sample_data.to_string()
            st.dataframe(sample_data)
            
            # Sample KPI visualization
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            
            with kpi_col1:
                st.metric("Total Impressions", f"{sample_data['Impressions'].sum():,}")
            with kpi_col2:
                st.metric("Total Clicks", f"{sample_data['Clicks'].sum():,}")
            with kpi_col3:
                st.metric("Total Conversions", f"{sample_data['Conversions'].sum():,}")
            with kpi_col4:
                ctr = (sample_data['Clicks'].sum() / sample_data['Impressions'].sum()) * 100
                st.metric("Average CTR", f"{ctr:.2f}%")
            
            # Sample charts
            tab1, tab2, tab3 = st.tabs(["Performance by Channel", "Trend Over Time", "Conversion Rates"])
            
            with tab1:
                channel_data = sample_data.groupby('Channel').sum().reset_index()
                fig = px.bar(channel_data, x='Channel', y=['Impressions', 'Clicks', 'Conversions'], 
                            barmode='group', title='Performance Metrics by Channel')
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                daily_data = sample_data.groupby('Date').sum().reset_index()
                fig = px.line(daily_data, x='Date', y=['Impressions', 'Clicks', 'Conversions'], 
                             title='Performance Trends Over Time')
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                channel_data['CTR'] = channel_data['Clicks'] / channel_data['Impressions'] * 100
                channel_data['CVR'] = channel_data['Conversions'] / channel_data['Clicks'] * 100
                channel_data['CPA'] = channel_data['Cost'] / channel_data['Conversions']
                
                fig = px.bar(channel_data, x='Channel', y=['CTR', 'CVR'], 
                            barmode='group', title='Conversion Rates by Channel')
                st.plotly_chart(fig, use_container_width=True)
        
        campaign_objectives = st.text_area(
            "Campaign Objectives",
            placeholder="What were your campaign objectives? E.g., increase brand awareness, drive sales, launch new product..."
        )
        
        if st.button("Analyze Performance"):
            if campaign_data and campaign_objectives:
                with st.spinner("Analyzing campaign performance..."):
                    analysis_result = performance_analysis(campaign_data, campaign_objectives)
                    st.session_state.generated_data["performance_analysis"] = analysis_result
                    
                    # Add to history
                    st.session_state.history.append({
                        "tool": "Performance Analyzer",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "inputs": {
                            "data": "Campaign data analyzed",
                            "objectives": campaign_objectives
                        }
                    })
                    
                st.success("Performance analysis completed!")
            else:
                st.warning("Please provide campaign data and objectives.")
        
        if st.session_state.generated_data["performance_analysis"]:
            st.markdown("### Analysis Results")
            st.markdown(st.session_state.generated_data["performance_analysis"])
            
            # Export options
            if st.button("Export Analysis Report"):
                report = f"""
                # Campaign Performance Analysis Report
                
                ## Campaign Data
                Data analysis based on provided campaign metrics
                
                ## Campaign Objectives
                {campaign_objectives}
                
                ## Performance Analysis
                {st.session_state.generated_data["performance_analysis"]}
                
                ## Generated on
                {datetime.now().strftime("%Y-%m-%d %H:%M")}
                """
                
                st.download_button(
                    label="Download Analysis Report",
                    data=report,
                    file_name="performance_analysis_report.md",
                    mime="text/markdown"
                )
                
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Optimization Recommendations")
        
        st.markdown("""
        After analyzing campaign performance, this tool will provide:
        
        1. **Key Insights**
           - Performance against objectives
           - Top-performing channels & content
           - Audience engagement patterns
           
        2. **Optimization Ideas**
           - Budget reallocation suggestions
           - Content strategy adjustments
           - Targeting refinements
           
        3. **Future Recommendations**
           - Testing ideas for improvement
           - New channel opportunities
           - Long-term strategy adjustments
        """)
        
        st.markdown("### Common Metrics to Track")
        metrics = {
            "Awareness": ["Impressions", "Reach", "Brand Lift"],
            "Engagement": ["Click-through Rate", "Engagement Rate", "Time on Page"],
            "Conversion": ["Conversion Rate", "Cost per Acquisition", "ROI/ROAS"]
        }
        
        for category, metric_list in metrics.items():
            with st.expander(f"{category} Metrics"):
                for metric in metric_list:
                    st.markdown(f"- **{metric}**")
        
        st.markdown("### Performance Benchmarks")
        
        benchmark_data = pd.DataFrame({
            'Channel': ['Social Media', 'Email', 'Content Marketing', 'Paid Search', 'Display Ads'],
            'CTR (%)': [0.8, 2.5, 1.2, 1.8, 0.5],
            'CVR (%)': [1.0, 3.2, 1.8, 2.5, 0.8],
            'ROI (x)': [3.5, 4.2, 3.8, 4.0, 2.8]
        })
        
        st.dataframe(benchmark_data)
                
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_tab == "Image Generator":
    st.markdown('<h2 class="sub-header">🖼️ Marketing Image Generator</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Generate Marketing Visuals")
        
        image_options = {
            "Product Showcase": "Create a professional image showcasing the product in use or on display",
            "Social Media Post": "Create an eye-catching social media post image with the product",
            "Advertisement": "Create a compelling advertisement image for digital or print",
            "Brand Story": "Create an image that tells a story about the brand or product",
            "Promotional Banner": "Create a promotional banner or header image"
        }
        
        image_type = st.selectbox(
            "Image Type",
            list(image_options.keys())
        )
        
        brand_style = st.selectbox(
            "Brand Style",
            ["Minimalist", "Bold & Vibrant", "Luxury", "Playful", "Eco/Natural", "Tech/Futuristic", "Vintage/Retro"]
        )
        
        additional_instructions = st.text_area(
            "Additional Instructions (Optional)",
            placeholder="Add any specific visual elements, colors, or themes you'd like to include..."
        )
        
        if st.button("Generate Image"):
            if product_description and image_type and brand_style:
                with st.spinner("Generating marketing image..."):
                    prompt = f"""
                    Create a professional {image_type.lower()} for:
                    
                    Product: {product_description}
                    Target Audience: {target_audience if target_audience else 'General consumers'}
                    Brand Style: {brand_style}
                    Purpose: {image_options[image_type]}
                    
                    {additional_instructions if additional_instructions else ''}
                    
                    The image should be high quality, visually appealing, and aligned with the brand style.
                    Make sure it would work well for digital marketing campaigns.
                    """
                    
                    image_data, mime_type = generate_image_with_gemini(prompt)
                    if image_data:
                        # Convert from base64 if needed
                        if isinstance(image_data, str):
                            image_data = base64.b64decode(image_data)
                        
                        # Display the image
                        st.image(BytesIO(image_data), caption=f"{image_type} Image", use_column_width=True)
                        
                        # Add download button
                        buffered = BytesIO(image_data)
                        file_extension = mimetypes.guess_extension(mime_type) or ".png"
                        st.download_button(
                            label="Download Image",
                            data=buffered,
                            file_name=f"{image_type.lower().replace(' ', '_')}{file_extension}",
                            mime=mime_type
                        )
                        
                        # Save to session state
                        st.session_state.generated_data["generated_images"].append({
                            "type": image_type,
                            "data": image_data,
                            "mime_type": mime_type,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                        })
                        
                        # Add to history
                        st.session_state.history.append({
                            "tool": "Image Generator",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "inputs": {
                                "product": product_description,
                                "image_type": image_type
                            }
                        })
                    else:
                        st.error("Failed to generate image. Please try again.")
            else:
                st.warning("Please fill all required fields.")
                
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Image Gallery")
        
        if st.session_state.generated_data["generated_images"]:
            # Display most recent images (limited to last 5)
            recent_images = st.session_state.generated_data["generated_images"][-5:]
            
            for idx, img_data in enumerate(recent_images):
                with st.expander(f"{img_data['type']} - {img_data['timestamp']}"):
                    # Display the image
                    st.image(BytesIO(img_data['data']), use_column_width=True)
                    
                    # Add download button
                    buffered = BytesIO(img_data['data'])
                    file_extension = mimetypes.guess_extension(img_data['mime_type']) or ".png"
                    st.download_button(
                        label="Download Image",
                        data=buffered,
                        file_name=f"image_{idx}{file_extension}",
                        mime=img_data['mime_type']
                    )
        else:
            st.info("No images generated yet. Use the generator to create marketing visuals.")
        
        st.markdown("### Image Usage Tips")
        st.markdown("""
        **Best Practices for Marketing Images:**
        
        - **Social Media Sizes:**
          - Instagram: 1080 x 1080px (square)
          - Facebook: 1200 x 630px (landscape)
          - Twitter: 1200 x 675px (landscape)
          - LinkedIn: 1200 x 627px (landscape)
          
        - **Content Tips:**
          - Use high contrast for better visibility
          - Keep text minimal and readable
          - Align with brand color palette
          - Use the rule of thirds for composition
          - Include clear call-to-action when needed
        
        - **Image SEO:**
          - Use descriptive filenames
          - Add alt text when posting
          - Compress without losing quality
        """)
                
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(
        """
        <div style="text-align: center;">
            <p>AI Marketing Agent | Made with ❤️ using Streamlit, Google Gemini, and Gemma</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Campaign Timeline")
        
        start_date = st.date_input("Campaign Start Date")
        end_date = st.date_input("Campaign End Date")
        
        if start_date and end_date and (end_date > start_date):
            # Create sample timeline chart
            timeline_data = pd.DataFrame({
                'Task': ['Research', 'Planning', 'Content Creation', 'Launch', 'Evaluation'],
                'Start': pd.to_datetime([start_date, start_date + pd.Timedelta(days=7), 
                                        start_date + pd.Timedelta(days=14), 
                                        start_date + pd.Timedelta(days=21),
                                        end_date - pd.Timedelta(days=7)]),
                'End': pd.to_datetime([start_date + pd.Timedelta(days=7), 
                                      start_date + pd.Timedelta(days=14), 
                                      start_date + pd.Timedelta(days=21),
                                      end_date - pd.Timedelta(days=7),
                                      end_date])
            })
            
            fig = px.timeline(timeline_data, x_start="Start", x_end="End", y="Task", color="Task")
            fig.update_layout(xaxis_title="Date", yaxis_title="Phase")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select valid start and end dates.")
            
        st.markdown("### Budget Allocation Tool")
        
        # Sample budget allocation
        total_budget = st.number_input("Total Campaign Budget ($)", min_value=0, value=1000)
        
        if total_budget > 0:
            # Preset allocations based on common marketing practice
            allocations = {
                "Social Media Ads": 0.40,
                "Content Creation": 0.25,
                "Influencer Marketing": 0.15,
                "Email Campaigns": 0.10,
                "Analytics & Tools": 0.05,
                "Contingency": 0.05
            }
            
            # Create budget allocation chart
            fig = go.Figure(data=[go.Pie(
                labels=list(allocations.keys()),
                values=[total_budget * v for v in allocations.values()],
                textinfo='label+percent+value',
                hole=.3
            )])
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Allow manual adjustments
            st.markdown("### Adjust Budget Allocation")
            for channel, default_pct in allocations.items():
                allocations[channel] = st.slider(
                    f"{channel} (%)", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=default_pct,
                    step=0.05,
                    format="%.2f"
                )
            
            # Normalize if total exceeds 100%
            total_pct = sum(allocations.values())
            if total_pct > 0:
                normalized = {k: v/total_pct for k, v in allocations.items()}
                
                # Updated chart
                fig2 = go.Figure(data=[go.Pie(
                    labels=list(normalized.keys()),
                    values=[total_budget * v for v in normalized.values()],
                    textinfo='label+percent+value',
                    hole=.3
                )])
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Show actual dollar amounts
                budget_df = pd.DataFrame({
                    'Channel': normalized.keys(),
                    'Percentage': [f"{v*100:.1f}%" for v in normalized.values()],
                    'Amount ($)': [f"${total_budget * v:.2f}" for v in normalized.values()]
                })
                
                st.dataframe(budget_df)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("### Market Analysis Benefits")
        st.markdown("""
        - Identify market opportunities
        - Understand competitor landscape
        - Find optimal positioning strategy
        - Determine best marketing channels
        - Create targeted campaigns
        - Maximize ROI on marketing spend
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Examples")
        st.markdown("""
        **Example Product:** 
        Eco-friendly water bottle with smart temperature tracking
        
        **Example Analysis:**
        - Target: Health-conscious millennials (25-40)
        - Trends: Sustainability, wellness tracking, hydration habits
        - Platforms: Instagram, TikTok, Pinterest
        - Positioning: Premium eco-tech that encourages healthy habits
        """)
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_tab == "Brand Builder":
    st.markdown('<h2 class="sub-header">🌟 Brand Builder</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Brand Identity Creator")
        
        brand_values = st.text_area("Brand Values/Personality", placeholder="Describe the personality and values of your brand...")
        
        if st.button("Generate Brand Identity"):
            if product_description and target_audience and brand_values:
                with st.spinner("Creating brand identity..."):
                    brand_result = brand_building(product_description, target_audience, brand_values)
                    st.session_state.generated_data["brand_identity"] = brand_result
                    
                    # Add to history
                    st.session_state.history.append({
                        "tool": "Brand Builder",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "inputs": {
                            "product": product_description,
                            "audience": target_audience,
                            "values": brand_values
                        }
                    })
                    
                st.success("Brand identity created!")
            else:
                st.warning("Please fill all required fields.")
        
        if st.session_state.generated_data["brand_identity"]:
            st.markdown("### Brand Identity Results")
            st.markdown(st.session_state.generated_data["brand_identity"])
            
            # Export options
            if st.button("Export Brand Identity"):
                report = f"""
                # Brand Identity Report
                
                ## Product Information
                {product_description}
                
                ## Target Audience
                {target_audience}
                
                ## Brand Values/Personality
                {brand_values}
                
                ## Brand Identity Results
                {st.session_state.generated_data["brand_identity"]}
                
                ## Generated on
                {datetime.now().strftime("%Y-%m-%d %H:%M")}
                """
                
                st.download_button(
                    label="Download Brand Identity",
                    data=report,
                    file_name="brand_identity_report.md",
                    mime="text/markdown"
                )
                
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Brand Image Generator")
        
        brand_style = st.selectbox(
            "Brand Style",
            ["Minimalist", "Bold & Vibrant", "Luxury", "Playful", "Eco/Natural", "Tech/Futuristic", "Vintage/Retro"]
        )
        
        image_type = st.selectbox(
            "Image Type",
            ["Logo Concept", "Social Media Banner", "Product Packaging", "Brand Moodboard"]
        )
        
        if st.button("Generate Brand Image"):
            if product_description and brand_style and image_type:
                with st.spinner("Generating brand image..."):
                    image_data, mime_type = generate_brand_imagery(product_description, brand_style, image_type)
                    if image_data:
                        # Convert from base64 if needed
                        if isinstance(image_data, str):
                            image_data = base64.b64decode(image_data)
                        
                        # Display the image
                        display_generated_image(image_data, mime_type)
                        
                        # Save to session state
                        st.session_state.generated_data["generated_images"].append({
                            "type": image_type,
                            "data": image_data,
                            "mime_type": mime_type,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                        })
                    else:
                        st.error("Failed to generate image. Please try again.")
            else:
                st.warning("Please fill all required fields.")
                
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_tab == "Content Creator":
    st.markdown('<h2 class="sub-header">🎨 Content Creator</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Content Generation")
        
        brand_name = st.text_input("Brand Name", placeholder="Enter your brand name...")
        platform = st.selectbox(
            "Platform",
            ["Instagram", "TikTok", "Facebook", "LinkedIn", "Twitter/X", "YouTube", "Pinterest"]
        )
        
        if st.button("Generate Content Ideas"):
            if product_description and target_audience and brand_name and platform:
                with st.spinner("Creating content ideas..."):
                    content_result = content_generation(product_description, target_audience, brand_name, platform)
                    st.session_state.generated_data["content_ideas"] = content_result
                    
                    # Add to history
                    st.session_state.history.append({
                        "tool": "Content Creator",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "inputs": {
                            "product": product_description,
                            "audience": target_audience,
                            "brand": brand_name,
                            "platform": platform
                        }
                    })
                    
                st.success("Content ideas generated!")
            else:
                st.warning("Please fill all required fields.")
        
        if st.session_state.generated_data["content_ideas"]:
            st.markdown("### Content Ideas")
            st.markdown(st.session_state.generated_data["content_ideas"])
            
            # Export options
            if st.button("Export Content Plan"):
                report = f"""
                # Content Plan
                
                ## Product Information
                {product_description}
                
                ## Target Audience
                {target_audience}
                
                ## Brand Name
                {brand_name}
                
                ## Platform
                {platform}
                
                ## Content Ideas
                {st.session_state.generated_data["content_ideas"]}
                
                ## Generated on
                {datetime.now().strftime("%Y-%m-%d %H:%M")}
                """
                
                st.download_button(
                    label="Download Content Plan",
                    data=report,
                    file_name="content_plan.md",
                    mime="text/markdown"
                )
                
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Video Script Generator")
        
        video_type = st.selectbox(
            "Video Type",
            ["Product Demo", "Brand Story", "Tutorial", "Customer Testimonial", "Behind the Scenes"]
        )
        
        duration = st.selectbox(
            "Video Duration",
            ["15 seconds", "30 seconds", "60 seconds", "2-3 minutes"]
        )
        
        if st.button("Generate Video Script"):
            if product_description and brand_name and video_type and duration:
                with st.spinner("Creating video script..."):
                    script_result = generate_video_script(product_description, brand_name, video_type, duration)
                    
                    # Display script
                    st.markdown("### Video Script")
                    st.markdown(script_result)
                    
                    # Download option
                    st.download_button(
                        label="Download Script",
                        data=script_result,
                        file_name=f"{video_type.lower().replace(' ', '_')}_script.md",
                        mime="text/markdown"
                    )
            else:
                st.warning("Please fill all required fields.")
                
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Social Media Response Generator")
        
        comments = st.text_area(
            "Customer Comments",
            placeholder="Paste comments/questions from customers that need responses..."
        )
        
        if st.button("Generate Responses"):
            if comments and product_description:
                with st.spinner("Creating response templates..."):
                    brand_voice = "Professional and friendly" if not brand_name else f"{brand_name}'s voice"
                    responses = generate_social_responses(comments, brand_voice, product_description)
                    
                    # Display responses
                    st.markdown("### Response Templates")
                    st.markdown(responses)
                    
                    # Download option
                    st.download_button(
                        label="Download Responses",
                        data=responses,
                        file_name="social_media_responses.md",
                        mime="text/markdown"
                    )
            else:
                st.warning("Please provide customer comments and product information.")
                
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_tab == "Campaign Planner":
    st.markdown('<h2 class="sub-header">📊 Campaign Planner</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Marketing Campaign Planner")
        
        brand_name = st.text_input("Brand Name", placeholder="Enter your brand name...")
        objectives = st.text_area(
            "Campaign Objectives",
            placeholder="What are your marketing objectives? E.g., increase brand awareness, drive sales, launch new product..."
        )
        
        if st.button("Generate Campaign Plan"):
            if product_description and target_audience and brand_name and objectives:
                with st.spinner("Creating campaign plan..."):
                    campaign_result = campaign_planning(product_description, target_audience, brand_name, objectives)
                    st.session_state.generated_data["campaign_plan"] = campaign_result
                    
                    # Add to history
                    st.session_state.history.append({
                        "tool": "Campaign Planner",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "inputs": {
                            "product": product_description,
                            "audience": target_audience,
                            "brand": brand_name,
                            "objectives": objectives
                        }
                    })
                    
                st.success("Campaign plan created!")
            else:
                st.warning("Please fill all required fields.")
        
        if st.session_state.generated_data["campaign_plan"]:
            st.markdown("### Campaign Plan")
            st.markdown(st.session_state.generated_data["campaign_plan"])
            
            # Export options
            if st.button("Export Campaign Plan"):
                report = f"""
                # Marketing Campaign Plan
                
                ## Product Information
                {product_description}
                
                ## Target Audience
                {target_audience}
                
                ## Brand Name
                {brand_name}
                
                ## Campaign Objectives
                {objectives}
                
                ## Campaign Plan
                {st.session_state.generated_data["campaign_plan"]}
                
                ## Generated on
                {datetime.now().strftime("%Y-%m-%d %H:%M")}
                """
                
                st.download_button(
                    label="Download Campaign Plan",
                    data=report,
                    file_name="marketing_campaign_plan.md",
                    mime="text/markdown"
                )
                
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
