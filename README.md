# AI Marketing Agent

A comprehensive AI-powered marketing assistant built with Streamlit, Google Gemini, and Gemma models.

## Features

- **Market Analysis:** Analyze product, audience, and industry to identify opportunities.
- **Brand Builder:** Create brand identity including names, taglines, and visual direction.
- **Content Creator:** Generate social media content, captions, hashtags, and video scripts.
- **Campaign Planner:** Develop marketing campaign strategies with timeline and budget tools.
- **Performance Analyzer:** Evaluate marketing performance and suggest optimizations.
- **Image Generator:** Create marketing visuals for various platforms and purposes.

## Tech Stack

- **Frontend:** Streamlit
- **AI Models:**
  - Google Gemini (for general text generation, multimodal inputs, and image generation)
  - Google Gemma 3 (for specialized marketing content and copy)
- **Data Visualization:** Plotly
- **Data Processing:** Pandas, NumPy

## Setup Instructions

### Prerequisites

- Python 3.9+
- Gemini API key
- HuggingFace token (optional, for Gemma model)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ai-marketing-agent.git
   cd ai-marketing-agent
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   HF_TOKEN=your_huggingface_token
   ```

### Running the App

Start the Streamlit app:
```
streamlit run app.py
```

## Deployment

The app can be deployed to Streamlit Cloud:

1. Push your code to GitHub
2. Connect your GitHub repository to Streamlit Cloud
3. Add your API keys as secrets in the Streamlit Cloud dashboard
4. Deploy!

## Usage Guide

### Market & Product Analysis

Enter your product description, target audience, and industry to get detailed market analysis including:
- Target audience insights
- Trending themes
- Platform recommendations
- Positioning strategy
- Competitor analysis

### Brand Builder

Create a complete brand identity with:
- Brand name suggestions
- Tagline options
- Voice and tone guidelines
- Brand story elements
- Visual direction

### Content Creator

Generate platform-specific content including:
- Social media posts
- Captions and hashtags
- Video content ideas
- Content calendars
- Engagement strategies

### Campaign Planner

Develop marketing campaigns with:
- Campaign themes and messaging
- Channel strategy
- Timeline planning
- Budget allocation
- Success metrics

### Performance Analyzer

Evaluate marketing performance with:
- Data visualization
- Performance metrics
- Optimization recommendations
- A/B testing suggestions

### Image Generator

Create marketing visuals for:
- Product showcases
- Social media posts
- Advertisements
- Brand stories
- Promotional banners

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
