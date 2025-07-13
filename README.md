# Pubmed (Taylor's Version) - Enhanced Literature Search

An AI-powered PubMed search tool with intelligent ranking, journal impact factors, and secure access.

## Features

- 🔐 **Secure Access**: Password-protected with session management
- ⚡ **AI Analysis**: Intelligent relevance ranking and synthesis
- 📊 **Impact Factors**: Journal ranking with JIF integration
- 🎯 **Smart Search**: Multiple search strategies for comprehensive results
- 📥 **Export**: RIS format for reference managers
- 🧠 **Learning**: Adaptive journal matching that improves over time

## Environment Variables

Required for deployment:

- `ACCESS_PASSWORD`: Password for accessing the application
- `OPENAI_API_KEY`: For AI features (optional but recommended)
- `NCBI_API_KEY`: For faster PubMed access (optional)
- `NCBI_EMAIL`: Your email for NCBI requests

## Local Development

1. Clone the repository
2. Create `.env` file from `.env.example`
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `python app.py`
5. Open: `http://localhost:8000`

## Deployment

This application is configured for deployment on Render with the included `render.yaml` file.

## Security

- Session-based authentication
- IP tracking for access control
- Secure cookie management
- Environment variable protection
