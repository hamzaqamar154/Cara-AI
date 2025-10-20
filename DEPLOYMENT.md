# Deployment Guide

## Deploying on Streamlit Cloud

### Prerequisites

1. GitHub account
2. Streamlit Cloud account (free at https://share.streamlit.io)
3. Repository pushed to GitHub

### Steps

1. **Push your code to GitHub**
   - Make sure all code is committed and pushed to your repository
   - Repository should be public (or you need Streamlit Cloud Pro for private repos)

2. **Sign in to Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Sign in with your GitHub account

3. **Deploy your app**
   - Click "New app"
   - Select your repository: `hamzaqamar154/Cara-AI`
   - Set the main file path: `ui/app.py`
   - Set the branch: `main`

4. **Add Environment Variables (Secrets)**
   - In the Streamlit Cloud dashboard, go to your app settings
   - Click on "Secrets" or "Advanced settings"
   - Add the following secrets:
     ```
     GROQ_API_KEY=your_groq_api_key_here
     API_BASE_URL=https://api.groq.com/openai/v1
     LLM_MODEL=llama-3.1-8b-instant
     ```
   - Optional: Set `CHUNK_SIZE` and `CHUNK_OVERLAP` if you want to customize

5. **Deploy**
   - Click "Deploy"
   - Streamlit Cloud will build and deploy your app
   - Your app will be available at: `https://your-app-name.streamlit.app`

### Important Notes

- The `.env` file is in `.gitignore` and won't be pushed to GitHub (this is correct for security)
- All environment variables must be set in Streamlit Cloud's Secrets section
- The app will automatically use the secrets you configure in Streamlit Cloud
- First deployment may take a few minutes to install dependencies

### Troubleshooting

- If the app fails to start, check the logs in Streamlit Cloud dashboard
- Ensure all dependencies in `requirements.txt` are compatible
- Verify that your secrets are set correctly (no extra spaces or quotes)

