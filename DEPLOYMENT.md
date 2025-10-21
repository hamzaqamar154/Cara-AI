# Streamlit Cloud Deployment

## Prerequisites

1. GitHub account
2. Streamlit Cloud account (free at https://streamlit.io/cloud)
3. Groq API key

## Deployment Steps

1. **Push your code to GitHub**
   - Ensure all files are committed and pushed to your repository

2. **Connect to Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"
   - Select your repository: `hamzaqamar154/Cara-AI`
   - Set Main file path: `ui/app.py`
   - Click "Deploy"

3. **Configure Secrets**
   - In Streamlit Cloud, go to your app settings
   - Click "Secrets" in the sidebar
   - Add the following secrets:
     ```
     GROQ_API_KEY=your_groq_api_key_here
     LLM_MODEL=llama-3.1-8b-instant
     API_BASE_URL=https://api.groq.com/openai/v1
     ```
   - Optional (with defaults):
     ```
     CHUNK_SIZE=800
     CHUNK_OVERLAP=200
     ```

4. **Deploy**
   - Streamlit Cloud will automatically detect changes and redeploy
   - Check the logs if there are any errors

## Notes

- The app uses Streamlit secrets for environment variables (falls back to .env locally)
- Vector store and data are stored in the app's file system (ephemeral - resets on redeploy)
- For persistent storage, consider using external storage (S3, etc.)

