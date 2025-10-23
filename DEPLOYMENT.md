# Deploying to Streamlit Community Cloud

This guide will help you deploy the Veterinary AI Assistant to Streamlit Community Cloud.

## Prerequisites

1. A GitHub account
2. Your Groq API key (get it from https://console.groq.com)
3. This repository pushed to GitHub

## Step-by-Step Deployment

### 1. Prepare Your Repository

Make sure these files are in your repository:
- ✅ `app.py` (main Streamlit app)
- ✅ `main.py` (CrewAI agents)
- ✅ `vector_db.py` (vector database)
- ✅ `requirements-streamlit.txt` (optimized dependencies)
- ✅ `.streamlit/config.toml` (Streamlit configuration)
- ✅ `packages.txt` (system dependencies)
- ✅ `.gitignore` (to avoid committing secrets)

**Important:** Make sure `.env` is in your `.gitignore` to avoid exposing API keys!

### 2. Push to GitHub

```bash
# If not already initialized
git init
git add .
git commit -m "Prepare for Streamlit Cloud deployment"

# Create a new repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

### 3. Deploy on Streamlit Community Cloud

1. **Go to Streamlit Cloud:**
   Visit https://share.streamlit.io/

2. **Sign in with GitHub:**
   Click "Sign in" and authorize with your GitHub account

3. **Create New App:**
   - Click "New app"
   - Select your repository
   - Select the branch (usually `main` or `master`)
   - Set **Main file path** to: `app.py`
   - Click "Advanced settings" (optional but recommended)

4. **Configure Advanced Settings:**

   **Python version:** 3.11 or 3.12

   **Requirements file:** Change from `requirements.txt` to `requirements-streamlit.txt`

   **Secrets:** Add your API keys in TOML format:
   ```toml
   GROQ_API_KEY = "your_actual_groq_api_key_here"
   ```

5. **Deploy:**
   Click "Deploy!" and wait 5-10 minutes for the initial deployment

### 4. Initialize Vector Database

After first deployment, you'll need to initialize the vector database:

**Option A - Run locally then commit:**
```bash
python vector_db.py
git add vector_db/
git commit -m "Add initialized vector database"
git push
```

**Option B - Add initialization to app.py:**
The app can initialize the database on first run automatically (already handled in the code).

### 5. Monitor Deployment

- Check the deployment logs for any errors
- Common issues:
  - **Out of memory:** The free tier has 1GB RAM limit. If this happens, you may need to upgrade or use Railway/Render
  - **Dependencies failing:** Check if all packages in requirements-streamlit.txt are compatible
  - **API rate limits:** Groq has rate limits on free tier

## Post-Deployment

### Managing Secrets

To update your API keys:
1. Go to your app dashboard on Streamlit Cloud
2. Click the three dots menu → "Settings"
3. Go to "Secrets" section
4. Update the TOML configuration

### Updating Your App

Every time you push to GitHub, Streamlit Cloud will automatically redeploy:
```bash
git add .
git commit -m "Update feature X"
git push
```

### Monitoring Usage

- View logs in the Streamlit Cloud dashboard
- Check app analytics in the dashboard
- Monitor API usage on Groq console

## Troubleshooting

### App Won't Start
- Check logs in Streamlit Cloud dashboard
- Verify all secrets are correctly formatted in TOML
- Ensure requirements-streamlit.txt has all necessary packages

### Out of Memory Errors
The app uses heavy ML models. Solutions:
1. Upgrade to Streamlit Cloud paid tier
2. Use lighter embedding models
3. Deploy to Railway or Render instead (see main README)

### Slow Performance
- First load is always slow (models need to download)
- Consider caching with `@st.cache_resource` for model loading
- Use Streamlit Cloud's caching features

### Vector Database Not Persisting
- Make sure `vector_db/` directory is committed to git
- Or initialize in app.py on first run (already implemented)

## Cost Considerations

- **Streamlit Cloud Free Tier:**
  - 1 private app OR unlimited public apps
  - 1GB RAM, 1 CPU core
  - 1GB storage

- **Groq API Free Tier:**
  - Rate limits apply
  - Monitor usage at https://console.groq.com

## Security Best Practices

1. ✅ Never commit `.env` file to GitHub
2. ✅ Use Streamlit Cloud secrets for API keys
3. ✅ Keep your repository private if handling sensitive data
4. ✅ Regularly rotate API keys
5. ✅ Monitor API usage for anomalies

## Need Help?

- Streamlit Community Forum: https://discuss.streamlit.io/
- Streamlit Documentation: https://docs.streamlit.io/
- CrewAI Documentation: https://docs.crewai.com/

## Alternative Deployment Options

If Streamlit Cloud doesn't meet your needs, consider:
- **Railway:** Better resources, ~$5-20/month
- **Render:** Similar to Railway, free tier available
- **Google Cloud Run:** Pay-per-use, scales to zero
- **Hugging Face Spaces:** Great for AI/ML apps
