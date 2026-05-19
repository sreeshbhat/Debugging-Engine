# BugFix Prompt Arena

BugFix Prompt Arena is a Streamlit practice app for AI-assisted debugging. Students are shown a buggy scenario and must write the prompt they would send to an AI coding assistant. The app scores the prompt quality instead of solving the bug directly.

## Features

- Student entry gate with required name and roll number before starting
- Gamified scoring with badges, grade display, progress bar, leaderboard, and attempt history
- Teacher-only controls for revealing the hidden fix, viewing all submissions, and exporting CSV
- Automatic load splitting across configured Gemini, Groq, and Cohere API keys
- Manual provider override from the sidebar when needed
- MongoDB storage with local JSON fallback

## Project Structure

```text
bugfix-prompt-arena/
|-- app.py
|-- requirements.txt
|-- .env.example
|-- README.md
|-- data/
|   `-- challenges.json
|-- storage/
|   `-- results.json
|-- services/
|   |-- __init__.py
|   |-- llm_service.py
|   |-- mongo_service.py
|   `-- scoring_service.py
`-- utils/
    |-- __init__.py
    `-- helpers.py
```

## Run Locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file from `.env.example`.
4. Start the app:

```bash
streamlit run app.py
```

## Environment Variables

Use `.env` locally or `st.secrets` on Streamlit Community Cloud:

```env
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
MONGODB_URI=mongodb://localhost:27017
TEACHER_PASSWORD=admin123
```

Notes:

- If more than one provider key is configured, `Auto Balanced` mode hash-splits students across providers to reduce rate-limit pressure.
- If no MongoDB connection works, the app stores results in `storage/results.json`.
- If `TEACHER_PASSWORD` is missing, the default password is `admin123`.

## Teacher Mode

- Enable teacher mode from the sidebar.
- Enter the teacher password.
- Teachers can reveal the actual fix for a challenge, inspect all student submissions, review the leaderboard, check the average score, and download the results CSV.

## Badges

- `90-100`: Debugging Prompt Master
- `80-89`: AI Coding Pro
- `60-79`: Good Debugger
- `40-59`: Needs Better Context
- `0-39`: Prompt Too Vague

## Deployment

### Streamlit Community Cloud

1. Push the project to GitHub.
2. Create a new Streamlit app pointing to `app.py`.
3. Add the environment values in the app Secrets panel.
4. Deploy.

### VPS

1. Install Python and create a virtual environment.
2. Install the requirements.
3. Set environment variables.
4. Run `streamlit run app.py --server.port 8501`.

## Add New Challenges

Append more objects to [data/challenges.json](data/challenges.json) using the existing schema. The app will load them automatically on restart.
