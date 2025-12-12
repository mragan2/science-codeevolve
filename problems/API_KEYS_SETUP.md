# API Keys Setup Guide

Quick guide for setting up API keys in CodeEvolve projects.

## Three Methods (Choose One)

### Method 1: In run.sh File (Quickest)

**Pros:** Easy, works immediately  
**Cons:** Less secure, must not commit to git

```bash
# Edit your project's run.sh
cd problems/YOUR_PROJECT
nano run.sh

# Find the API CONFIGURATION section and uncomment/set:
API_KEY="your-api-key-here"
API_BASE="https://api.openai.com/v1"
```

⚠️ **Important:** If you do this, add `run.sh` to your project's `.gitignore` to prevent accidentally committing keys!

---

### Method 2: Environment Variables (Development)

**Pros:** Secure, no files to manage  
**Cons:** Must set every time you open a new terminal

```bash
# Set in your terminal
export API_KEY="your-api-key-here"
export API_BASE="https://api.openai.com/v1"

# Then run normally
bash problems/YOUR_PROJECT/run.sh
```

**Make it permanent** (add to `~/.bashrc` or `~/.zshrc`):
```bash
echo 'export API_KEY="your-api-key-here"' >> ~/.bashrc
echo 'export API_BASE="https://api.openai.com/v1"' >> ~/.bashrc
source ~/.bashrc
```

---

### Method 3: External File (Recommended - Most Secure)

**Pros:** Secure, reusable, git-safe  
**Cons:** One extra step to set up

#### Step 1: Create API keys file

```bash
# Copy the example
cp problems/.api_keys.example problems/.api_keys

# Edit with your actual keys
nano problems/.api_keys
```

Your `problems/.api_keys` file should look like:
```bash
# Your actual keys
export API_KEY="sk-your-real-api-key-here"
export API_BASE="https://api.openai.com/v1"
```

#### Step 2: Reference it in run.sh

Edit your project's `run.sh` and uncomment this line in the API CONFIGURATION section:
```bash
source problems/.api_keys
```

Or if your run.sh is in the project folder:
```bash
source ../.api_keys
```

#### Step 3: Run normally
```bash
bash problems/YOUR_PROJECT/run.sh
```

The `.api_keys` file is automatically ignored by git for security.

---

## API Endpoints by Provider

### OpenAI
```bash
export API_KEY="sk-..."
export API_BASE="https://api.openai.com/v1"
```

### Google Gemini
```bash
export API_KEY="AIza..."
export API_BASE="https://generativelanguage.googleapis.com/v1beta"
```

### Azure OpenAI
```bash
export API_KEY="your-azure-key"
export API_BASE="https://your-resource.openai.azure.com"
```

### Anthropic Claude
```bash
export API_KEY="sk-ant-..."
export API_BASE="https://api.anthropic.com/v1"
```

### Local/Self-hosted (e.g., Ollama, vLLM)
```bash
export API_KEY=""  # Often not needed for local
export API_BASE="http://localhost:8080/v1"
```

---

## Verification

Check if your API keys are set:

```bash
# Check environment
echo $API_KEY
echo $API_BASE

# Or look for the warning in run output
bash run.sh
# Should NOT show: "WARNING: API_KEY is not set"
```

---

## Security Best Practices

✅ **DO:**
- Use Method 3 (external file) for production
- Add `.api_keys` to `.gitignore` (already done)
- Use different keys for different projects/teams
- Rotate keys periodically

❌ **DON'T:**
- Commit API keys to git
- Share keys in chat/email
- Use production keys for testing
- Store keys in plaintext in public places

---

## Troubleshooting

### "WARNING: API_KEY is not set"

The run script detected no API key. Fix using any method above.

### "Authentication failed" or "Invalid API key"

- Check your key is correct (no extra spaces)
- Verify the API_BASE matches your provider
- Ensure the key hasn't expired
- Try the key in a simple curl test:

```bash
curl $API_BASE/models \
  -H "Authorization: Bearer $API_KEY"
```

### "source: .api_keys: file not found"

- Check the path in your run.sh is correct
- If run.sh is in project folder, use `../.api_keys`
- Verify the file exists: `ls -la problems/.api_keys`

### Keys work in terminal but not in run.sh

If you set environment variables but they don't work in run.sh:
- Make sure to `export` (not just set) the variables
- Or use Method 1 or 3 instead

---

## Quick Reference Card

```bash
# Method 1: Direct in run.sh
API_KEY="..." in run.sh

# Method 2: Environment
export API_KEY="..."
export API_BASE="..."

# Method 3: External file
source problems/.api_keys

# Check if set
echo $API_KEY

# Run
bash problems/YOUR_PROJECT/run.sh
```

---

For more details, see the main [README.md](README.md).
