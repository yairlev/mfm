# MFM (Mainframe Analysis Tool)

A command-line tool for analyzing mainframe JCL files.

## Installation

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the dependencies:

```bash
pip install -e .
```

3. Set up environment variables:

Create a `.env` file in the root directory with the following variables:
```bash
VERTEX_MODEL_NAME=gemini-2.0-flash-exp  # The Vertex AI model to use
VERTEX_PROJECT=your-project-id          # Your Google Cloud project ID
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json  # Path to your GCP service account key
```

You'll need to:
1. Create a service account in your Google Cloud Console
2. Generate a JSON key for that service account
3. Save the JSON key file somewhere secure
4. Point GOOGLE_APPLICATION_CREDENTIALS to that file location

4. Run the CLI:

```bash
# Basic usage (uses single model from VERTEX_MODEL_NAME in .env)
mfm domains -f ./path/to/your/mainframe.zip

# Using multiple models (requires specifying an arbiter)
mfm domains -f ./path/to/your/mainframe.zip \
    --models gemini-2.0-flash-exp gemini-2.0-pro \
    --arbiter gemini-2.0-pro

# Using short form options
mfm domains -f ./path/to/your/mainframe.zip \
    -m gemini-2.0-flash-exp gemini-2.0-pro \
    -a gemini-2.0-pro
```

Note: 
- If no models are specified, the tool will use the model defined in VERTEX_MODEL_NAME from your .env file
- When using multiple models, an arbiter must be specified to reconcile their outputs