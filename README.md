# MIRA

## Setup 

### Setup Python Environment

1. Use conda or virtualenv to create a new python 3.10 environment.
2. Install the required packages using the following command:
```bash
pip install -r requirements.txt
```
  - If you run into an issue building wheels for hnswlib or chroma-hnswlib, you may need to install an underlying system package.

### Setup Hugging Face API Token

1. Go to the HuggingFace website and create an account.
2. Go to the settings page and create a new API token.
3. Edit token permissions for the following:
    - `Read access to contents of all public gated repos you can access`
    - `Make calls to the serverless Inference API`
    - `Manage Inference Endpoints`

### Setup .env file

1. Create a `.env` file in the root directory of the project.
2. Add the following environment variables to the `.env` file:
```bash
HUGGINGFACEHUB_API_TOKEN="<Put your token here>"
```

### Setup Streamlit app secrets

1. Set up the .streamlit directory and secrets file.

```bash
mkdir .streamlit
touch .streamlit/secrets.toml
chmod 0600 .streamlit/secrets.toml
```

2. Edit secrets.toml

```toml
HUGGINGFACEHUB_API_TOKEN="<Put the same token here>"
```

## Usage

Run the streamlit app to see how the model works with RAG.

```bash
streamlit run streamlit_app.py
```