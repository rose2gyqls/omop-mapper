# OMOP Mapper

Map clinical entity names to OMOP CDM standard concepts with a 3-stage pipeline:

1. Candidate retrieval from Elasticsearch
2. Standard concept collection
3. LLM-based final scoring

The repository now includes a local Streamlit app so users can clone the project, configure their own secrets locally, and try single-entity mapping from a browser.

## What This Deployment Supports

- Local demo UI with Streamlit
- Single-entity mapping by `entity name` and `domain`
- OpenAI-backed scoring
- Remote Elasticsearch indexes hosted outside this repository

## What This Deployment Does Not Do

- It does not publish Elasticsearch credentials to GitHub
- It does not require users to run indexing before trying the UI
- It does not replace the existing CLI, indexing, or evaluation workflows

## Prerequisites

- Python 3.9+
- An OpenAI API key
- Elasticsearch connection details for the hosted OMOP indexes:
  - `ES_SERVER_HOST`
  - `ES_SERVER_PORT`
  - `ES_SERVER_USERNAME`
  - `ES_SERVER_PASSWORD`

## Quick Start

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/omop-mapper.git
cd omop-mapper

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create your local environment file:

```bash
cp .env.example .env
```

Fill in `.env` with your local secrets:

```bash
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-5-mini-2025-08-07

ES_SERVER_HOST=your-es-host
ES_SERVER_PORT=9200
ES_SERVER_USERNAME=your-es-username
ES_SERVER_PASSWORD=your-es-password
ES_USE_SSL=false
```

Run the local app:

```bash
streamlit run app.py
```

Then open the browser URL printed by Streamlit, enter:

- `Entity name`
- `Target domain`

The app will run the existing mapping pipeline and show ranked OMOP candidates.

## Deploy on Render

This repository includes a [`render.yaml`](./render.yaml) Blueprint for deploying the Streamlit app as a Render web service.

### Recommended path: Render Blueprint

1. Push this repository to GitHub
2. In Render, click `New +` → `Blueprint`
3. Select this repository
4. Review the `render.yaml` settings before the first deploy:
   - `plan: free`
   - `region: singapore`
   - `buildCommand: pip install -r requirements.txt`
   - `startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
5. Provide the prompted secret values for:
   - `OPENAI_API_KEY`
   - `ES_SERVER_HOST`
   - `ES_SERVER_USERNAME`
   - `ES_SERVER_PASSWORD`
6. Confirm the non-secret defaults, or override them if needed:
   - `PYTHON_VERSION=3.11.11`
   - `OPENAI_MODEL=gpt-5-mini-2025-08-07`
   - `ES_SERVER_PORT=9200`
   - `ES_USE_SSL=false`
7. Create the Blueprint and wait for the first deploy to finish

After deployment, Render will assign a public URL like `https://mapomop.onrender.com`.

### Manual path: Render Web Service

If you prefer not to use the Blueprint, create a `Web Service` in the Render dashboard with:

- Runtime: `Python 3`
- Build Command: `pip install -r requirements.txt`
- Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
- Health Check Path: `/_stcore/health`

Then add the environment variables from the table below in the Render `Environment` settings.

### Custom domain

To use `mapomop.com` after the app is live:

1. Open the Render service
2. Go to `Settings` → `Custom Domains`
3. Add `mapomop.com`
4. Update your DNS records with your domain provider using the values Render shows
5. Optionally add `www.mapomop.com` if you want a separate redirect target

Render keeps the `onrender.com` subdomain unless you explicitly disable it in the dashboard.

## Local App Behavior

The Streamlit app:

- Reads OpenAI and Elasticsearch settings from your local `.env`
- Verifies Elasticsearch connectivity on load
- Uses the existing `EntityMappingAPI`
- Returns the top mapped concept plus ranked candidates

This keeps the core mapping logic untouched while giving users a simple local UI.

## Required Environment Variables

| Variable | Required | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM scoring |
| `OPENAI_MODEL` | No | OpenAI model override |
| `ES_SERVER_HOST` | Yes | Elasticsearch host |
| `ES_SERVER_PORT` | No | Elasticsearch port, default `9200` |
| `ES_SERVER_USERNAME` | Yes | Elasticsearch username |
| `ES_SERVER_PASSWORD` | Yes | Elasticsearch password |
| `ES_USE_SSL` | No | `true` or `false`, default `false` |
| `PYTHON_VERSION` | Render only | Python runtime version, recommended `3.11.11` |

## Security Notes

- Do not commit `.env`
- Do not hardcode Elasticsearch credentials in tracked source files
- Share Elasticsearch credentials with users out of band
- Use read-only Elasticsearch credentials for local demo users
- In Render, store secrets in the service `Environment` page or the initial Blueprint prompt

## Existing CLI Workflows

The repository still includes the original command-line workflows.

### Mapping CLI

```bash
./scripts/run_mapping.sh snuh
```

Useful options:

- `--sample-size`, `-n`
- `--sample-per-domain`
- `--random`
- `--workers`, `-w`
- `--repeat`, `-r`
- `--scoring`
- `--llm-provider`
- `--llm-model`

Outputs are written to `test_logs/`.

### Indexing CLI

```bash
./scripts/index.sh
```

This is only needed if you want to build or rebuild OMOP indexes yourself. It is not required for the local Streamlit demo when you already have access to a hosted Elasticsearch cluster.

If you use the PostgreSQL indexing path, provide `PG_HOST`, `PG_PORT`, `PG_DBNAME`, `PG_USER`, and `PG_PASSWORD` through your local `.env` or explicit CLI flags.

### Evaluation Utilities

The `eval/` directory remains available for offline evaluation and analysis workflows. The local UI does not change those scripts.

## Python API

```python
from MapOMOP import EntityInput, EntityMappingAPI, DomainID

api = EntityMappingAPI()
entity = EntityInput(entity_name="myocardial ischemia", domain_id=DomainID.CONDITION)
results = api.map_entity(entity)
```

## Project Structure

```text
omop-mapper/
├── app.py                    # Streamlit local demo UI
├── run_indexing.py           # Indexing CLI
├── run_mapping.py            # Mapping CLI
├── indexing/                 # Index-building pipeline
├── eval/                     # Evaluation scripts
├── scripts/                  # Shell wrappers and utilities
├── src/MapOMOP/              # Core mapping package
└── .env.example              # Local config template
```

## Troubleshooting

### Missing configuration

If the app says configuration is missing:

1. Check that `.env` exists in the project root
2. Confirm `OPENAI_API_KEY` is filled in
3. Confirm all required `ES_SERVER_*` values are filled in

### Elasticsearch connection failure

If the app cannot reach Elasticsearch:

1. Confirm host and port
2. Confirm username and password
3. Check whether `ES_USE_SSL` should be `true`
4. Verify the cluster is reachable from your local machine
5. If deployed on Render, verify the cluster is also reachable from the Render region you selected

### Render deploy fails to boot

If Render shows a deploy or health check failure:

1. Confirm the service is a `Web Service`, not a static site
2. Confirm the start command uses `--server.port $PORT --server.address 0.0.0.0`
3. Confirm the health check path is `/_stcore/health`
4. Confirm all required secrets are present in Render
5. Check the Render logs for import or dependency errors
6. If needed, pin a different `PYTHON_VERSION` in Render and redeploy

### No mapping results

If a term returns no candidates:

1. Try a different domain
2. Try a normalized or less specific source phrase
3. Confirm the target indexes contain the expected vocabularies

## License

MIT
