# Security and release checklist

- Store API keys in environment variables or an untracked `.env` only.
- Store YouTube cookies outside the repository and pass their path at runtime.
- Authenticate `hf` with its credential store or `HF_TOKEN`; never hard-code a token.
- Before every release, scan both contents and filenames for secrets and absolute paths.
- Model-evaluation manifests contain local filesystem paths; keep them under the
  ignored `work/` directory and publish only aggregate metric JSON when needed.
- If a real key or token was ever committed or shared, removing it from the repository is not enough: revoke or rotate it at the provider.
