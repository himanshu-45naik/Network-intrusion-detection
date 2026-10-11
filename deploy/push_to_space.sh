#!/usr/bin/env bash
# Deploys the inference service to a Hugging Face Docker Space.
#
# A Space builds from its own git repo and needs a README with YAML frontmatter
# at the root, so this assembles a minimal Space repo in a temp directory rather
# than pushing this repository as-is: the Space gets only the serving code, the
# model bundle and a Dockerfile, not the dataset, pipeline or training stack.
#
# Prerequisites:
#   - a Hugging Face account and a write token (https://huggingface.co/settings/tokens)
#   - artifacts/nids_model.joblib built by scripts/train_serving_model.py
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   ./deploy/push_to_space.sh <hf-username> <space-name>

set -euo pipefail

USER="${1:?usage: push_to_space.sh <hf-username> <space-name>}"
SPACE="${2:?usage: push_to_space.sh <hf-username> <space-name>}"
: "${HF_TOKEN:?set HF_TOKEN to a Hugging Face write token}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

for f in artifacts/nids_model.joblib artifacts/samples.json artifacts/model_card.json; do
  [[ -f "$f" ]] || { echo "missing $f - run scripts/train_serving_model.py first" >&2; exit 1; }
done

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp -R serving "$STAGE/serving"
mkdir -p "$STAGE/artifacts"
cp artifacts/nids_model.joblib artifacts/samples.json artifacts/model_card.json "$STAGE/artifacts/"
cp requirements-serving.txt "$STAGE/"
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} +

# Spaces serve on 7860 unless app_port says otherwise; 7860 keeps the default.
sed 's/^ENV PORT=8000$/ENV PORT=7860/; s/127.0.0.1:8000/127.0.0.1:7860/; s/^EXPOSE 8000$/EXPOSE 7860/' \
  Dockerfile > "$STAGE/Dockerfile"

MODEL_VERSION="$(python3 -c 'import json;print(json.load(open("artifacts/model_card.json"))["git_sha"])')"

cat > "$STAGE/README.md" <<EOF
---
title: Network Intrusion Detection
emoji: 🛡️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Network Intrusion Detection API

Random Forest intrusion detector trained on CICIDS2017 flow features, served
over FastAPI. Built from commit \`${MODEL_VERSION}\` of
[the source repository](https://github.com/${GITHUB_REPO:-himanshu-45naik/Network-intrusion-detection}).

- \`/\` interactive demo
- \`/docs\` OpenAPI reference
- \`/model\` served model card and held-out metrics
- \`/metrics\` Prometheus exposition
- \`/drift\` feature drift (PSI) against the training distribution
EOF

# Create the Space if it does not exist yet. 409 means it already does, which is
# the normal case on every redeploy.
CREATE_STATUS="$(curl -s -o /dev/null -w '%{http_code}' -X POST \
  https://huggingface.co/api/repos/create \
  -H "Authorization: Bearer ${HF_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"type\":\"space\",\"name\":\"${SPACE}\",\"sdk\":\"docker\",\"private\":false}")"

case "$CREATE_STATUS" in
  200|201) echo "Created Space ${USER}/${SPACE}" ;;
  409)     echo "Space ${USER}/${SPACE} already exists, redeploying" ;;
  401|403) echo "Token rejected (HTTP ${CREATE_STATUS}) - needs write scope" >&2; exit 1 ;;
  *)       echo "Space creation returned HTTP ${CREATE_STATUS}, attempting push anyway" >&2 ;;
esac

cd "$STAGE"
git init -q
git checkout -q -b main
git add -A
git -c user.email="deploy@local" -c user.name="deploy" commit -qm "Deploy model ${MODEL_VERSION}"
git push -q --force "https://${USER}:${HF_TOKEN}@huggingface.co/spaces/${USER}/${SPACE}" main

echo "Pushed to https://huggingface.co/spaces/${USER}/${SPACE}"
echo "Live at https://${USER}-${SPACE}.hf.space (first build takes a few minutes)"
