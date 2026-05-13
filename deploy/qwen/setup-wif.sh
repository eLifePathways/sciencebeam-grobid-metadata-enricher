#!/usr/bin/env bash
# One-time GCP setup so the qwen-bench workflow can authenticate via
# Workload Identity Federation (no JSON key files in GitHub secrets).
#
# Idempotent: re-run safely. Each step probes for existing resources and
# only creates what's missing.
#
# After this script runs, set the following GitHub Actions repo variables
# under Settings -> Secrets and variables -> Actions -> Variables:
#   GCP_PROJECT        = <project-id>
#   GCP_WIF_PROVIDER   = projects/<project-number>/locations/global/workloadIdentityPools/<pool>/providers/<provider>
#   GCP_BENCH_SA       = <sa-email>
#   GCP_LORA_BUCKET    = <bucket-name>      (default: <project>-qwen-bench)
#
# Usage:
#   PROJECT=<id> GH_REPO=eLifePathways/sciencebeam-grobid-metadata-enricher \
#     bash deploy/qwen/setup-wif.sh

set -euo pipefail

: "${PROJECT:?Set PROJECT=<gcp-project-id>}"
: "${GH_REPO:?Set GH_REPO=<owner/repo>}"

POOL="${POOL:-github-pool}"
PROVIDER="${PROVIDER:-github-provider}"
SA_NAME="${SA_NAME:-qwen-bench-ci}"
SA_EMAIL="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
BUCKET="${BUCKET:-${PROJECT}-qwen-bench}"

PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"

enable_apis() {
  echo "[wif] enabling APIs"
  gcloud services enable \
    iam.googleapis.com \
    iamcredentials.googleapis.com \
    cloudresourcemanager.googleapis.com \
    compute.googleapis.com \
    storage.googleapis.com \
    sts.googleapis.com \
    --project="$PROJECT" --quiet
}

ensure_pool() {
  echo "[wif] ensuring identity pool $POOL"
  if ! gcloud iam workload-identity-pools describe "$POOL" \
        --project="$PROJECT" --location=global --quiet >/dev/null 2>&1; then
    gcloud iam workload-identity-pools create "$POOL" \
      --project="$PROJECT" --location=global \
      --display-name="GitHub Actions" --quiet
  fi
}

ensure_provider() {
  echo "[wif] ensuring provider $PROVIDER bound to $GH_REPO"
  if ! gcloud iam workload-identity-pools providers describe "$PROVIDER" \
        --project="$PROJECT" --location=global \
        --workload-identity-pool="$POOL" --quiet >/dev/null 2>&1; then
    gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" \
      --project="$PROJECT" --location=global \
      --workload-identity-pool="$POOL" \
      --display-name="GitHub" \
      --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
      --attribute-condition="assertion.repository=='${GH_REPO}'" \
      --issuer-uri="https://token.actions.githubusercontent.com" --quiet
  fi
}

ensure_sa() {
  echo "[wif] ensuring service account $SA_EMAIL"
  if ! gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT" --quiet >/dev/null 2>&1; then
    gcloud iam service-accounts create "$SA_NAME" \
      --project="$PROJECT" \
      --display-name="qwen-bench CI" \
      --description="Spins / tears down spot Qwen bench cluster" --quiet
  fi
}

grant_roles() {
  echo "[wif] granting roles to $SA_EMAIL"
  for role in \
      roles/compute.instanceAdmin.v1 \
      roles/iam.serviceAccountUser \
      roles/compute.networkUser; do
    gcloud projects add-iam-policy-binding "$PROJECT" \
      --member="serviceAccount:${SA_EMAIL}" --role="$role" \
      --condition=None --quiet >/dev/null
  done
  gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
    --member="serviceAccount:${SA_EMAIL}" --role=roles/storage.objectViewer --quiet >/dev/null \
    || echo "[wif] (bucket $BUCKET not present yet; create later and add binding)"
}

bind_wif_to_sa() {
  echo "[wif] binding GH repo $GH_REPO to $SA_EMAIL via WIF"
  gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
    --project="$PROJECT" \
    --role=roles/iam.workloadIdentityUser \
    --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/attribute.repository/${GH_REPO}" \
    --quiet >/dev/null
}

enable_apis
ensure_pool
ensure_provider
ensure_sa
grant_roles
bind_wif_to_sa

cat <<EOF

[wif] DONE — set these in GitHub repo > Settings > Variables:

  GCP_PROJECT        = $PROJECT
  GCP_WIF_PROVIDER   = projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/providers/${PROVIDER}
  GCP_BENCH_SA       = ${SA_EMAIL}
  GCP_LORA_BUCKET    = ${BUCKET}

EOF
