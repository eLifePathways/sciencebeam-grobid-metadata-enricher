#!/usr/bin/env bash
# Provision Azure Managed Grafana, attach the bench Postgres as a data source,
# and publish the dashboard at benchmarks/grafana/sciencebeam-bench.json.
#
# Idempotent: re-running is safe; existing resources are reused. Requires:
# - az login with Contributor on RG
# - benchmarks/provision_azure.sh has already run (Postgres + DSN exist)
# - PG_PW env var OR ~/.config/sb-bench/pg_pw with the Postgres admin password
set -euo pipefail

RG="${RG:-sb-bench-rg}"
LOC="${LOC:-westus3}"
WORKSPACE="${WORKSPACE:-sb-bench-grafana}"
PG_SERVER="${PG_SERVER:-sb-bench-pg-leon}"
PG_DB="${PG_DB:-sciencebeam_bench}"
PG_USER="${PG_USER:-sbbench}"
DASHBOARD_JSON="${DASHBOARD_JSON:-$(dirname "$0")/grafana/sciencebeam-bench.json}"

if [[ -z "${PG_PW:-}" ]]; then
  PG_PW="$(cat "${HOME}/.config/sb-bench/pg_pw")"
fi

az provider register --namespace Microsoft.Dashboard --wait >/dev/null

# Allow Azure-internal traffic (AMG runs in Azure infra).
az postgres flexible-server firewall-rule show \
    -g "$RG" --name "$PG_SERVER" --rule-name AllowAllAzureServices --output none 2>/dev/null || \
  az postgres flexible-server firewall-rule create \
    -g "$RG" --name "$PG_SERVER" --rule-name AllowAllAzureServices \
    --start-ip-address 0.0.0.0 --end-ip-address 0.0.0.0 --output none

az grafana show -g "$RG" -n "$WORKSPACE" --output none 2>/dev/null || \
  az grafana create -g "$RG" -n "$WORKSPACE" -l "$LOC" --sku Standard --output none

# Idempotent data-source: delete then create (cheaper than diffing).
az grafana data-source delete -n "$WORKSPACE" --data-source sciencebeam-bench --output none 2>/dev/null || true
az grafana data-source create -n "$WORKSPACE" --definition @- <<JSON >/dev/null
{
  "name": "sciencebeam-bench",
  "type": "postgres",
  "url": "${PG_SERVER}.postgres.database.azure.com:5432",
  "database": "${PG_DB}",
  "user": "${PG_USER}",
  "access": "proxy",
  "jsonData": { "sslmode": "require", "postgresVersion": 1600, "timescaledb": false },
  "secureJsonData": { "password": "${PG_PW}" }
}
JSON

# Resolve the data-source uid and rewrite the dashboard JSON's ${DS_SCIENCEBEAM_BENCH}
# input placeholder to the live uid before publish.
DS_UID=$(az grafana data-source show -n "$WORKSPACE" --data-source sciencebeam-bench --query 'uid' -o tsv)
TMP_DASH=$(mktemp)
sed "s/\${DS_SCIENCEBEAM_BENCH}/${DS_UID}/g" "$DASHBOARD_JSON" > "$TMP_DASH"

az grafana dashboard create -g "$RG" -n "$WORKSPACE" \
  --title "ScienceBeam bench: commit comparison" \
  --definition @"$TMP_DASH" --overwrite true --output none

ENDPOINT=$(az grafana show -g "$RG" -n "$WORKSPACE" --query 'properties.endpoint' -o tsv)
echo
echo "Grafana endpoint: $ENDPOINT"
echo "Dashboard:        $ENDPOINT/d/sciencebeam-bench"
rm -f "$TMP_DASH"
