#!/usr/bin/env bash
# Provision the Postgres + Blob backing store for benchmark exports.
# Idempotent: re-running is safe; existing resources are reused.
#
# Prereqs: az login, an active subscription, and that subscription must have
# Microsoft.DBforPostgreSQL registered (the script registers it if missing).
#
# After this runs:
#   1) export BENCH_PG_DSN as printed at the end
#   2) psql "$BENCH_PG_DSN" -f benchmarks/sql/schema.sql
#   3) Add BENCH_PG_DSN as a GitHub Actions secret
#   4) Create an AZURE_CREDENTIALS service principal for Azure/postgresql@v1:
#        az ad sp create-for-rbac --name "$SERVER-ci" \
#          --role "Contributor" \
#          --scopes /subscriptions/<sub>/resourceGroups/$RG \
#          --json-auth
#      Save the JSON output as the AZURE_CREDENTIALS secret.
set -euo pipefail

RG="${RG:-sb-bench-rg}"
LOC="${LOC:-westeurope}"
SERVER="${SERVER:-sb-bench-pg}"
ADMIN="${ADMIN:-sbbench}"
DB="${DB:-sciencebeam_bench}"
SA="${SA:-sbbenchblob$RANDOM}"
CONTAINER="${CONTAINER:-runs}"

if [[ -z "${PG_PW:-}" ]]; then
  PG_PW="$(openssl rand -base64 24 | tr -d '/+=' | head -c 24)"
  echo "Generated PG_PW (record this): $PG_PW"
fi

az provider register --namespace Microsoft.DBforPostgreSQL --wait >/dev/null

az group create -n "$RG" -l "$LOC" --output none

az postgres flexible-server show -g "$RG" -n "$SERVER" --output none 2>/dev/null || \
  az postgres flexible-server create \
    --resource-group "$RG" --name "$SERVER" --location "$LOC" \
    --tier Burstable --sku-name Standard_B1ms \
    --storage-size 32 --version 16 \
    --admin-user "$ADMIN" --admin-password "$PG_PW" \
    --public-access None \
    --yes --output none

az postgres flexible-server db show \
    -g "$RG" -s "$SERVER" -d "$DB" --output none 2>/dev/null || \
  az postgres flexible-server db create \
    --resource-group "$RG" --server-name "$SERVER" --database-name "$DB" \
    --output none

az storage account show -n "$SA" -g "$RG" --output none 2>/dev/null || \
  az storage account create -n "$SA" -g "$RG" -l "$LOC" --sku Standard_LRS --output none

az storage container show --account-name "$SA" -n "$CONTAINER" --output none 2>/dev/null || \
  az storage container create --account-name "$SA" -n "$CONTAINER" --output none

cat <<EOF

---
Postgres server:  $SERVER.postgres.database.azure.com
Database:         $DB
Admin:            $ADMIN

DSN (export this and add to GitHub Actions secret BENCH_PG_DSN):

BENCH_PG_DSN="postgresql://$ADMIN:$PG_PW@$SERVER.postgres.database.azure.com:5432/$DB?sslmode=require"

Next steps:
  1. Apply schema:
       psql "\$BENCH_PG_DSN" -f benchmarks/sql/schema.sql
     (you'll need to add your local IP to the firewall first:
        MY_IP=\$(curl -s ifconfig.me)
        az postgres flexible-server firewall-rule create \\
          -g $RG --name $SERVER --rule-name local-dev \\
          --start-ip-address \$MY_IP --end-ip-address \$MY_IP)
  2. Backfill past CI runs:
       BENCH_PG_DSN="..." uv run python -m benchmarks.export --backfill
  3. Create Azure service principal for CI (Azure/postgresql@v1 needs it):
       az ad sp create-for-rbac --name "$SERVER-ci" \\
         --role Contributor \\
         --scopes \$(az group show -n $RG --query id -o tsv) \\
         --json-auth
     Save the JSON as the AZURE_CREDENTIALS GitHub Actions secret.
EOF
