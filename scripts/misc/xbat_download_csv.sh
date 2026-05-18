#!/bin/bash

# ==== CONFIGURATION ====
API_BASE="https://demo.xbat.dev"
TOKEN_URL="$API_BASE/oauth/token"
VALIDATE_URL="$API_BASE/api/v1/current_user"
TOKEN_FILE=".env.xbat"

USERNAME="demo"
PASSWORD="demo"
CLIENT_ID="demo"

# ==== CHANGE ONLY THESE PARAMETERS ==== 
# 248750
JOB_ID="249755"     # <-- Must not be empty
GROUP=""            # <-- If empty will download all groups and metrics at once (job-level only)
METRIC=""           # <-- Must be empty if GROUP is empty
LEVEL="job"         # <-- Leaving this empty will default to 'job' level
NODE=""             # <-- Only required when level is 'node'
# ==========================


# ==== VALIDATION ====
if [[ -z "$JOB_ID" ]]; then
  echo "[!] Error: JOB_ID must not be empty."
  exit 1
fi

if [[ -z "$GROUP" && -n "$METRIC" ]]; then
  echo "[!] Error: METRIC must be empty if GROUP is empty."
  exit 1
fi

if [[ -z "$LEVEL" ]]; then
  echo "[*] Warning: LEVEL not set, defaulting to 'job'."
  LEVEL="job"
fi

if [[ "$LEVEL" == "node" && -z "$NODE" ]]; then
  echo "[!] Error: NODE must be set when LEVEL is 'node'."
  exit 1
fi
# =====================


# ==== BUILD OUTPUT FILENAME ====
LEVEL_PART="${LEVEL:-job}"

if [[ -n "$GROUP" ]]; then
  # Group is set, include both group and metric
  GROUP_PART="$GROUP"
  METRIC_PART="${METRIC:-all}"
  if [[ -n "$NODE" ]]; then
    OUTPUT_FILE="${JOB_ID}_${GROUP_PART}_${METRIC_PART}_${LEVEL_PART}_${NODE}.csv"
  else
    OUTPUT_FILE="${JOB_ID}_${GROUP_PART}_${METRIC_PART}_${LEVEL_PART}.csv"
  fi
else
  # Group not set, omit metric
  if [[ -n "$NODE" ]]; then
    OUTPUT_FILE="${JOB_ID}_all_${LEVEL_PART}_${NODE}.csv"
  else
    OUTPUT_FILE="${JOB_ID}_all_${LEVEL_PART}.csv"
  fi
fi
# ============================


# ==== TOKEN MANAGEMENT ====

load_token() {
  if [[ -f "$TOKEN_FILE" ]]; then
    source "$TOKEN_FILE"
  fi
}

save_token() {
  echo "ACCESS_TOKEN=$ACCESS_TOKEN" > "$TOKEN_FILE"
}

validate_token() {
  local status
  status=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $ACCESS_TOKEN" "$VALIDATE_URL")
  [[ "$status" == "200" ]]
}

request_new_token() {
  echo "[*] Requesting new access token..."
  TOKEN_RESPONSE=$(curl -s -X POST "$TOKEN_URL" \
    -d "grant_type=password" \
    -d "username=$USERNAME" \
    -d "password=$PASSWORD" \
    -d "client_id=$CLIENT_ID")

  ACCESS_TOKEN=$(echo "$TOKEN_RESPONSE" | grep -oP '"access_token"\s*:\s*"\K[^"]+')

  if [[ -z "$ACCESS_TOKEN" ]]; then
    echo "[!] Failed to get access token. Response:"
    echo "$TOKEN_RESPONSE"
    exit 1
  fi

  echo "[+] New token acquired."
  save_token
}

# Load and validate token
load_token
if [[ -z "$ACCESS_TOKEN" ]] || ! validate_token; then
  request_new_token
else
  echo "[+] Using cached access token."
fi
# ============================


# ==== BUILD QUERY ====
QUERY_STRING=""
[[ -n "$GROUP" ]]  && QUERY_STRING+="group=$GROUP"
[[ -n "$METRIC" ]] && QUERY_STRING+="${QUERY_STRING:+&}metric=$METRIC"
[[ -n "$LEVEL" ]]  && QUERY_STRING+="${QUERY_STRING:+&}level=$LEVEL"
[[ -n "$NODE" ]]   && QUERY_STRING+="${QUERY_STRING:+&}node=$NODE"

FULL_URL="$API_BASE/api/v1/measurements/$JOB_ID/csv"
[[ -n "$QUERY_STRING" ]] && FULL_URL+="?$QUERY_STRING"
# =======================


# ==== DOWNLOAD CSV ====
echo "[*] Downloading from: $FULL_URL"
TEMP_FILE="$OUTPUT_FILE.tmp"

RESPONSE=$(curl -s -w "%{http_code}" -X GET "$FULL_URL" \
  -H "accept: text/csv" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -o "$TEMP_FILE")

HTTP_STATUS="${RESPONSE: -3}"

if [[ "$HTTP_STATUS" == "404" ]]; then
  echo "[!] Error: Job ID '$JOB_ID' or combination of Job ID, GROUP, and METRIC not found on the server."
  rm -f "$TEMP_FILE"
  exit 1
elif [[ "$HTTP_STATUS" == "200" ]]; then
  mv "$TEMP_FILE" "$OUTPUT_FILE"
  ABS_PATH=$(realpath "$OUTPUT_FILE")
  echo "[+] CSV downloaded successfully."
  echo "[→] File saved at: $ABS_PATH"
else
  echo "[!] Download failed. HTTP status: $HTTP_STATUS"
  rm -f "$TEMP_FILE"
  exit 1
fi
# ========================
