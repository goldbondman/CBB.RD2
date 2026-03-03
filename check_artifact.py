import urllib.request
import json
import ssl
import sys

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

repo = "goldbondman/CBB.RD2"
url = f"https://api.github.com/repos/{repo}/actions/workflows/cbb_predictions_rolling.yml/runs?status=success&per_page=1"

req = urllib.request.Request(url)
with urllib.request.urlopen(req, context=ctx) as response:
    data = json.loads(response.read().decode())

runs = data.get("workflow_runs", [])
if not runs:
    sys.exit()

run_id = runs[0]["id"]
arts_url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/artifacts"
a_req = urllib.request.Request(arts_url)
with urllib.request.urlopen(a_req, context=ctx) as a_resp:
    a_data = json.loads(a_resp.read().decode())

for a in a_data.get("artifacts", []):
    if a["name"] == "INFRA-predictions-rolling":
        print(f"Artifact {a['name']} matches.")
        # We can't easily download the zip without a token because artifact download endpoint requires authentication
        print("Note: We can't download without token. But we can deduce path nesting.")
