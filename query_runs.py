import urllib.request
import json
import ssl

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
    print("No successful runs found.")
else:
    run_id = runs[0]["id"]
    print(f"Found run_id: {run_id}")
    arts_url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/artifacts"
    a_req = urllib.request.Request(arts_url)
    with urllib.request.urlopen(a_req, context=ctx) as a_resp:
        a_data = json.loads(a_resp.read().decode())
    arts = a_data.get("artifacts", [])
    print("Artifacts:")
    for a in arts:
        print(f"- {a['name']} ({a['size_in_bytes']} bytes)")
