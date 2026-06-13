#!/usr/bin/env python3
"""QHDALabs Sentinel Diagnostic v6 - CDSE requires dataMask output"""
import os, sys, requests, json

CLIENT_ID     = os.environ.get("CDSE_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("CDSE_CLIENT_SECRET", "")
if not CLIENT_ID:
    print("ERROR: Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET"); sys.exit(1)

token = requests.post(
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
    data={"grant_type":"client_credentials","client_id":CLIENT_ID,"client_secret":CLIENT_SECRET}
).json()["access_token"]
print(f"Token OK, len={len(token)}")

# CDSE Statistical API forces dataMask output — must declare it explicitly
# and return it from evaluatePixel as second value
EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B03", "B08", "SCL", "dataMask"] }],
    output: [
      { id: "data",     bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1, sampleType: "UINT8"   }
    ]
  };
}
function evaluatePixel(s) {
  if (s.dataMask === 0) return { data: [NaN], dataMask: [0] };
  var scl = s.SCL;
  if (scl===3||scl===8||scl===9||scl===10||scl===11)
    return { data: [NaN], dataMask: [0] };
  var d = s.B03 + s.B08;
  var ndwi = (d === 0) ? 0.0 : (s.B03 - s.B08) / d;
  return { data: [ndwi], dataMask: [1] };
}"""

payload = {
    "input": {
        "bounds": {
            "bbox": [16.10, 51.00, 16.20, 51.10],
            "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}
        },
        "data": [{"type": "sentinel-2-l2a"}]
    },
    "aggregation": {
        "timeRange": {"from": "2026-05-01T00:00:00Z", "to": "2026-05-31T23:59:59Z"},
        "aggregationInterval": {"of": "P10D"},
        "evalscript": EVALSCRIPT,
        "resx": 0.001,
        "resy": 0.001
    }
}

resp = requests.post(
    "https://sh.dataspace.copernicus.eu/statistics/v1",
    json=payload,
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    },
    timeout=30
)
print(f"Status: {resp.status_code}")
print(f"Response:\n{resp.text[:4000]}")
