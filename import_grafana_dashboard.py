"""
Import Grafana Dashboard via API
"""
import json

import requests

# Grafana configuration
GRAFANA_URL = "http://localhost:3000"
GRAFANA_USER = "admin"
GRAFANA_PASSWORD = "admin"


def import_dashboard():
    """Import Rice Disease API dashboard to Grafana"""

    # Read dashboard JSON
    with open("monitoring/grafana/dashboards/rice-disease-api.json", "r") as f:
        dashboard_data = json.load(f)

    # Get Prometheus datasource UID
    print("🔍 Getting Prometheus datasource...")
    datasources_response = requests.get(
        f"{GRAFANA_URL}/api/datasources", auth=(GRAFANA_USER, GRAFANA_PASSWORD)
    )

    if datasources_response.status_code != 200:
        print(f"❌ Failed to get datasources: {datasources_response.text}")
        return False

    datasources = datasources_response.json()
    prometheus_ds = None

    for ds in datasources:
        if ds.get("type") == "prometheus":
            prometheus_ds = ds
            break

    if not prometheus_ds:
        print("❌ Prometheus datasource not found!")
        print("📝 Please add Prometheus datasource first:")
        print("   Settings → Data sources → Add Prometheus")
        print("   URL: http://prometheus:9090")
        return False

    print(f"✅ Found Prometheus datasource: {prometheus_ds['name']} (UID: {prometheus_ds['uid']})")

    # Add datasource to all panels
    if "dashboard" in dashboard_data:
        dashboard = dashboard_data["dashboard"]
    else:
        dashboard = dashboard_data

    for panel in dashboard.get("panels", []):
        if "targets" in panel:
            for target in panel["targets"]:
                target["datasource"] = {"type": "prometheus", "uid": prometheus_ds["uid"]}

    # Prepare import payload
    import_payload = {"dashboard": dashboard, "overwrite": True, "message": "Imported via API"}

    # Import dashboard
    print("\n📤 Importing dashboard...")
    import_response = requests.post(
        f"{GRAFANA_URL}/api/dashboards/db",
        auth=(GRAFANA_USER, GRAFANA_PASSWORD),
        headers={"Content-Type": "application/json"},
        json=import_payload,
    )

    if import_response.status_code in [200, 201]:
        result = import_response.json()
        dashboard_url = f"{GRAFANA_URL}{result.get('url', '')}"
        print(f"\n✅ Dashboard imported successfully!")
        print(f"🔗 Dashboard URL: {dashboard_url}")
        print(f"📊 Dashboard ID: {result.get('id')}")
        print(f"📝 Dashboard UID: {result.get('uid')}")
        print("\n🎯 Next steps:")
        print("   1. Open the dashboard URL above")
        print("   2. Set auto-refresh to 5s (top-right corner)")
        print("   3. Generate traffic via Swagger UI: http://localhost:8000/docs")
        print("   4. Watch metrics update in real-time!")
        return True
    else:
        print(f"\n❌ Failed to import dashboard!")
        print(f"Status code: {import_response.status_code}")
        print(f"Response: {import_response.text}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("🎨 Grafana Dashboard Importer")
    print("=" * 60)

    success = import_dashboard()

    if success:
        print("\n" + "=" * 60)
        print("✅ Import completed successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Import failed. Please check the errors above.")
        print("=" * 60)
