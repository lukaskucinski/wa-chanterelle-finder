"""
Upload COG files to Mapbox using the Uploads API.

This script bypasses the broken mapboxcli and uses the API directly.
Designed for frequent re-uploads during development.

Setup (one time):
    1. pip install boto3 python-dotenv
    2. Copy .env.example to .env
    3. Fill in MAPBOX_ACCESS_TOKEN and MAPBOX_USERNAME

Usage:
    python scripts/upload_to_mapbox.py           # Upload all files
    python scripts/upload_to_mapbox.py habitat   # Upload only habitat
    python scripts/upload_to_mapbox.py access    # Upload only access
"""

import os
import sys
import time
import argparse
import requests
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # dotenv not installed, rely on environment variables

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
MAPBOX_DIR = PROJECT_ROOT / "mapbox"
BASE_URL = "https://api.mapbox.com"

# Files to upload - maps short name to (filename, tileset_name)
UPLOAD_CONFIGS = {
    "habitat": ("habitat_suitability_cog.tif", "chanterelle-habitat"),
    "access": ("access_quality_cog.tif", "chanterelle-access"),
}


class ProgressCallback:
    """Callback for S3 upload progress."""
    def __init__(self, total_size):
        self.total_size = total_size
        self.uploaded = 0

    def __call__(self, bytes_transferred):
        self.uploaded += bytes_transferred
        pct = (self.uploaded / self.total_size) * 100
        print(f"\r    Progress: {pct:.1f}%", end="", flush=True)


def get_credentials(token: str, username: str) -> dict:
    """Get S3 credentials for upload."""
    url = f"{BASE_URL}/uploads/v1/{username}/credentials?access_token={token}"
    response = requests.post(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"  Error getting credentials: {response.status_code}")
        print(f"  {response.text}")
        return None


def upload_to_s3(file_path: Path, credentials: dict) -> bool:
    """Upload file to Mapbox's S3 staging bucket."""
    import boto3
    from botocore.config import Config

    s3 = boto3.client(
        's3',
        aws_access_key_id=credentials['accessKeyId'],
        aws_secret_access_key=credentials['secretAccessKey'],
        aws_session_token=credentials['sessionToken'],
        region_name='us-east-1',
        config=Config(signature_version='s3v4')
    )

    bucket = credentials['bucket']
    key = credentials['key']
    file_size = file_path.stat().st_size

    print(f"    Uploading {file_size / (1024*1024):.1f} MB to staging...")

    try:
        s3.upload_file(
            str(file_path),
            bucket,
            key,
            Callback=ProgressCallback(file_size)
        )
        print()  # New line after progress
        return True
    except Exception as e:
        print(f"\n    S3 upload error: {e}")
        return False


def create_upload(token: str, username: str, credentials: dict, tileset_name: str) -> str:
    """Create the upload job (overwrites existing tileset)."""
    url = f"{BASE_URL}/uploads/v1/{username}?access_token={token}"

    tileset_id = f"{username}.{tileset_name}"

    payload = {
        "url": credentials['url'],
        "tileset": tileset_id,
        "name": tileset_name,
    }

    response = requests.post(url, json=payload)

    if response.status_code in [200, 201]:
        return response.json().get('id')
    else:
        print(f"  Error creating upload: {response.status_code}")
        print(f"  {response.text}")
        return None


def wait_for_upload(token: str, username: str, upload_id: str, timeout: int = 600) -> bool:
    """Wait for upload to complete."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        url = f"{BASE_URL}/uploads/v1/{username}/{upload_id}?access_token={token}"
        response = requests.get(url)

        if response.status_code != 200:
            print("    Error checking status")
            return False

        status = response.json()
        complete = status.get('complete', False)
        error = status.get('error')
        progress = status.get('progress', 0)

        if error:
            print(f"\n    Upload error: {error}")
            return False

        if complete:
            print(f"\r    Processing: 100% - Complete!")
            return True

        print(f"\r    Processing: {progress*100:.0f}%", end="", flush=True)
        time.sleep(3)

    print("\n    Upload timed out")
    return False


def upload_file(file_path: Path, tileset_name: str, token: str, username: str) -> bool:
    """Upload a single file to Mapbox."""
    tileset_id = f"{username}.{tileset_name}"
    print(f"\n  {file_path.name} -> {tileset_id}")

    # Step 1: Get credentials
    print("    Getting credentials...")
    credentials = get_credentials(token, username)
    if not credentials:
        return False

    # Step 2: Upload to S3
    if not upload_to_s3(file_path, credentials):
        return False

    # Step 3: Create upload job
    print("    Creating tileset...")
    upload_id = create_upload(token, username, credentials, tileset_name)
    if not upload_id:
        return False

    # Step 4: Wait for completion
    return wait_for_upload(token, username, upload_id)


def main():
    parser = argparse.ArgumentParser(
        description="Upload COGs to Mapbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/upload_to_mapbox.py           # Upload all
  python scripts/upload_to_mapbox.py habitat   # Upload habitat only
  python scripts/upload_to_mapbox.py access    # Upload access only
        """
    )
    parser.add_argument(
        "layers",
        nargs="*",
        choices=list(UPLOAD_CONFIGS.keys()) + [[]],
        help="Specific layers to upload (default: all)"
    )
    parser.add_argument("--token", help="Mapbox access token (or set MAPBOX_ACCESS_TOKEN)")
    parser.add_argument("--username", help="Mapbox username (or set MAPBOX_USERNAME)")
    args = parser.parse_args()

    # Get credentials
    token = args.token or os.environ.get("MAPBOX_ACCESS_TOKEN")
    username = args.username or os.environ.get("MAPBOX_USERNAME")

    if not token:
        print("ERROR: No Mapbox access token")
        print("\nSetup instructions:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your MAPBOX_ACCESS_TOKEN and MAPBOX_USERNAME")
        print("\nOr set environment variable:")
        print("  $env:MAPBOX_ACCESS_TOKEN = 'sk.xxx...'")
        return 1

    if not username:
        print("ERROR: No Mapbox username")
        print("\nAdd MAPBOX_USERNAME to .env file or set:")
        print("  $env:MAPBOX_USERNAME = 'yourusername'")
        return 1

    # Check for boto3
    try:
        import boto3
    except ImportError:
        print("ERROR: boto3 required")
        print("Install with: pip install boto3")
        return 1

    # Determine which files to upload
    layers_to_upload = args.layers if args.layers else list(UPLOAD_CONFIGS.keys())

    print("=" * 60)
    print("Mapbox Tileset Upload")
    print("=" * 60)
    print(f"\nUsername: {username}")
    print(f"Layers: {', '.join(layers_to_upload)}")

    # Upload files
    success_count = 0
    for layer in layers_to_upload:
        filename, tileset_name = UPLOAD_CONFIGS[layer]
        file_path = MAPBOX_DIR / filename

        if not file_path.exists():
            print(f"\n  WARNING: {filename} not found")
            print(f"    Run: python scripts/08_export_for_mapbox.py")
            continue

        if upload_file(file_path, tileset_name, token, username):
            success_count += 1

    # Summary
    print("\n" + "=" * 60)
    if success_count == len(layers_to_upload):
        print(f"SUCCESS: {success_count}/{len(layers_to_upload)} tilesets uploaded")
    else:
        print(f"PARTIAL: {success_count}/{len(layers_to_upload)} tilesets uploaded")
    print("=" * 60)

    print(f"\nView tilesets: https://studio.mapbox.com/tilesets/")
    print(f"\nTileset IDs:")
    for layer in layers_to_upload:
        _, tileset_name = UPLOAD_CONFIGS[layer]
        print(f"  {username}.{tileset_name}")

    return 0 if success_count == len(layers_to_upload) else 1


if __name__ == "__main__":
    sys.exit(main())
