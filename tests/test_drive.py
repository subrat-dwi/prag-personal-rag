import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ingestion.drive_sync import get_drive_service, list_drive_files

service = get_drive_service()
files = list_drive_files(service)

if files:
    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  {f['name']} — {f['md5Checksum']}")
else:
    print("No supported files found. Add some files to your Drive folder and retry.")