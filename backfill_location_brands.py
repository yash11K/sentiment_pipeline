"""
One-time script to backfill the brands JSON column on the locations table.
Adds avis (ours) and hertz (competitor) to all existing locations.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dotenv import load_dotenv
load_dotenv()

import json
from storage.models import Location, create_db_engine, get_database_url
from sqlalchemy.orm import sessionmaker

engine = create_db_engine()
Session = sessionmaker(bind=engine)

brands = json.dumps([
    {"brand": "avis", "is_competitor": False},
    {"brand": "hertz", "is_competitor": True}
])

with Session() as session:
    locations = session.query(Location).all()
    for loc in locations:
        loc.brands = brands
        print(f"  Updated {loc.location_id} ({loc.name})")
    session.commit()
    print(f"\nDone — updated {len(locations)} locations with avis + hertz.")
