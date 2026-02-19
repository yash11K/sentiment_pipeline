"""
Fast brand backfill — sets all NULL-brand reviews to 'avis' (is_competitor=false).
The original non-branded files were all avis location scrapes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from storage.db import Database
from sqlalchemy import text

db = Database()

with db.get_session() as session:
    result = session.execute(text("""
        UPDATE reviews
        SET brand = 'avis', is_competitor = false
        WHERE brand IS NULL
    """))
    session.commit()
    print(f"Updated {result.rowcount} reviews -> brand='avis', is_competitor=false")

print("Done!")
