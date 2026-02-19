import sys; sys.path.insert(0, 'src')
from dotenv import load_dotenv; load_dotenv()
from storage.db import Database
from sqlalchemy import text

db = Database()
with db.get_session() as s:
    r = s.execute(text(
        "SELECT location_id, brand, is_competitor, COUNT(*) as cnt "
        "FROM reviews GROUP BY location_id, brand, is_competitor "
        "ORDER BY location_id, brand"
    ))
    for row in r.fetchall():
        print(f"{row.location_id}  brand={row.brand}  competitor={row.is_competitor}  count={row.cnt}")
