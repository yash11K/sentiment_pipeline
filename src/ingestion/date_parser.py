from datetime import datetime, timedelta, date
import re

def parse_relative_date(relative_date: str, scrape_date: datetime = None) -> date:
    """Convert relative date strings to date objects"""
    if not scrape_date:
        scrape_date = datetime.now()
    
    relative_date = relative_date.lower().strip()
    
    # Strip "edited" prefix (e.g., "Edited 2 months ago" -> "2 months ago")
    relative_date = re.sub(r'^edited\s+', '', relative_date)
    
    # Patterns
    if 'today' in relative_date or 'just now' in relative_date:
        return scrape_date.date()
    
    if 'yesterday' in relative_date:
        return (scrape_date - timedelta(days=1)).date()
    
    # Extract number and unit
    match = re.search(r'(\d+)\s*(day|week|month|year)s?\s*ago', relative_date)
    if match:
        num = int(match.group(1))
        unit = match.group(2)
        
        if unit == 'day':
            return (scrape_date - timedelta(days=num)).date()
        elif unit == 'week':
            return (scrape_date - timedelta(weeks=num)).date()
        elif unit == 'month':
            return (scrape_date - timedelta(days=num * 30)).date()
        elif unit == 'year':
            return (scrape_date - timedelta(days=num * 365)).date()
    
    # Match "a week ago", "a month ago"
    match = re.search(r'a\s+(day|week|month|year)\s*ago', relative_date)
    if match:
        unit = match.group(1)
        if unit == 'day':
            return (scrape_date - timedelta(days=1)).date()
        elif unit == 'week':
            return (scrape_date - timedelta(weeks=1)).date()
        elif unit == 'month':
            return (scrape_date - timedelta(days=30)).date()
        elif unit == 'year':
            return (scrape_date - timedelta(days=365)).date()
    
    # Default to scrape date if can't parse
    return scrape_date.date()
