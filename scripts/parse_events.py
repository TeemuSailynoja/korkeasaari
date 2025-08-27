import csv
import re
from datetime import datetime


def translate_event_name(finnish_name):
    """Translate Finnish event names to English and create shortened versions"""

    # Remove any years from the translated name
    finnish_name = re.sub(r"\s*\d{4}", "", finnish_name)

    # Remove any dates (like "13.9.", "11.9.", etc.) from the translated name
    finnish_name = re.sub(r"\s*\d+\.\d+\.?", "", finnish_name)
    translations = {
        # Free days - all categorized as "Free Entry"
        "Korkeasaaren joulukuun ilmaispäivä": "Free Entry",
        "Korkeasaaren marraskuun ilmaispäivä": "Free Entry",
        "Korkeasaaren lokakuun ilmaispäivä": "Free Entry",
        "Korkeasaaren kevätkauden viimeinen ilmaispäivä": "Free Entry",
        "Korkeasaaren ilmaispäivä": "Free Entry",
        "Marraskuun ilmaispäivä": "Free Entry",
        "Lokakuun ilmaispäivä Korkeasaaressa": "Free Entry",
        "Korkeasaaripäivä, ilmainen sisäänpääsy": "Free Entry",
        # Seasonal events
        "Trooppiset viikonloput Korkeasaaressa": "Tropical Weekends",
        "Korkeasaaren syysloma": "Autumn Holiday",
        "Syysloma Korkeasaaressa": "Autumn Holiday",
        "Syysloma eläinten saarella": "Autumn Holiday",
        "Talviloma Korkeasaaressa": "Winter Holiday",
        "Pääsiäissaari Korkeasaaressa": "Easter Island",
        "Pääsiäissaari 2022": "Easter Island",
        "Pääsiäissaari": "Easter Island",
        # Special events
        "Kissojen Yö": "Cats Night",
        "Kissojen yö": "Cats Night",
        "Kissojen Yö K18": "Cats Night",
        "Kissojen Illat": "Cats Evenings",
        "Lux Korkeasaari": "Lux",
        "Kohtaa pelkosi!": "Face Your Fears!",
        "Viettelysten Ilta": "Temptations Evening",
        "Viettelysten ilta": "Temptations Evening",
        "Amur-illallinen": "Amur Dinner",
        "Halloween Korkeasaaressa": "Halloween",
        "Korkeasaari-päivä": "Zoo Day",
        # Animal days
        "Apinapäivä Korkeasaaressa": "Monkey Day",
        "Apinapäivä": "Monkey Day",
        "Käärmepäivä": "Snake Day",
        "Makakipäivä Korkeasaaren Apinalinnalla": "Macaque Day",
        "Tiikeripäivä": "Tiger Day",
        "Pikkupandapäivä": "Little Panda Day",
        # Other events
        "Koe merellinen Helsinki": "Experience Maritime Helsinki",
        "Helsinki Biennaalin perheviikonloppu": "Helsinki Biennial Family Weekend",
        "Suomen luonnon päivä Korkeasaaressa": "Finnish Nature Day",
        "Itämeripäivä Korkeasaaressa": "Baltic Sea Day",
        "BSAG:n ja Korkeasaaren Itämeripäivä": "Baltic Sea Day",
        "Helsinki-päivän videolähetykset Korkeasaaresta": "Helsinki Day Video Broadcasts",
        "Riikinkukkojen ulosmarssi Fb-live": "Peacock Parade",
        "Talviluonnonpäivä": "Winter Nature Day",
        "Lumileopardin mailla -valokuvanäyttely": "Photo Exhibition",
        "Fair Saturday Korkeasaaressa": "Fair Saturday",
        "Zoohackathon Finland": "Zoohackathon",
        "Photo Walks at Korkeasaari Zoo": "Photo Walks",
        "Lähimatkailupäivät, 2 yhden hinnalla Korkeasaareen": "Entry Discount",
        "5 euron arki-illat elokuussa Korkeasaaressa": "Entry Discount",
        "Kepparikisat Korkeasaaressa talvilomalla": "Hobbyhorse Races",
    }

    # Try exact match first
    if finnish_name in translations:
        return translations[finnish_name]

    # Handle patterns
    if "ilmaispäivä" in finnish_name:
        return "Free Entry"

    # Handle year patterns - remove years from event names
    if re.search(r"\d{4}", finnish_name):
        base_name = re.sub(r"\s*\d{4}", "", finnish_name)
        if base_name in translations:
            return translations[base_name]

    # If no translation found, return a simplified version
    print(finnish_name)
    translated_name = finnish_name

    # Replace any remaining "Korkeasaari" with "Zoo"
    translated_name = translated_name.replace("Korkeasaari", "Zoo")

    return translated_name


def parse_date(date_str):
    """Parse Finnish date format to standard format"""
    # Remove day abbreviation and clean up
    date_str = re.sub(r"^[a-z]{2},\s*", "", date_str.strip())

    # Parse Finnish date format (DD.MM.YYYY)
    try:
        date_obj = datetime.strptime(date_str, "%d.%m.%Y")
        return date_obj.strftime("%Y-%m-%d")
    except ValueError:
        return date_str


def parse_events_file(filename):
    """Parse the events file and return list of events"""
    events = []

    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check if this is a date line (starts with day abbreviation)
        if re.match(r"^[a-z]{2},\s*\d+\.\d+\.\d+", line):
            date = parse_date(line)

            # Next line should be the event name
            if i + 1 < len(lines):
                event_name = lines[i + 1].strip()
                translated_name = translate_event_name(event_name)

                events.append({"date": date, "name": translated_name})

            # Skip the next 3 lines (name, location, description)
            i += 4
        else:
            i += 1

    return events


def main():
    # Parse events (this includes translation)
    events = parse_events_file("data/raw/events.txt")

    # Convert to DataFrame for easier analysis
    import pandas as pd

    df = pd.DataFrame(events)

    # Get value counts using pandas
    event_counts = df["name"].value_counts()

    # Find events that occur less than 5 times
    rare_events = event_counts[event_counts < 5].index.tolist()

    # Categorize events that occur less than 5 times as "Other Event"
    for event in events:
        if event["name"] in rare_events:
            event["name"] = "Other Event"

    # After further study, I decided to remove this event as it was fully contained within the winter holiday season.

    events = [event for event in events if event["name"] != "Hobbyhorse Races"]
    # Write to CSV
    with open(
        "data/clean/all_events.csv", "w", newline="", encoding="utf-8"
    ) as csvfile:
        fieldnames = ["date", "name"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for event in events:
            writer.writerow(event)

    print(f"Successfully parsed {len(events)} events to data/clean/all_events.csv")
    print("\nEvents categorized as 'Other Event' (occurring less than 5 times):")
    for event_name in sorted(rare_events):
        count = event_counts[event_name]
        print(f"  - {event_name} ({count} occurrences)")

    print("\nMajor event categories (5+ occurrences):")
    major_events = event_counts[event_counts >= 5]
    for event_name, count in major_events.items():
        print(f"  - {event_name} ({count} occurrences)")


if __name__ == "__main__":
    main()
