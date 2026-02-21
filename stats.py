import json
import re
from pathlib import Path
from collections import Counter
import pandas as pd

# ========= CONFIG =========
DATA_FOLDER = "../discord_jsons"
OUTPUT_FILE = "user_message_counts.csv"
# ===========================


def extract_username(content):
    """
    Extract username from:
    "[2024-01-19 04:02:06 UTC] mathbrook#0: 👀 square"
    """
    try:
        after_bracket = content.split("] ", 1)[1]
        username = after_bracket.split(":", 1)[0]
        return username.strip()
    except Exception:
        return "UNKNOWN"


def load_all_messages(folder):
    user_message_count = Counter()

    for path in Path(folder).rglob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

            for block in data:
                for msg in block.get("messages", []):
                    content = msg.get("content")
                    if not content:
                        continue

                    username = extract_username(content)
                    user_message_count[username] += 1

    return user_message_count


def export_csv(counter, output_file):
    df = pd.DataFrame(counter.items(), columns=["Username", "Message Count"])
    df = df.sort_values(by="Message Count", ascending=False)
    df.to_csv(output_file, index=False)


def main():
    print("Processing JSON files...")
    user_counts = load_all_messages(DATA_FOLDER)

    print(f"Found {sum(user_counts.values())} total messages.")
    print("Exporting CSV...")

    export_csv(user_counts, OUTPUT_FILE)

    print(f"Done. File saved as: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
