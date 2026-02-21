import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import emoji

# ========= CONFIG =========
DATA_FOLDER = "../discord_jsons"
OUTPUT_FILE = "user_message_counts.csv"
EMOJI_OUTPUT_FILE = "emoji_statistics.csv"
CHANNEL_OUTPUT_FILE = "channel_top_posters.csv"
# ===========================


def extract_username(content):
    """
    Extract username from:
    "[<date> <time> UTC] <username>: <message>"
    """
    try:
        after_bracket = content.split("] ", 1)[1]
        username = after_bracket.split(":", 1)[0]
        return username.strip()
    except Exception:
        return "UNKNOWN"


def extract_emojis(text):
    """Extract all emojis from text"""
    return emoji.emoji_list(text)


def extract_channel_name(path):
    """Extract channel name from file path"""
    # Assuming file names follow pattern like "channel_name.json" or "channel_name_123.json"
    filename = Path(path).name
    # Remove .json extension
    channel_name = filename.replace(".json", "")
    return channel_name


def load_all_messages(folder):
    user_message_count = Counter()
    user_emoji_count = Counter()
    total_emoji_count = Counter()
    message_lengths = []
    total_messages = 0
    channel_user_counts = defaultdict(Counter)
    
    for path in Path(folder).rglob("*.json"):
        channel_name = extract_channel_name(path)
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

            for block in data:
                for msg in block.get("messages", []):
                    content = msg.get("content")
                    if not content:
                        continue

                    username = extract_username(content)
                    user_message_count[username] += 1
                    channel_user_counts[channel_name][username] += 1
                    total_messages += 1
                    
                    # Track message length
                    message_lengths.append(len(content))
                    
                    # Extract and count emojis
                    emojis = extract_emojis(content)
                    for emoji_data in emojis:
                        emoji_char = emoji_data['emoji']
                        total_emoji_count[emoji_char] += 1
                        user_emoji_count[(username, emoji_char)] += 1

    return user_message_count, total_emoji_count, user_emoji_count, message_lengths, total_messages, channel_user_counts


def export_user_stats(counter, output_file):
    df = pd.DataFrame(counter.items(), columns=["Username", "Message Count"])
    df = df.sort_values(by="Message Count", ascending=False)
    df.to_csv(output_file, index=False)


def export_emoji_stats(total_emoji_count, user_emoji_count, output_file):
    # Top emojis
    top_emojis = total_emoji_count.most_common(50)
    emoji_df = pd.DataFrame(top_emojis, columns=["Emoji", "Count"])
    emoji_df.to_csv(output_file, index=False)
    
    # User emoji stats
    user_emoji_df = pd.DataFrame([
        {"Username": user, "Emoji": emoji, "Count": count} 
        for (user, emoji), count in user_emoji_count.most_common()
    ])
    user_emoji_df.to_csv("user_emoji_stats.csv", index=False)


def export_channel_top_posters(channel_user_counts, output_file):
    """Export the top poster for each channel"""
    channel_top_posters = []
    
    for channel, user_counts in channel_user_counts.items():
        if user_counts:
            top_poster = user_counts.most_common(1)[0][0]
            top_poster_count = user_counts.most_common(1)[0][1]
            channel_top_posters.append({
                "Channel": channel,
                "Top Poster": top_poster,
                "Message Count": top_poster_count
            })
    
    # Sort by message count descending
    channel_top_posters.sort(key=lambda x: x["Message Count"], reverse=True)
    df = pd.DataFrame(channel_top_posters)
    df.to_csv(output_file, index=False)


def export_statistics(user_message_count, total_emoji_count, message_lengths, total_messages, channel_user_counts):
    # Calculate statistics
    avg_message_length = sum(message_lengths) / len(message_lengths) if message_lengths else 0
    total_unique_users = len(user_message_count)
    total_unique_emojis = len(total_emoji_count)
    total_channels = len(channel_user_counts)
    
    stats = {
        "Total Messages": [total_messages],
        "Total Unique Users": [total_unique_users],
        "Total Unique Emojis": [total_unique_emojis],
        "Total Channels": [total_channels],
        "Average Message Length": [round(avg_message_length, 2)]
    }
    
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv("statistics_summary.csv", index=False)
    
    # Top 10 users by message count
    top_users = user_message_count.most_common(10)
    top_users_df = pd.DataFrame(top_users, columns=["Username", "Message Count"])
    top_users_df.to_csv("top_users.csv", index=False)
    
    # Top 20 emojis
    top_emojis = total_emoji_count.most_common(20)
    top_emojis_df = pd.DataFrame(top_emojis, columns=["Emoji", "Count"])
    top_emojis_df.to_csv("top_emojis.csv", index=False)


def main():
    print("Processing JSON files...")
    user_counts, total_emojis, user_emojis, message_lengths, total_messages, channel_user_counts = load_all_messages(DATA_FOLDER)

    print(f"Found {total_messages} total messages.")
    print("Exporting CSV files...")

    export_user_stats(user_counts, OUTPUT_FILE)
    export_emoji_stats(total_emojis, user_emojis, EMOJI_OUTPUT_FILE)
    export_channel_top_posters(channel_user_counts, CHANNEL_OUTPUT_FILE)
    export_statistics(user_counts, total_emojis, message_lengths, total_messages, channel_user_counts)

    print(f"Done. Files saved as:")
    print(f"- {OUTPUT_FILE}")
    print(f"- {EMOJI_OUTPUT_FILE}")
    print(f"- {CHANNEL_OUTPUT_FILE}")
    print(f"- statistics_summary.csv")
    print(f"- top_users.csv")
    print(f"- top_emojis.csv")
    print(f"- user_emoji_stats.csv")


if __name__ == "__main__":
    main()