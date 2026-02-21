import discord
import asyncio
import os
import json
from datetime import timezone

TOKEN = "BOT_KEY"
EXPORT_DIR = "discord_exports"
STATE_FILE = "export_state.json"

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True

client = discord.Client(intents=intents)


def sanitize_filename(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in (" ", "-", "_")).rstrip()


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR)

    state = load_state()

    for guild in client.guilds:
        print(f"\nChecking server: {guild.name}")

        guild_dir = os.path.join(EXPORT_DIR, sanitize_filename(guild.name))
        os.makedirs(guild_dir, exist_ok=True)

        for channel in guild.text_channels:
            perms = channel.permissions_for(guild.me)
            if not perms.read_message_history:
                print(f"Skipping {channel.name} (no permission)")
                continue

            channel_key = f"{guild.id}-{channel.id}"
            last_id = state.get(channel_key)

            filename = os.path.join(
                guild_dir,
                f"{sanitize_filename(channel.name)}.txt"
            )
            
            metadata_filename = os.path.join(
                guild_dir,
                f"{sanitize_filename(channel.name)}_metadata.json"
            )

            print(f"Updating #{channel.name}")

            new_last_id = last_id
            message_count = 0
            
            # Load existing metadata
            metadata_map = {}
            if os.path.exists(metadata_filename):
                try:
                    with open(metadata_filename, "r", encoding="utf-8") as mf:
                        metadata_map = json.load(mf)
                except:
                    metadata_map = {}

            # Open file in append mode
            with open(filename, "a", encoding="utf-8") as f:
                history_kwargs = {
                    "limit": None,
                    "oldest_first": True
                }

                if last_id:
                    history_kwargs["after"] = discord.Object(id=int(last_id))

                # Get message count in existing file
                line_count = sum(1 for _ in open(filename, "rb")) if os.path.exists(filename) else 0

                async for message in channel.history(**history_kwargs):
                    timestamp = message.created_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                    author = f"{message.author.name}#{message.author.discriminator}"
                    content = message.content.replace("\n", " ")

                    f.write(f"[{timestamp}] {author}: {content}\n")
                    
                    # Store metadata for this line
                    line_count += 1
                    metadata_map[str(line_count)] = {
                        "guild_id": guild.id,
                        "channel_id": channel.id,
                        "message_id": message.id
                    }

                    if message.attachments:
                        for attachment in message.attachments:
                            f.write(f"    [Attachment] {attachment.url}\n")
                            line_count += 1

                    if message.embeds:
                        f.write("    [Embed]\n")
                        line_count += 1

                    new_last_id = message.id
                    message_count += 1

            if message_count > 0:
                state[channel_key] = str(new_last_id)
                save_state(state)
                # Save metadata
                with open(metadata_filename, "w", encoding="utf-8") as mf:
                    json.dump(metadata_map, mf, indent=2)
                print(f"  Added {message_count} new messages")
            else:
                print("  No new messages")

            await asyncio.sleep(0.5)  # polite delay

    print("\nIncremental update complete.")
    await client.close()


client.run(TOKEN)
