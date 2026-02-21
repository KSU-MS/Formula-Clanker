# Search-Guy
Scraping our discord (and soon OneNote) to make a universally searchable index for our team. 

Instructions:
1. Clone this repo.
2. make a venv (use uv)
3. insert your bot token into "BOT_TOKEN" in "scrapeDiscord.py", then run it
4. run "chunk.py" then "create_vectors.py". The latter will require significant resources. (took over an hour on a 64GB RAM and 16 core machine)
5. run "app.py" and in theory your search will work

# Troubleshooting
* make sure all the python libraries are installed
* you need uv installed, and a beefy machine for vector creation
* make sure your bot is set up correctly
* dm me on discord with any questions: @nick75704 in the discord