import os
import random
import discord
#import google.generativeai as genai
from dotenv import load_dotenv
import requests
import re
API_HOST = os.getenv("API_HOST")
API_URL = f"{API_HOST}/query"
#connecting the bot
from discord.ext import commands
#loading the token and guild
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
#genai.configure(api_key=GEMINI_API_KEY)
#model = genai.GenerativeModel('models/gemini-1.5-pro')

MAX_LEN = 1900 #for discord message limit
def looks_like_link(word):
    return (
        word.startswith("http://")
        or word.startswith("https://")
        or word.startswith("www.")
    )
async def send_safe(sender, text, MAX_LEN):
    send = sender.send
    current = ""
    lines = text.split("\n")
    for line in lines:
        words = line.split(" ")
        for w in words:
            if looks_like_link(w) and len(w) > MAX_LEN:
                # Send accumulated text first
                if current.strip():
                    await send(current.strip())
                    current = ""
                await send(w)
            else:
                if len(current) + len(w) + 1 > MAX_LEN:
                    await send(current.strip())
                    current = w + " "
                else:
                    current += w + " "
        current += "\n"
    # Send anything remaining
    if current.strip():
        await send(current.strip())

async def citationCreator(sender, citations):
    seen_ids = set()
    for cite in citations:
        cid = cite.get('id')
        if cid in seen_ids:
            continue  # skip duplicates
        seen_ids.add(cid)
        author = cite.get('author', 'Unknown author')
        title = cite.get('title', 'Untitled')
        url = cite.get('url','')
        snippet = cite.get('snippet','')
        cite_message = f"[{cid}] [{author} — **{title}**]({url})\n> {snippet}"
        await send_safe(sender, cite_message, MAX_LEN)
intents = discord.Intents.default()
intents.message_content = True  
intents.members = True

#removes default !help
bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)



@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord yay!')  

@bot.event
async def on_member_join(member):
    await member.create_dm()
    await member.dm_channel.send(
        f'Hello {member.name}, welcome to myTrueNorth.app, if you have any questions, refer to the info text channel!'
    )

#truenorth backend test
@bot.command(name='askTrueNorth')
async def ask_truenorth(ctx, *, question):
    """Ask a question to TrueNorth"""
    await ctx.send("Thinking...")

    payload = {"question": question, "chat_history": []}

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        answer = data.get("response", "No response found")
        citations = data.get("citations", [])

        full_message = f"**Q:** {question}\n**A:** {answer}"
        await send_safe(ctx, full_message, MAX_LEN)
        await citationCreator(ctx, citations)
        await ctx.message.add_reaction('✅')

    except requests.exceptions.RequestException as e:
        print(f"❌ Error contacting backend: {e}")
        await ctx.send("Sorry, I could not reach the TrueNorth API.")


#help
@bot.command(name='help')
async def helpme(ctx):
    help_text = "Here are my current commands:\n!askTrueNorth (Your Response Here!)"
    await ctx.send(help_text)
#raise exception for errors
@bot.command(name='raise-exception')
async def raise_exception(ctx):
    raise discord.DiscordException

#GEMINI ADDITION
@bot.command(name='geminiquestion')
async def ask_gemini(ctx, *, question):
    """Ask question to TrueNorth (rerouted from Gemini)"""
    await ctx.send("Thinking with TrueNorth...")

    payload = {"question": question, "chat_history": []}

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        answer = data.get("response", "No response found")
        citations = data.get("citations", [])

        full_message = f"**Q:** {question}\n**A:** {answer}"
        await send_safe(ctx, full_message, MAX_LEN)
        await citationCreator(ctx, citations)
        await ctx.message.add_reaction('✅')

    except requests.exceptions.RequestException as e:
        print(f"Error contacting TrueNorth backend: {e}")
        await ctx.send("Sorry, TrueNorth is not responding right now.")

        
@bot.tree.command(name="geminiquestion", description="Ask TrueNorth anything")
async def ask_slash(interaction: discord.Interaction, question: str):
    """Slash command rerouted to TrueNorth"""
    await interaction.response.defer()
    payload = {"question": question, "chat_history": []}

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        answer = data.get("response", "No response found")
        citations = data.get("citations", [])

        full_message = f"**Q:** {question}\n**A:** {answer}"
        await send_safe(interaction.followup, full_message, MAX_LEN)
        await citationCreator(interaction.followup, citations)

    except requests.exceptions.RequestException as e:
        print(f"Error contacting TrueNorth backend: {e}")
        await interaction.followup.send("Sorry, TrueNorth is not responding right now.")


@bot.event
async def on_error(event, *args, **kwargs):
    with open('err.log', 'a') as f:
        if event == 'on_message':
            f.write(f'Unhandled message: {args[0]}\n')
        else:
            raise

bot.run(TOKEN)

#run with python bot.py
