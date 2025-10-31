import os
import random
import discord
#import google.generativeai as genai
from dotenv import load_dotenv
from Knowledge import vector_store
from Knowledge import get_embedding_model, vector_store_dir
from langchain_community.vectorstores import FAISS
import requests

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
        f'Hiii {member.name}, welcome to my server!!!!!!!!!!!!!!'
    )




#silly quotes, references from toontown, 
#https://www.mmocentralforums.com/forums/showthread.php?t=234847

@bot.command(name='joke')
async def joke(ctx):
    sillyquote = [
        "What travels around the world but stays in a corner? ||A stamp.||",
        "Why was the math book unhappy? ||It had too many problems.||",
        "What did the doctor say to the sick orange? ||Are you peeling well?||",
        "What did the peanut say to the elephant? ||Nothing peanuts can't talk||",
        "Why do mummies make excellent spies? ||Because they're good at keeping things under wraps||",
        "What do you call a blind dinosaur? ||An I-Dont-Think-He-Saurus!||",
        "What kind of mistakes do spooks make? ||Boo boos||",
        "What do you call a chicken at the north pole? ||Lost||",
        "What did the cashier say to the register? ||Im counting on you.||"
    ]
    response = random.choice(sillyquote)
    await ctx.send(response)
    await ctx.message.add_reaction('✅')
#truenorth backend test
@bot.command(name='askTrueNorth')
async def ask_truenorth(ctx, *, question):
    global vector_store
    if vector_store is None:
        # Load vector store once
        embedding_model = get_embedding_model()
        try:
            vector_store = FAISS.load_local(
                f"{vector_store_dir}/truenorth_kb_vectorstore",
                embedding_model,
                allow_dangerous_deserialization=True
            )
            print("✅ Loaded vector store for bot")
        except Exception as e:
            print(f"❌ Failed to load vector store: {e}")
            await ctx.send("⚠️ Vector store not available. Please build it first.")
            return

    await ctx.send("Thinking...")

    # --- query local knowledge base ---
    docs = vector_store.similarity_search(question, k=5)

    if docs:
        # Deduplicate content
        seen_texts = set()
        unique_chunks = []
        for doc in docs:
            content = doc.page_content.strip()
            if content and content not in seen_texts:
                seen_texts.add(content)
                unique_chunks.append(content)

        # discord limit
        msg = "\n\n".join(unique_chunks)
        if len(msg) > 2000:
            msg = msg[:1997] + "..."

        await ctx.send(msg)
        await ctx.message.add_reaction('✅')
        return

    # --- fallback to TrueNorth backend ---
    payload = {"question": question, "chat_history": []}
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        answer = data.get("response", "No response found")

        
        MAX_LEN = 1900
        full_message = f"**Q:** {question}\n**A:** {answer}"

        for chunk in [full_message[i:i+MAX_LEN] for i in range(0, len(full_message), MAX_LEN)]:
            await ctx.send(chunk)
        await ctx.message.add_reaction('✅')

    except requests.exceptions.RequestException as e:
        print(f"❌ Error contacting backend: {e}")
        await ctx.send("Sorry, I could not reach the TrueNorth API.")


#help
@bot.command(name='help')
async def helpme(ctx):
    help_text = "Here are my current commands:\n!joke: tells a joke!"
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

        max_length = 1900
        full_message = f"**Q:** {question}\n**A:** {answer}"

        for chunk in [full_message[i:i+MAX_LEN] for i in range(0, len(full_message), MAX_LEN)]:
            await ctx.send(chunk)
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

        MAX_LEN = 1900
        full_message = f"**Q:** {question}\n**A:** {answer}"

        for chunk in [full_message[i:i+MAX_LEN] for i in range(0, len(full_message), MAX_LEN)]:
            await interaction.followup.send(chunk)

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
