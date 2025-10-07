import os
import random
import discord
import google.generativeai as genai
from dotenv import load_dotenv
import requests

API_URL = "http://localhost:8000/query"
#connecting the bot
from discord.ext import commands
#loading the token and guild
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('models/gemini-1.5-pro')


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
    """Ask a question to TrueNorth"""
    await ctx.send("Thinking...")

    payload = {"question": question, "chat_history": []}

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        answer = data.get("response", "No response found")
        await ctx.send(f"**Q:**{question}\n**A:**{answer}")
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
    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
        ]
        response = model.generate_content(question)
        max_length = 2000
        answer = f"**Q:** {question}\n**A:** {response.text}"
        if len(answer) > max_length:
            answer = answer[:max_length - 3] + "..."
        await ctx.send(answer)
        await ctx.message.add_reaction('✅')

    except Exception as e:
        if "429" in str(e):
            await ctx.send("Sorry, I have hit the limit.")
        else:
            await ctx.send("Please Try Again")
            print(f"Error in geminiquestion command: {e}")
        
@bot.tree.command(name="geminiquestion", description="Ask Gemini anything")
async def ask_slash(interaction: discord.Interaction, question: str):
    await interaction.response.defer()
    try:
        response = model.generate_content(question)
        max_length = 2000
        answer = f"**Q:** {question}\n**A:** {response.text}"
        if len(answer) > max_length:
            answer = answer[:max_length - 3] + "..."
        await interaction.followup.send(answer)
    except Exception as e:
        await interaction.followup.send("Something went wrong. Try again.")
        print(f"Error in slash geminiquestion: {e}")
@bot.event
async def on_error(event, *args, **kwargs):
    with open('err.log', 'a') as f:
        if event == 'on_message':
            f.write(f'Unhandled message: {args[0]}\n')
        else:
            raise

bot.run(TOKEN)

#run with python bot.py