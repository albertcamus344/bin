#!/usr/bin/env python3
"""
Personal Assistant Telegram Bot
Powered by Google Gemini — runs on your own machine.
"""

import io
import logging
import os

import PIL.Image
import google.generativeai as genai
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ─── Configuration ────────────────────────────────────────────────────────────

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Optional: restrict the bot to specific Telegram user IDs.
# Get your ID by messaging @userinfobot on Telegram.
# Leave ALLOWED_USER_IDS blank in .env to allow everyone.
_raw_ids         = os.getenv("ALLOWED_USER_IDS", "")
ALLOWED_USER_IDS = {int(x) for x in _raw_ids.split(",") if x.strip().isdigit()}

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful, friendly, and concise personal assistant. "
    "You assist with questions, writing, analysis, brainstorming, and everyday tasks. "
    "Keep responses clear and well-structured. "
    "Since this is a Telegram chat, prefer plain readable text over heavy markdown. "
    "Use emojis sparingly — only when they genuinely add value.",
)

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ─── Gemini setup ────────────────────────────────────────────────────────────

genai.configure(api_key=GEMINI_API_KEY)

_model = genai.GenerativeModel(
    model_name=GEMINI_MODEL,
    system_instruction=SYSTEM_PROMPT,
)

# Per-user multi-turn chat sessions stored in memory.
# Sessions are lost when the bot restarts — use /new to reset manually.
_sessions: dict[int, genai.ChatSession] = {}


def get_session(uid: int) -> genai.ChatSession:
    """Return the existing chat session for a user, or create a new one."""
    if uid not in _sessions:
        _sessions[uid] = _model.start_chat(history=[])
    return _sessions[uid]


def clear_session(uid: int) -> None:
    _sessions.pop(uid, None)


def is_allowed(uid: int) -> bool:
    """Return True if the user is authorised to use the bot."""
    return not ALLOWED_USER_IDS or uid in ALLOWED_USER_IDS


async def split_send(update: Update, text: str) -> None:
    """Send a reply, splitting into chunks to stay within Telegram's 4096-char limit."""
    for chunk in (text[i : i + 4096] for i in range(0, len(text), 4096)):
        await update.message.reply_text(chunk)


# ─── Command handlers ─────────────────────────────────────────────────────────


async def cmd_start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    u = update.effective_user
    if not is_allowed(u.id):
        await update.message.reply_text("🚫 You are not authorised to use this bot.")
        return
    await update.message.reply_text(
        f"👋 Hey {u.first_name}! I'm your personal AI assistant.\n\n"
        "💬 Send me any message and I'll help you out.\n"
        "🖼️  Send a photo (with an optional caption) and I'll analyse it.\n\n"
        "Commands:\n"
        "  /new   — Clear conversation history\n"
        "  /help  — Usage help\n"
        "  /about — About this bot"
    )


async def cmd_help(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🤖 Personal Assistant Help\n\n"
        "Just type anything — I remember your conversation context within a session.\n"
        "Send a photo (with an optional question as caption) for image analysis.\n\n"
        "Commands:\n"
        "  /start — Welcome message\n"
        "  /new   — Clear history and start fresh\n"
        "  /help  — This message\n"
        "  /about — About this bot"
    )


async def cmd_new(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    clear_session(update.effective_user.id)
    await update.message.reply_text("✅ Conversation cleared. Fresh start!")


async def cmd_about(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🤖 Personal Assistant Bot\n\n"
        f"• AI model:   {GEMINI_MODEL}\n"
        "• Framework:  python-telegram-bot\n"
        "• AI SDK:     google-generativeai\n\n"
        "Your private assistant — runs entirely on your own machine."
    )


# ─── Message handlers ─────────────────────────────────────────────────────────


async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Forward text messages to Gemini and reply with the response."""
    u = update.effective_user
    if not is_allowed(u.id):
        return

    await ctx.bot.send_chat_action(update.effective_chat.id, "typing")

    try:
        reply = get_session(u.id).send_message(update.message.text).text
        log.info("[user %d] text → %d-char reply", u.id, len(reply))
        await split_send(update, reply)
    except Exception as exc:
        log.error("[user %d] text error: %s", u.id, exc)
        await update.message.reply_text(
            "⚠️ Something went wrong. Please try again, or use /new to reset."
        )


async def handle_image(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Download a photo, pass it to Gemini vision, and reply with the analysis."""
    u = update.effective_user
    if not is_allowed(u.id):
        return

    await ctx.bot.send_chat_action(update.effective_chat.id, "typing")

    try:
        # Download the highest-resolution version of the photo
        tg_file = await ctx.bot.get_file(update.message.photo[-1].file_id)
        raw     = await tg_file.download_as_bytearray()
        img     = PIL.Image.open(io.BytesIO(bytes(raw)))

        # Use the message caption as the prompt, or fall back to a description request
        prompt = update.message.caption or "Describe this image in detail."

        # Image analysis is single-turn (not added to the conversation session)
        reply = _model.generate_content([img, prompt]).text
        log.info("[user %d] image → %d-char reply (caption: %r)", u.id, len(reply), prompt[:40])
        await split_send(update, reply)

    except Exception as exc:
        log.error("[user %d] image error: %s", u.id, exc)
        await update.message.reply_text(
            "⚠️ Couldn't process that image. Please try again."
        )


# ─── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    if not TELEGRAM_TOKEN:
        raise SystemExit("❌  TELEGRAM_TOKEN is not set. Add it to your .env file.")
    if not GEMINI_API_KEY:
        raise SystemExit("❌  GEMINI_API_KEY is not set. Add it to your .env file.")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(CommandHandler("new",   cmd_new))
    app.add_handler(CommandHandler("about", cmd_about))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))

    print(f"✅  Bot is running  (model: {GEMINI_MODEL})")
    print("   Press Ctrl+C to stop.\n")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
