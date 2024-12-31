"""
# Temporarily commented out Telegram bot service
import asyncio
from telegram import Bot, Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from app.core.config import get_settings
from app.services.x_ai_service import generate_content
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class TelegramBotService:
    def __init__(self):
        self.application = None
        self.initialized = False
        self.settings = get_settings()

    async def initialize(self):
        if not self.initialized:
            logger.info("Initializing Telegram bot service...")
            self.application = ApplicationBuilder().token(self.settings.TELEGRAM_TOKEN).build()
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
            self.initialized = True
            logger.info("Telegram bot service initialized successfully")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        logger.info(f"Received /start command from user {user.id} ({user.username})")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Hello! I'm your AI assistant. How can I help you today?"
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user = update.effective_user
            user_message = update.message.text
            logger.info(f"📩 Received message from {user.username} (ID: {user.id})")
            logger.info(f"📝 Message content: {user_message}")
            
            response = await generate_content(user_message)
            logger.info(f"✅ Generated response: {response['text'][:100]}...")
            
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=response["text"]
            )
            
            if response.get("image_url"):
                logger.info("📷 Sending image response...")
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id,
                    photo=response["image_url"]
                )
                logger.info("✅ Image sent successfully")
        except Exception as e:
            logger.error(f"❌ Error handling message: {str(e)}", exc_info=True)
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Sorry, I encountered an error processing your message."
            )

    async def start(self):
        logger.info("Starting Telegram bot...")
        await self.initialize()
        logger.info("Starting message polling...")
        await self.application.start()
        await self.application.run_polling(allowed_updates=Update.ALL_TYPES)

    async def stop(self):
        if self.application:
            logger.info("Stopping Telegram bot...")
            await self.application.stop()
            logger.info("Telegram bot stopped successfully")

bot_service = TelegramBotService()
""" 