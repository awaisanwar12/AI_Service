import asyncio
from telegram import Bot, Update
from telegram.ext import Application, ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import NetworkError, TimedOut
from app.core.config import get_settings
from app.services.rag_service import rag_service
import logging
import sys
import codecs
from typing import Optional
import backoff
import json
import time

# Ensure stdout can handle Unicode
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for maximum logging
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('telegram_bot.log', encoding='utf-8'),
        logging.FileHandler('services.log', encoding='utf-8'),  # Separate file for services
        logging.StreamHandler(sys.stdout)
    ]
)

# Create separate loggers for different components
logger = logging.getLogger(__name__)
redis_logger = logging.getLogger('redis_service')
pinecone_logger = logging.getLogger('pinecone_service')
xai_logger = logging.getLogger('xai_service')
rag_logger = logging.getLogger('rag_service')

class TelegramBotService:
    def __init__(self):
        self.application: Optional[Application] = None
        self.initialized = False
        self.settings = get_settings()
        self.retry_count = 0
        self.max_retries = 5
        logger.info("🔄 TelegramBotService initialization started")
        logger.debug(f"Settings loaded: {self.settings}")

    async def initialize(self):
        """Initialize the bot application"""
        if not self.initialized:
            start_time = time.time()
            logger.info("🚀 Starting Telegram bot initialization...")
            try:
                token = self.settings.TELEGRAM_TOKEN
                logger.info(f"Using Telegram token: {token[:5]}...{token[-5:]}")
                logger.debug("Creating application with custom settings")
                
                # Create application with custom settings
                self.application = (
                    Application.builder()
                    .token(token)
                    .connect_timeout(30.0)
                    .read_timeout(30.0)
                    .write_timeout(30.0)
                    .get_updates_read_timeout(30.0)
                    .pool_timeout(30.0)
                    .build()
                )
                logger.debug("Application instance created successfully")
                
                # Add handlers
                self.application.add_handler(CommandHandler("start", self.start_command))
                self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
                logger.debug("Command and message handlers added")
                
                # Add error handler
                self.application.add_error_handler(self.error_handler)
                logger.debug("Error handler added")
                
                self.initialized = True
                end_time = time.time()
                logger.info(f"✅ Telegram bot initialized successfully in {end_time - start_time:.2f} seconds")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Telegram bot: {str(e)}", exc_info=True)
                raise

    @backoff.on_exception(
        backoff.expo,
        (NetworkError, TimedOut),
        max_tries=5,
        max_time=300
    )
    async def start(self):
        """Start the bot with retry logic"""
        if not self.initialized:
            await self.initialize()
        
        logger.info("Starting Telegram bot...")
        try:
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True,
                read_timeout=30,
                timeout=30
            )
            logger.info("Bot started successfully")
        except Exception as e:
            logger.error(f"Failed to start bot: {str(e)}", exc_info=True)
            raise

    async def stop(self):
        """Stop the bot"""
        if self.application:
            logger.info("Stopping Telegram bot...")
            try:
                await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()
                logger.info("Bot stopped successfully")
            except Exception as e:
                logger.error(f"Failed to stop bot: {str(e)}", exc_info=True)

    async def error_handler(self, update: Optional[Update], context: ContextTypes.DEFAULT_TYPE):
        """Handle errors in the bot"""
        try:
            if isinstance(context.error, NetworkError):
                logger.warning(f"Network error occurred: {str(context.error)}")
                # Wait a bit before retrying
                await asyncio.sleep(1)
            elif isinstance(context.error, TimedOut):
                logger.warning(f"Request timed out: {str(context.error)}")
                await asyncio.sleep(1)
            else:
                logger.error(f"Update {update} caused error: {str(context.error)}", exc_info=context.error)
        except Exception as e:
            logger.error(f"Error in error handler: {str(e)}", exc_info=True)

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user
        chat_id = update.effective_chat.id
        logger.info("START command received:")
        logger.info(f"  - User ID: {user.id}")
        logger.info(f"  - Username: {user.username}")
        logger.info(f"  - Chat ID: {chat_id}")
        
        try:
            # Show typing indicator
            await context.bot.send_chat_action(
                chat_id=chat_id,
                action="typing"
            )
            
            await context.bot.send_message(
                chat_id=chat_id,
                text="Hello! I'm your AI assistant. How can I help you today?"
            )
            logger.info("Start message sent successfully")
        except Exception as e:
            logger.error(f"Error sending start message: {str(e)}", exc_info=True)
            await self.error_handler(update, context)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming messages"""
        message_start_time = time.time()
        try:
            user = update.effective_user
            chat_id = update.effective_chat.id
            user_message = update.message.text
            
            # Log incoming message details
            logger.info("=" * 80)
            logger.info("TELEGRAM MESSAGE RECEIVED")
            logger.info(f"FROM: User {user.username} (ID: {user.id})")
            logger.info(f"CHAT ID: {chat_id}")
            logger.info(f"MESSAGE: '{user_message}'")
            logger.info("-" * 40)
            
            # Show typing indicator while processing
            await context.bot.send_chat_action(
                chat_id=chat_id,
                action="typing"
            )
            
            # Get response from RAG service
            response = await rag_service.process_message(user_message, "")
            
            # Send response
            await context.bot.send_message(
                chat_id=chat_id,
                text=response["text"]
            )
            
            # Send image if present
            if response.get("image_url"):
                # Show upload photo action
                await context.bot.send_chat_action(
                    chat_id=chat_id,
                    action="upload_photo"
                )
                await context.bot.send_photo(
                    chat_id=chat_id,
                    photo=response["image_url"]
                )
            
            # Log completion
            message_end_time = time.time()
            logger.info("-" * 40)
            logger.info(f"MESSAGE PROCESSED IN: {message_end_time - message_start_time:.1f}s")
            logger.info("=" * 80)
                
        except Exception as e:
            logger.error("ERROR PROCESSING TELEGRAM MESSAGE:")
            logger.error(f"Type: {type(e).__name__}")
            logger.error(f"Message: {str(e)}")
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="Sorry, I encountered an error processing your message."
                )
            except Exception as send_error:
                logger.error(f"Failed to send error message: {str(send_error)}")
            await self.error_handler(update, context)

bot_service = TelegramBotService()