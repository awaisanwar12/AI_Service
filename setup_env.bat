@echo off
setlocal enabledelayedexpansion

:: Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Create virtual environment
echo Creating virtual environment...
if exist .venv (
    echo Removing existing virtual environment...
    rmdir /s /q .venv
)
python -m venv .venv

:: Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

:: Upgrade pip and install build tools
echo Installing build tools...
python -m pip install --upgrade pip wheel setuptools

:: Clean any existing installations
pip uninstall -y fastapi pydantic pydantic-settings uvicorn typing-extensions

:: Install core dependencies with specific versions
echo Installing core dependencies...
pip install typing-extensions==4.5.0
pip install pydantic==1.10.13
pip install fastapi==0.88.0
pip install "uvicorn[standard]==0.20.0"
pip install python-telegram-bot==20.6
pip install python-dotenv==1.0.0
pip install httpx==0.25.1

:: Install AI and database dependencies
echo Installing AI and database dependencies...
pip install langchain==0.0.350
pip install openai==0.28.1
pip install pinecone-client==2.2.4
pip install aiosqlite==0.19.0
pip install tiktoken==0.5.2
pip install redis==5.0.1
pip install aioredis==2.0.1
pip install celery==5.3.6

:: Verify installations
echo Verifying installations...
python -c "import fastapi; import uvicorn; import pydantic; import langchain" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Some packages failed to install correctly.
    echo Please run the script again or install manually.
    pause
    exit /b 1
)

echo Setup complete!
echo.
echo To run the application:
echo 1. STAY IN THIS COMMAND PROMPT WINDOW
echo 2. Run: python main.py
echo.

:: Run the application directly
set /p RUNNOW="Do you want to run the application now? (Y/N) "
if /i "%RUNNOW%"=="Y" (
    python main.py
) else (
    echo Press any key to exit...
    pause
) 