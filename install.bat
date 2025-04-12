@echo off
REM Apertis Installation Script for Windows
REM This script helps you install and set up the Apertis LLM framework

echo ================================================
echo   Welcome to Apertis LLM Installation (Windows)
echo   A user-friendly multimodal LLM framework
echo ================================================
echo.

REM Check Python version
echo Checking Python version...
python --version 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Python not found. Please install Python 3.8 or newer.
    exit /b 1
)

REM Ask about virtual environment
set /p create_venv="Do you want to create a virtual environment for Apertis? (recommended) [Y/n]: "
if not defined create_venv set create_venv=Y

if /I "%create_venv%"=="Y" (
    echo Checking for virtual environment tools...
    
    REM Install virtualenv if not already installed
    python -m pip show virtualenv >NUL 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo Installing virtualenv...
        python -m pip install virtualenv
    )
    
    REM Create and activate virtual environment
    echo Creating virtual environment...
    python -m virtualenv venv
    
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    
    echo Virtual environment created and activated!
)

REM Install dependencies
echo Installing dependencies from requirements.txt...
python -m pip install -r requirements.txt

REM Install the package in development mode
echo Installing Apertis in development mode...
python -m pip install -e .

REM Create data directory if it doesn't exist
echo Setting up data directory...
if not exist data mkdir data

REM Create a simple vocabulary file if it doesn't exist
if not exist data\vocab.json (
    echo Creating a sample vocabulary file...
    echo {"^<pad^>": 0, "^<bos^>": 1, "^<eos^>": 2, "^<unk^>": 3, "the": 4, "a": 5, "an": 6, "is": 7, "was": 8, "are": 9, "were": 10} > data\vocab.json
)

REM Create models directory if it doesn't exist
if not exist models mkdir models

REM Create a small model for testing
echo Creating a small test model...
python -c "import os; import sys; import torch; sys.path.insert(0, os.path.abspath('.')); from src.model.core import create_apertis_model; os.makedirs('models', exist_ok=True); model = create_apertis_model(model_size='small', multimodal=True); torch.save(model.state_dict(), 'models/test_model.pt')"

REM Installation complete
echo ================================================
echo   Apertis installation complete!
echo ================================================
echo.

REM Print usage instructions
echo Quick Start:
echo.
echo 1. Start the web interface:
echo    python -m src.inference.interface --model-path models/test_model.pt --vocab-file data/vocab.json --web --multimodal
echo.
echo 2. Start the command-line interface:
echo    python -m src.inference.interface --model-path models/test_model.pt --vocab-file data/vocab.json --multimodal
echo.
echo 3. Run the example scripts:
echo    python examples/simple_chat.py
echo.

REM If using virtual environment, remind the user to activate it
if /I "%create_venv%"=="Y" (
    echo Remember to activate the virtual environment before using Apertis:
    echo    venv\Scripts\activate.bat
    echo.
)

echo For more information, see the documentation in docs/README.md
echo Enjoy using Apertis!
pause
