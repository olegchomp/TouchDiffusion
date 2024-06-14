@echo off

set PYTHON_PATH=

REM Check if Git is installed
where git > nul 2>&1
if %errorlevel% neq 0 (
    echo Git is not installed or not found in the system PATH.
    pause
    exit /b 1
) else (
    echo Git is installed.
)

if not exist .venv (
    echo Creating .venv directory...
    %PYTHON_PATH% -m venv ".venv" || (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )

    echo Activating virtual environment...
    call .venv\Scripts\activate || (
        echo Failed to activate virtual environment.
        pause
        exit /b 1
    )

    echo Installing dependencies...
    python -m pip install --upgrade pip || (
        echo Failed to update pip.
        pause
        exit /b 1
    )

    echo Installing dependencies...
    pip install gradio || (
        echo Failed to install gradio.
        pause
        exit /b 1
    )
    
    if not exist StreamDiffusion (
        echo Downloading StreamDiffusion...
        git clone https://github.com/olegchomp/StreamDiffusion || (
            echo Failed to download StreamDiffusion
            pause
            exit /b 1
        )
    )
    
    echo Installation complete.

    cd StreamDiffusion
    
    echo Launching WebUI...
    python webui.py || (
        echo No launch file found
        pause
        exit /b 1
    )

) else (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat || (
        echo Failed to activate virtual environment.
        pause
        exit /b 1
    )

    cd StreamDiffusion
    
    echo Launching WebUI...
    python webui.py || (
        echo No launch file found
        pause
        exit /b 1
    )
)

pause
