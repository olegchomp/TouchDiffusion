@echo off

REM Check if Git is installed
where git > nul 2>&1
if %errorlevel% neq 0 (
    echo Git is not installed or not found in the system PATH.
    pause
    exit /b 1
) else (
    echo Git is installed.
)

REM Set default Python path manually if needed
set PYTHON_PATH=

REM Find the Python executable path
for /f "delims=" %%i in ('where python') do set PYTHON_PATH=%%i

REM Check if Python executable is found
if "%PYTHON_PATH%" neq "" (
    REM Extract Python version string
    for /f "tokens=2 delims= " %%v in ('"%PYTHON_PATH%" --version 2^>^&1') do (
        set PYTHON_VERSION=%%v
        echo Found Python version: %%v

        REM Split version based on dot and get first two elements
        for /f "tokens=1,2 delims=." %%a in ("%%v") do (
            set PYTHON_MINOR_VERSION=%%b

            REM Check if minor version is 11
            if "%%b" equ "11" (
                echo Python path: %PYTHON_PATH%
            ) else (
                echo Python version is not 3.11 or later.
                pause
                exit /b 1
            )
        )
    )
) else (
    echo Python is not installed.
    pause
    exit /b 1
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
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118 || (
        echo Failed to install torch and torchvision.
        pause
        exit /b 1
    )

    if not exist engines (
        echo Creating 'engines' folder...
        mkdir engines
    )

    if not exist models (
        echo Creating 'models' folder...
        mkdir models
    )

    if not exist StreamDiffusion (
        echo Downloading StreamDiffusion...
        git clone https://github.com/olegchomp/StreamDiffusion || (
            echo Failed to download StreamDiffusion
            pause
            exit /b 1
        )
    )

    cd StreamDiffusion

    echo Installing dependencies...
    pip install -r requirements.txt|| (
        echo Failed to install dependencies.
        pause
        exit /b 1
    )

    python setup.py develop easy_install streamdiffusion[tensorrt] || (
        echo Failed to run setup.py and install streamdiffusion.
        pause
        exit /b 1
    )

    python -m streamdiffusion.tools.install-tensorrt || (
        echo Failed to install TensorRT.
        pause
        exit /b 1
    )

    pip uninstall -y nvidia-cudnn-cu11
    
    echo Installation complete.

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
