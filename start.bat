@echo off
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting AI Hub...
echo Open http://localhost:8000 in your browser
echo.
python main.py
pause
