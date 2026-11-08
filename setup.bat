@echo off
py -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup complete. Run run.bat to launch PromoRank AI.
