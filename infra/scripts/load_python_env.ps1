Write-Host 'Creating Python virtual environment ".venv"...'
python3.11 -m venv .venv

Write-Host 'Installing dependencies from "requirements.txt" into virtual environment (in quiet mode)...'
.\.venv\Scripts\python.exe -m pip --quiet --disable-pip-version-check install -r requirements.txt