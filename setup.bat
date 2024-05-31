git clone https://github.com/piyushK52/comfy-runner
copy comfy-runner\requirements.txt requirements.txt
copy comfy-runner\inf.py inf.py
copy comfy-runner\constants.py constants.py
robocopy /e comfy-runner\utils utils
robocopy /e comfy-runner\data data
rmdir /s /q comfy-runner

python -m venv venv
call ./venv/Scripts/activate.bat
pip install -r requirements.txt
