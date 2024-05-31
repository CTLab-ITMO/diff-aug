git clone https://github.com/piyushK52/comfy-runner
ren comfy-runner comfy_runner
copy NUL comfy_runner/__init__.py
python3 -m venv venv
./venv/bin/activate.bat
pip install -r comfy_runner/requirements.txt
