1. Create venv
```bash
python -m venv .venv
```
2. Activate venv
```bash
.venv/Scripts/activate
```
3a. If NVIDIIA GPU:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
3b. If not
```bash
pip install torch torchvision torchaudio
```
4. Download requirements
```bash
pip install -r "requirements.txt"
```