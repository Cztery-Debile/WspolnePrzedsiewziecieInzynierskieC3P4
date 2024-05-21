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
5. Install the rest of packages
```bash
pip install opencv-python customtkinter ultralytics
```

## Using RTSP stream [Optional]
As of right now, the code is using RTSP stream hosted on localhost. If you want to use your own RTSP stream, feel free to change it directly in the main.py code, line 42-48

```python
        if place == "krakau":
            model = "models/tokioKrakow6000.pt"
            detect_from_video("rtsp://localhost:8554/file?file=krakauStragan.mkv",model, centered="false")
        elif place == "tokio":
            model = "models/tokioKrakow5000.pt"
            detect_from_video("rtsp://localhost:8554/file?file=tokio.mkv",model,centered="true")
    elif method == "faces":
        model = "models/first_own_dataset.pt"
        detect_from_video_zone("rtsp://localhost:8554/file?file=ProjektMBox.mkv", model)
```
The first argument in detect_from_video and detect_from_video_zone is link to RTSP stream.

If  you don't have access to the RTSP camera, you can simulate  one using  for example this  emulator: https://github.com/vzhn/bstreamer

I attached directory with bstreamer-0.5.1 so you can directly copy it to your WSL and launch it from there. Ensure first that your firewall isn't blocking connections from WSL and you have execute permission on bstreamer directory.

##  Using different models
You can change path to the models directly inside the main.py code. The lines are started with "model="  and path to the model. To change models for different actions you have to change it in line:
1. Analiza całego zdjęcia/Analizuj fragmentami/Analizuj zaznaczony obszar - Analyze photo
   * Line 278
2.  "Kraków" button - RTSP stream from Kraków
    * Line 41
3. "Tokio" button - RTSP stream from Tokio
    * Line 44
4. "Wykrywanie twarzy i liczenie czynności" button - analyze video and indentify person + count time
    * Line 47
5. "Wideo" button -  analyze general video
    * Line 50