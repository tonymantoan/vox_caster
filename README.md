# Vox Caster TTS and STT Terminal UI 
Vox Caster is a terminal text UI for Text To Speech and speech to text applications. It is designed to be a UI for my locally running LLM assistant, but could be used for other TTS/STT applications such as transcribing. It's still very much a work in progress, but core functionality works decently, though it can be fragile in a noisy environment.

## Run the Demo App
The demo app show how to use Vox Caster with a local LLM (or really any inference endpoint).

Python 3.12 is recommended (had some issues with 3.13).

1. Edit the url or any other config you need to at the top of vox_caster_demo.py
2. Create a virtual env: `python3.12 -m venv vox_venv`
3. `source vox_venv/bin/activate`
4. Install the pip requirements: `pip install -r requirements.txt`
5. For debug info, open another terminal in the same directory, source the venv, and do `textual console`
6. Make sure your llm server is running with the endpoint specified in vox_caster_demo.py
7. Run the app `textual run --dev vox_caster_demo.py`

The first time is runs will need to download several models for whisper, kokoro, speechbrain, and astroid.

### Voice Recognition
Vox Caster needs audio samples of a voice to compare recorded audio too. I am planning to add some UI widgets to make it easy to add voices, but in the meantime you can edit the __main__ function in `vox_caster_audio.py`. Comment out the call to `main()` and uncomment the section under "record new known voices". Edit the `voice_name` var and run it as an app `python vox_caster_audio.py`. Talk when it says to and it will record and save the sample, then exit.

## Vox Caster Features
* Speech to text with Whisper models
* Text to speech with Kokoro 82M
* Speaker recognition (needs a voice recording first)
* Audio source separation for two sources with speaker ID for both
* Audio input noise cancellation
* TTS interruption by voice
* STT is placed in a text area input that can be corrected before being sent
* Scrollable log view
* Turn audio input on or off as needed

## Todo List
* Easier configuration
* Auto detect OS/processor and use openai-whisper if not on Apple silicon
* Improved documentation
* General voice detection to determine when a speaker is done instead of waiting for silence (should help in noisy environments)
* Source separation for more voices (ideally an arbitrary number)
* Better noise cancellation
* System audio cancellation with loopback device
* Timestamps for speech segments from different voices
* Improved controls and indicators
* Option to integrate with multi modal modes instead of using whisper and kokoro
* Make Vox Caster a pypi library