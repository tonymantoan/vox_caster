import sys
import os
import numpy as np
import signal
from threading import Event, Thread
import time

from lightning_whisper_mlx import LightningWhisperMLX
from speechbrain.inference.speaker import SpeakerRecognition
from asteroid.models import BaseModel
from vox_caster_ui import VoxCasterUI
from kokoro_processor import KokoroProcessor

import sounddevice as sd
import soundfile as sf
import torch
import torchaudio

# from speechbrain.inference.separation import SepformerSeparation
# import noisereduce as nr

class VoxCaster:
    def __init__(self, ui_app: VoxCasterUI | None = None):
        self.ui_app = ui_app
        # audio settings
        self.SAMPLE_RATE = 24000
        self.WHISPER_SAMPLE_RATE = 16000
        self.PRE_BUFFER_SIZE = 1600
        self.SILENCE_THRESHOLD = 0.01   # default volume level that counts as silence
        self.SILENCE_DURATION = 1.5    # seconds of silence before cutting recording
        self.INTERRUPTION_DURATION = 0.75
        self.INTERRUPTION_SAMPLE_SIZE = self.WHISPER_SAMPLE_RATE * self.INTERRUPTION_DURATION
        self.interruption_detection_buffer = np.zeros((1, 1))

        self.shutdown_event = Event()
        signal.signal(signal.SIGINT, self._signal_handler)

        print("Initializing Whisper Lightning")
        self.whisper_mlx = LightningWhisperMLX(model="distil-large-v2", batch_size=12)

        os.makedirs("known_voices", exist_ok=True)

        self.kokoro = KokoroProcessor()
        self.input_stream = None

        self._init_known_voices()
        print("TAKING AMBIENT NOISE SAMPLE, PLEASE BE QUIET.")
        ambient_sample_size = 1.5 * self.WHISPER_SAMPLE_RATE
        self.ambient_noise_sample = sd.rec(int(ambient_sample_size), samplerate=self.WHISPER_SAMPLE_RATE, channels=1)
        sd.wait()
        self.SILENCE_THRESHOLD = (np.abs(self.ambient_noise_sample).mean())
        print(f"DONE SAMPLING AMBIENT NOISE - Volume is {self.SILENCE_THRESHOLD}")

    def _signal_handler(self, signum, frame):
        print("\nStopping...")
        self.shutdown_event.set()
    
    def _init_known_voices( self ):
        self.known_voices = dict()
        for f in os.listdir( 'known_voices' ):
            voice_name = f.split('/')[-1].split('.')[0]
            print( f"Found known voice {f} name: {voice_name}" )
            signal, fs = torchaudio.load( f"known_voices/{f}" )
            self.known_voices[voice_name] = signal
        
        self.verification = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir='speechbrain/speaker_recognition')
        
        self.source_separator = BaseModel.from_pretrained(
            "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k")

    def handle_interruption_detect( self, sources ):
        for source in sources:
            print(f"Task is done with source voice: {source[0]}")
            if source[0] != "teletran-1":
                self.kokoro.interrupt_generation()
                break

    def id_voice( self, voice_sample ):
        voice_id = 'unknown'
        voice_sample_tensor = torch.from_numpy( voice_sample ).float()
        for voice_name in self.known_voices:
            score, prediction = self.verification.verify_batch(self.known_voices[voice_name], voice_sample_tensor)
            print ( f"Voice: {voice_name}, Score: {score.item()}, prediction: {prediction.item()}" )
            if prediction.item():
                voice_id = voice_name
                break
        
        return voice_id
    
    def id_sources( self, audio_data ):
        est_sources = self.source_separator.separate( audio_data )
        est_sources = est_sources[0] # strip batch dimension
        sources = []
        for source in est_sources:
            source_volume = np.abs(source).mean()
            if source_volume < self.SILENCE_THRESHOLD:
                print(f"Low source volume: {source_volume} ")
                continue
            
            voiceId = self.id_voice( source )
            sources.append( (voiceId, source) )
        print( f"Found sources: {len(sources)}" )
        return sources

    def check_for_interruption( self, audio_buffer ):
        sources = self.id_sources( np.array( audio_buffer, dtype=np.float32 ) )
        self.handle_interruption_detect( sources )
    
    def noise_cancellation( self, audio ):
        """Applies a simple spectral noise reduction filter."""
        noise_reduction_factor = 0.1
        level = np.abs(audio).mean()
        # print(f"MEan noise level: {level}")
        noise_threshold = level * noise_reduction_factor
        denoised_audio = np.where(audio > noise_threshold, audio, 0)
        return denoised_audio

    def record_and_transcribe(self):
        # state for audio recording
        audio_buffer = np.array([])
        audio_prebuffer = np.array([])
        silence_frames = 0
        # total_frames = 0
        input_ready_event = Event()
        audio_segment = []
        current_speaker_name = 'unknown'
        # chunk_size = int(self.SAMPLE_RATE * 50 / 1000)

        def callback(indata, frames, time_info, status):
            # callback function that processes incoming audio frames
            if self.shutdown_event.is_set():
                raise sd.CallbackStop()

            nonlocal audio_buffer, audio_prebuffer, silence_frames, current_speaker_name, audio_segment

            if status:
                print(status)

            audio_raw = indata.flatten()
            audio_prebuffer = np.append( audio_prebuffer, audio_raw )
            if self.PRE_BUFFER_SIZE > len(audio_prebuffer):
                return

            # print( f"raw mean: {np.abs(audio_prebuffer).mean()}" )
            audio = self.noise_cancellation( audio_prebuffer )
            # audio = audio_prebuffer.copy()
            audio_prebuffer = np.array([])

            # This is bus erroring, but might be conflicting with other packages, could try again after fresh installs
            # nr.reduce_noise(y=audio, sr=self.WHISPER_SAMPLE_RATE)

            if self.kokoro.run_generation:
                self.interruption_detection_buffer = np.append( self.interruption_detection_buffer, indata )
                if len( self.interruption_detection_buffer ) > self.INTERRUPTION_SAMPLE_SIZE:
                    self.interruption_detection_buffer = self.interruption_detection_buffer.reshape( 1,1, self.interruption_detection_buffer.shape[0])
                    t = Thread( target=self.check_for_interruption, args=(self.interruption_detection_buffer.copy()) )
                    t.start()
                    self.interruption_detection_buffer = np.array([])
                    audio_buffer = np.array([])
                    return

            level = np.abs(audio).mean()
            # print( f"Audio Level {level}, silence thres: {self.SILENCE_THRESHOLD}" )


            # track silence duration
            if level < self.SILENCE_THRESHOLD:
                silence_frames += len(audio)
            else:
                silence_frames = 0

            audio_buffer = np.append( audio_buffer, audio )

            # process audio when silence is detected
            if silence_frames > self.SILENCE_DURATION * self.WHISPER_SAMPLE_RATE:
                speech_frames = len(audio_buffer) - silence_frames
                audio_segment = audio_buffer[:speech_frames] #audio_buffer.copy() #np.array(audio_buffer, dtype=np.float32)
                audio_segment_volume = np.abs(audio_segment).mean() if len(audio_segment>0) else 0
                print( f"AUDIO_SEG_VOL: {audio_segment_volume}, LENGHT: {len(audio_segment)}, SILENCE: {silence_frames}, SPEECH_FRAMES: {speech_frames}" )
                
                # print(f"Audio length: {len(audio_segment)}")
                if len(audio_segment) > self.WHISPER_SAMPLE_RATE:
                    current_speaker_name = self.id_voice( audio_segment )
                    if current_speaker_name != "teletran-1":
                        input_ready_event.set()

                # reset state
                audio_buffer = np.array([]) #.clear()
                silence_frames = 0

        # start recording loop
        try:
            with sd.InputStream(
                callback=callback,
                channels=1,
                samplerate=self.WHISPER_SAMPLE_RATE,
                dtype=np.float32
            ) as stream:
                print("Recording... Press Ctrl+C to stop")
                self.input_stream = stream
                while not self.shutdown_event.is_set():
                    sd.sleep(100)
                    input_ready_event.wait()
                    # stream.stop()
                    input_ready_event.clear()
                    # nr_audio = nr.reduce_noise(y=audio_segment, y_noise=self.ambient_noise_sample, sr=self.WHISPER_SAMPLE_RATE)
                    nr_audio = audio_segment
                    audio_volume = np.abs(nr_audio).mean()

                    if current_speaker_name == "unknown" and audio_volume < 0.003:
                        stream.start()
                        continue
                    
                    text = ""
                    if audio_volume != float('nan'):
                        text = self.whisper_mlx.transcribe(nr_audio)['text'].strip()
    
                    print(f"Add to input: {current_speaker_name}: {text.strip()}")
                    if len( text.split(' ') ) > 2:
                        self.ui_app.append_to_input( f"{current_speaker_name}: {text.strip()}" )
                        # print(f"{current_speaker_name}: {text.strip()}")
                    # else:
                    #     print( "Restarting Stream after false positive" )
                    #     stream.start()
        except sd.CallbackStop:
            print( "Stop input stream...." )
            pass
    
    def generate_voice_response( self, text, voice="am_puck" ):
        if self.input_stream:
            self.input_stream.start()
        
        # t = Thread( target=self.kokoro.generate_audio, args=(text, voice) )
        # t.start()

        self.kokoro.generate_audio( text, voice )
    
    def pause_input_stream( self ):
        print( "Pausing input Stream" )
        if self.input_stream:
            self.input_stream.stop()

    def resume_input_stream( self ):
        print( "Restarting Stream after response" )
        if self.input_stream:
            self.input_stream.start()

def main():
    def handle_input(input_text):
        app.toggle_audio_switch()
        print("TURN INDICATOR ON")
        time.sleep(2)
        print("TURN INDICATOR OFF")
        app.toggle_audio_switch()

    
    def handle_audio_switch( switch_state ):
        app.add_to_log( f"Audio switch handler: {switch_state}" )
        if switch_state == True:
            vc.resume_input_stream()
        else:
            vc.pause_input_stream()
    
    my_ui_callbacks = {
        "audio_switch_callback": handle_audio_switch,
        "tx_button_callback": handle_input
    }

    app = VoxCasterUI( ui_callbacks=my_ui_callbacks )
    vc = VoxCaster( app )
   
    t = Thread( target=vc.record_and_transcribe )
    try:
        t.start()
        app.run()
        # t.join()
    except KeyboardInterrupt:
        sys.exit('\nExit by user')

    # try:
    #     k = KokoroProcessor()
    #     k.generate_audio( k.demo_text )
    # except KeyboardInterrupt:
    #     sys.exit('\nExit by user')


if __name__ == "__main__":
    main()

    # print(sd.query_devices())

    # Record new known voices"
    #
    # voice_sr = 16000
    # duration = 5
    # name = "Kelly"
    # input("Press enter when you are ready to Record Audio")
    # voice_data = sd.rec(int(duration * voice_sr), samplerate=voice_sr, channels=1)
    # sd.wait()
    # sf.write( f"known_voices/{name}.wav", voice_data, voice_sr )