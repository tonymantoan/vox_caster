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
import webrtcvad

# from speechbrain.inference.separation import SepformerSeparation
# import noisereduce as nr

class VoxCaster:
    def __init__(self, ui_app: VoxCasterUI | None = None, input_handler = None):
        self.ui_app = ui_app
        self.input_handler = input_handler
        # audio settings
        self.SAMPLE_RATE = 24000
        self.WHISPER_SAMPLE_RATE = 16000
        self.PRE_BUFFER_SIZE = 1920
        self.VAD_BUFFER = 480 # 30ms
        self.SILENCE_DURATION = 1.5    # seconds of silence before cutting recording
        self.INTERRUPTION_DURATION = 0.75
        self.INTERRUPTION_SAMPLE_SIZE = self.WHISPER_SAMPLE_RATE * self.INTERRUPTION_DURATION
        self.interruption_detection_buffer = np.zeros((1, 1))

        self.vad = webrtcvad.Vad( 3 )

        self.shutdown_event = Event()
        signal.signal(signal.SIGINT, self._signal_handler)

        print("Initializing Whisper Lightning")
        self.whisper_mlx = LightningWhisperMLX(model="distil-large-v2", batch_size=12)

        os.makedirs("known_voices", exist_ok=True)

        self.kokoro = KokoroProcessor()
        self.input_stream = None

        self._init_known_voices()

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
        # TODO: Update this when voice detection is implemented so this can be more accurate
        for source in sources:
            print(f"Task is done with source voice: {source[0]}")
            if source[0] != "teletran-1" and source[0] != "unknown":
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
    
    def id_sources( self, audio_data, min_vad_samples ):
        est_sources = self.source_separator.separate( audio_data )
        est_sources = est_sources[0] # strip batch dimension
        sources = []
        for source in est_sources:
            speech_detected = self.vad_for_segment( source, min_vad_samples )
            print( f"ID source voice detected: {speech_detected}")
            if not speech_detected:
                continue
            
            voiceId = self.id_voice( source )
            print( f"Voice ID for detected voice: {voiceId}" )
            sources.append( (voiceId, source) )
        print( f"Found sources: {len(sources)}" )
        return sources
    
    def vad_for_segment( self, audio_segment, min_vad_samples ) -> bool:
        vad_audio_segment = (audio_segment * 32768).astype(np.int16)
        vad_frames = np.array_split(vad_audio_segment, np.ceil(len(vad_audio_segment) / self.VAD_BUFFER))
        voice_frame_count = 0

        for vad_frame in vad_frames:
            if self.vad.is_speech( vad_frame.tobytes(), self.WHISPER_SAMPLE_RATE ):
                voice_frame_count += 1
        
        # vad_percent = voice_frame_count / len(vad_frames)
        
        # Check for specified amount of audio
        # print( f"VAD check if {voice_frame_count*self.VAD_BUFFER} is >= {min_vad_samples}" )
        if (voice_frame_count*self.VAD_BUFFER) >= min_vad_samples:
            return True
        else:
            return False

    def check_for_interruption( self, audio_buffer ):
        sources = self.id_sources( np.array( audio_buffer, dtype=np.float32 ), (self.PRE_BUFFER_SIZE/2) )
        self.handle_interruption_detect( sources )

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

            #spectrogram
            # if self.ui_app.ready == True:
            #     self.ui_app.update_spectrogram( audio_prebuffer.tolist() )

            audio = audio_prebuffer.copy()
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

            # track silence duration
            speech_detected = self.vad_for_segment( audio, self.PRE_BUFFER_SIZE)
            if not speech_detected:
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
                dtype=np.float32,
                blocksize=480
            ) as stream:
                print("Recording... Press Ctrl+C to stop")
                self.input_stream = stream
                while not self.shutdown_event.is_set():
                    sd.sleep(100)
                    input_ready_event.wait()
                    # stream.stop()
                    input_ready_event.clear()
                    # nr_audio = nr.reduce_noise(y=audio_segment, y_noise=self.ambient_noise_sample, sr=self.WHISPER_SAMPLE_RATE)
                    source_audio = audio_segment.reshape( 1,1, audio_segment.shape[0] )
                    source_audio = np.array( source_audio, dtype=np.float32 )

                    # Only include audio if the source is known, or if there is 1 sec confirmed voice activity:
                    name_source_map = self.id_sources( source_audio, self.WHISPER_SAMPLE_RATE )
                    if len( name_source_map ) == 0:
                        stream.start()
                        continue
                    
                    for name, source in name_source_map:
                        text = ""
                        # if audio_volume != float('nan'):
                        text = self.whisper_mlx.transcribe(source)['text'].strip()
        
                        print(f"Add to input: {name}: {text.strip()}")
                        if len( text.split(' ') ) > 2:
                            if self.ui_app == None:
                                self.input_handler( f"{name}: {text.strip()}" )
                            else:
                                self.ui_app.append_to_input( f"{name}: {text.strip()}" )

        except sd.CallbackStop:
            print( "Stop input stream...." )
            pass
    
    def generate_voice_response( self, text, voice="am_puck" ):
        if self.input_stream:
            self.input_stream.start()
        
        t = Thread( target=self.kokoro.generate_audio, args=(text, voice) )
        t.start()

        # self.kokoro.generate_audio( text, voice )
    
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
        app.toggle_processing_indicator()
        # vc.generate_voice_response( input_text, "af_sarah" )
        time.sleep(2)
        app.toggle_audio_switch()
        app.toggle_processing_indicator()
    
    def handle_input_print(input_text):
        print(f"Input recieved: {input_text}")

    
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
    vc = VoxCaster( input_handler=handle_input_print )
   
    t = Thread( target=vc.record_and_transcribe )
    try:
        t.start()
        # app.run()
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