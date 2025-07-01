from kokoro import KPipeline
import sounddevice as sd
from threading import Event, Thread
import librosa
import numpy as np

class KokoroProcessor:
    def __init__(self):
        self.pipeline = KPipeline(lang_code='a')
        self.DEFAULT_FREQUENCY = 24000
        self.demo_text = "Hello, friends. I am Teletran-1! Nice to meet you. I am a helpful assistant who can answer questions and do some tasks."
        self.current_frame = 0
        self.run_generation = False
        self.audio_ready_for_playback = False
        # self.output_runtime = 0
    
    def audio_output_callback( self, outdata, frames, time, status ):
        if status:
            print( f"Kokoro play audio status: {status}" )
        
        chunksize = min(len(self.audio_data) - self.current_frame, frames)
        outdata[:chunksize] = self.audio_data[self.current_frame:self.current_frame + chunksize].reshape(-1, 1)
        if chunksize < frames:
            outdata[chunksize:] = 0
            self.current_frame = 0
            raise sd.CallbackStop()
        self.current_frame += chunksize

    def generate_audio( self, text, voice="am_puck" ):
        self.run_generation = True
        self.generator = self.pipeline(
            text, voice=voice,
            speed=1, split_pattern=r'\n+'
        )

        for i, (gs, ps, audio) in enumerate(self.generator):
            self.audio_ready_for_playback = False
            if not self.run_generation:
                break
            
            self.current_frame = 0
            print(i)  # i => index
            print(gs) # gs => graphemes/text
            self.audio_data = librosa.resample( np.array(audio), orig_sr=self.DEFAULT_FREQUENCY, target_sr=16000 )
            print("Kokoro audio ready for playback")
            self.audio_ready_for_playback = True
            self.generation_complete_event = Event()
            self.generation_complete_event.wait()
            print("Kokoro done with current audio generation")
        
        print( "Done with all voice generation tasks" )
        self.run_generation = False
        self.audio_ready_for_playback = False
    
    def interrupt_generation( self ):
        self.run_generation = False
        self.audio_ready_for_playback = False
        self.generation_complete_event.set()
        self.generator.close()
        self.current_frame = 0
    
if __name__ == "__main__":
    kokoro = KokoroProcessor()
    voice = "af_sarah"
    text = "Done with all voice generation tasks."
    while text != "bye":
        text = input( "Enter text: " )
        t = Thread( target=kokoro.generate_audio, args=(text, voice) )
        t.start()