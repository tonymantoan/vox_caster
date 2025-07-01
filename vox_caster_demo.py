import requests
import typing
import sys
from threading import Thread

from vox_caster_ui import VoxCasterUI
from vox_caster_audio import VoxCaster

SYSTEM_PROMPT = "You are a helpful AI assistant."
BASE_LLM_URL = "http://127.0.0.1:8080/completion"
    
class Message():
    role: str
    content: str

    def __init__( self, role: str, content: str ):
        self.role = role
        self.content = content

SYS_MESSAGE = Message( 'system', SYSTEM_PROMPT )

class LocalAiClient():
    def __init__( self, vc_ui: VoxCasterUI, vc_audio: VoxCaster ):
        self.vc_ui = vc_ui
        self.vc_audio = vc_audio

        self.base_url = BASE_LLM_URL
        self.max_tokens = 16384
        self.temp = 0.7
    
    def generate_response( self, messages: list[Message] ) -> dict[str, typing.Any]:
        print( f'Generate Local llm response for prompts:' )
        full_prompt = self.format_prompt_for_v7_tekken( messages )
        print( f'Full prompt is: {full_prompt}' )

        request_body = {
            "prompt": full_prompt,
            "n_predict": self.max_tokens,
            "temperature": self.temp
        }

        # Call the local LLM inference endpoint
        response = requests.post( self.base_url, json=request_body )
        return response.json()['content'].strip()

    # This prompt format is for recent (2025) Mistral models, update or change for whichever model you are running locally
    def format_prompt_for_v7_tekken(self, messages: list[Message]) -> str:
        # <s>[SYSTEM_PROMPT]<system prompt>[/SYSTEM_PROMPT][INST]<user message>[/INST]<assistant response></s>[INST]<user message>[/INST]
        system_prompt = """
        <s>
        [INST]
        """
        instructions = ""
        for m in messages:
            if m.role == 'system':
                system_prompt = system_prompt + "\n" + m.content
            else:
                instructions = instructions + "\n" + m.content

        return f"<s>[SYSTEM_PROMPT]{system_prompt}[/SYSTEM_PROMPT]\n[INST]\n{instructions}\n[/INST]"


    # This will add the LLM's response to the UI chat log, and also trigger Vox Caster's TTS
    def handle_llm_response( self, llm_response ):
        print( f"Response Received: {llm_response}")
        # self.vc_ui.toggle_processing_indicator()
        self.vc_ui.toggle_audio_switch()
        self.vc_ui.add_to_log( f"AI Assistant: {llm_response}" )
        self.vc_audio.generate_voice_response( llm_response )

    # When the UI tx butten is pressed, the text from the input box will be passed here
    # If you want to keep the conversation context, you can get the text from the UI conversation log
    def handle_input( self, input_text ):
        self.vc_ui.toggle_audio_switch()
        user_message = Message( 'user', input_text )
        self.vc_ui.toggle_processing_indicator()

        # self.vc_ui.toggle_processing_indicator()

        # If you want to use the UI while the LLM is thinking, it's best to run the LLM call in a separate thread.
        def call_llm():
            llm_response = self.generate_response( [SYS_MESSAGE, user_message] )
            self.handle_llm_response( llm_response )
            self.vc_ui.toggle_processing_indicator()
        
        llm_thread = Thread(target=call_llm)
        llm_thread.start()
    
    # When the audio switch in the UI is changed, resume or pause audio recording accordingly
    def handle_audio_switch( self, switch_state ):
        if switch_state == True:
            self.vc_audio.resume_input_stream()
        else:
            self.vc_audio.pause_input_stream()


if __name__ == "__main__":

    def handle_audio_switch( switch_state ):
        llm.handle_audio_switch( switch_state )
    
    def handle_input( input ):
        llm.handle_input( input )

    # dictionary of callbacks that the ui will use when there is interaction with widgets
    my_ui_callbacks = {
        "audio_switch_callback": handle_audio_switch,
        "tx_button_callback": handle_input
    }

    vox_ui = VoxCasterUI( ui_callbacks=my_ui_callbacks )
    vc = VoxCaster( vox_ui )
    llm = LocalAiClient( vox_ui, vc )

    t = Thread( target=vc.record_and_transcribe )
    try:
        t.start()
        vox_ui.run()
        t.join()
    except KeyboardInterrupt:
        t.join(2)
        sys.exit('\nExit by user')