from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal, Container
from textual.widgets import Static, Button, RichLog, TextArea, Switch, LoadingIndicator
from textual.theme import Theme
from textual.css.query import NoMatches
from threading import Thread

import traceback


guardsman_theme = Theme(
    name="guardsman",
    primary="#D4D4D4",     # Muted parchment/khaki
    secondary="#8B0A1A",   # Imperial Scarlet
    accent="#1A1A1A",      # Near black
    foreground="#D4D4D4",   #Ash Grey
    background="#3B441C", # Dark olive
    success="#4E7540", # from chatGPT
    warning="#c9a227", # Hazzard yellow
    error="#a83c3c",   # Deep Red (battle alert)
    panel="#2a2d29", #from chatGTP
    dark=True,
    variables={
        "border": "#2B351A", # Darker olive
    },
)


class VoxCasterUI(App):
    CSS_PATH = "vox_layout.tcss"

    def __init__( self, ui_callbacks ):
        super().__init__()
        self.ui_callbacks = ui_callbacks

    def compose(self) -> ComposeResult:
        with Vertical(id="sidebar"):
            yield Static( "Sidebar" )
            yield Static(" Rec ", classes="indicator")
            with Horizontal(id="audio_switch_container"):
                yield Static("Audio:  ", classes="label")
                yield Switch(animate=True, value=True, id="audio_switch")
            
            yield Static(" Prcoessing...", id="processing_label")
            
        
        with Vertical(id="main_body"):
            with Horizontal(id="spectro_container", classes="box"):
                yield Static("Spectrogram", id="spectrogram")
            
            with Horizontal(id="convo_container", classes="box"):
                yield RichLog(highlight=True, id="conversation") # markup=True
            
            with Horizontal(id="user_input", classes="box"):
                yield TextArea("Type stuff here", id="user_input_txt")
                yield Button.success("TX", id="send_button")
    
    def on_mount(self) -> None:
        # Register the theme
        self.register_theme(guardsman_theme)  

        # Set the app's theme
        self.theme = "guardsman"  
    
    def on_ready(self) -> None:
        text_log = self.query_one(RichLog)
        text_log.write("[bold magenta]Conversation Goes Here!")

        convo_container = self.query_one("#convo_container")
        convo_container.border_title = "Log"

        h = self.query_one("#user_input")
        h.border_title = "Input"

        sc = self.query_one("#spectro_container")
        sc.border_title = "Vox Spect"

        self.query_one(TextArea).show_line_numbers = True
        self.recording_indicator_on()
    
    def toggle_audio_switch( self ):
        switch = self.query_one( "#audio_switch" )
        switch.value = not switch.value
    
    def on_switch_changed( self, event: Switch.Changed ):
        if event.value == True:
            self.recording_indicator_on()
        else:
            self.recording_indicator_off()

        if "audio_switch_callback" in self.ui_callbacks:
            self.ui_callbacks["audio_switch_callback"]( event.value )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send_button":
            text_input = self.query_one(TextArea)
            self.add_to_log( text_input.text )
            if "tx_button_callback" in self.ui_callbacks:
                t = Thread( target=self.ui_callbacks["tx_button_callback"], args=[text_input.text]  )
                t.start()
            text_input.clear()
    
    def recording_indicator_on( self ):
        rec = self.query_one( ".indicator" )
        rec.add_class( "on" )
    
    def recording_indicator_off( self ):
        rec = self.query_one( ".indicator" )
        rec.remove_class( "on" )
    
    def append_to_input( self, text ):
        text_input = self.query_one(TextArea)
        text_input_end = text_input.document.end

        white_space = ""
        last_char = text_input.text[-1] if len(text_input.text) > 0 else "\n"
        if last_char != "\n":
            white_space = "\n"

        text_input.insert(f"{white_space}{text}", text_input_end)
    
    def add_to_log( self, text ):
        text_log = self.query_one(RichLog)
        text_log.write(f"{text}\n")
    
    def get_log_text( self ):
        text_log = self.query_one(RichLog)
        return text_log.text
    
    def toggle_processing_indicator( self ):
        try:
            traceback.print_stack()
            indicator = self.query_one(LoadingIndicator)
            indicator.remove()
        except NoMatches as e:
            self.mount( LoadingIndicator(), after="#processing_label" )

if __name__ == "__main__":
    app = VoxCasterUI()
    app.run()