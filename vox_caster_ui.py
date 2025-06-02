from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal, Container
from textual.widgets import Static, Button, RichLog, TextArea, Switch, LoadingIndicator
from textual.theme import Theme
from textual.css.query import NoMatches
from textual.widgets import Sparkline
from textual.color import Color

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

guardsman_logo = '''
=====  .-.  =====
  === (0.0) ===
   ==  |m|  ==

'''


class VoxCasterUI(App):
    CSS_PATH = "vox_layout.tcss"

    def __init__( self, ui_callbacks = {} ):
        super().__init__()
        self.ui_callbacks = ui_callbacks
        self.ready = False
        self.processing = False

    def compose(self) -> ComposeResult:
        with Vertical(id="sidebar"):
            with Vertical( id="sidebar-1" ):
                yield Static( "Contols", id="sidebar-title" )
                with Horizontal(id="audio_switch_container"):
                    yield Static("Audio:  ", classes="label")
                    yield Switch(animate=True, value=True, id="audio_switch")
                yield Static(" Rec ", classes="indicator")
                yield Static("  ", classes="separator")
                yield Static(" Prcoessing...", id="processing_label")

            with Vertical( id="sidebar-2" ):
                yield Static(guardsman_logo, id="logo")
            
        
        with Vertical(id="main_body"):
            yield Sparkline([3,5,4], id="spectro_container", classes="box", summary_function=max)
            
            with Horizontal(id="convo_container", classes="box"):
                yield RichLog(highlight=True, id="conversation", markup=True) 
            
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
        text_log.write("[bold #D4D4D4]Conversation Goes Here!")

        convo_container = self.query_one("#convo_container")
        convo_container.border_title = "Log"

        h = self.query_one("#user_input")
        h.border_title = "Input"

        sc = self.query_one("#spectro_container")
        sc.border_title = "Vox Spect"

        self.query_one(TextArea).show_line_numbers = True
        self.recording_indicator_on()

        self.ready = True
    
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
        self.processing = not self.processing
        fade_alpha = 1.0 if self.processing else 0.0
        indicator = self.query_one("#processing_label")
        fade_in_color = Color(34, 139, 34, fade_alpha)

        def animate_cb():
            nonlocal fade_in_color
            if fade_in_color.a == 1.0:
                fade_in_color = fade_in_color.with_alpha(0.0)
            else:
                fade_in_color = fade_in_color.with_alpha(1.0)
            
            if self.processing:
                indicator.styles.animate( "background", value=fade_in_color, duration=2.0, on_complete=animate_cb )

        indicator.styles.animate( "background", value=fade_in_color, duration=2.0, on_complete=animate_cb )
    
    def update_spectrogram( self, spec_data ):
        sparkline = self.query_one(Sparkline)
        sparkline.data = spec_data

if __name__ == "__main__":
    app = VoxCasterUI()
    app.run()