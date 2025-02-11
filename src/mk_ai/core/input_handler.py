import pyglet
from mk_ai.configs import P1_KEY_MAP, P2_KEY_MAP
from typing import Set

# =============================================================================
# Input Handler (Handles User Input)
# =============================================================================

class InputHandler:
    """
    Handles user input events via pyglet. Binds to a given `pyglet.window.Window`
    and listens for key press/release events for both Player 1 (P1) and Player 2 (P2).
    
    Attributes:
        window (pyglet.window.Window): The window object that receives input events.
        p1_pressed (Set[int]): A set of keys (pyglet symbols) currently pressed for Player 1.
        p2_pressed (Set[int]): A set of keys (pyglet symbols) currently pressed for Player 2.
    """
    def __init__(self, window: pyglet.window.Window):
        """
        Initializes the InputHandler with a reference to the pyglet window.
        Also sets up the window event handlers.

        Parameters:
            window (pyglet.window.Window): The window from which input events will be captured.
        """
        self.window = window
        self.p1_pressed: Set[int] = set()
        self.p2_pressed: Set[int] = set()
        self.setup_window_events()

    def setup_window_events(self):
        """
        Registers pyglet event handlers for key press and key release.
        Captures which keys are currently pressed for each player.

        The actual event handler methods (`on_key_press` and `on_key_release`)
        are defined as closures inside this method.
        """

        @self.window.event
        def on_key_press(symbol: int, modifiers: int) -> None:
            """
            Pyglet callback for when a key is pressed.

            Parameters:
                symbol (int): The pyglet key symbol.
                modifiers (int): Modifier keys (e.g., SHIFT, CTRL).
            """
            if symbol in P1_KEY_MAP:
                self.p1_pressed.add(symbol)
            elif symbol in P2_KEY_MAP:
                self.p2_pressed.add(symbol)

        @self.window.event
        def on_key_release(symbol: int, modifiers: int) -> None:
            """
            Pyglet callback for when a key is released.

            Args:
                symbol (int): The pyglet key symbol.
                modifiers (int): Modifier keys (e.g. SHIFT, CTRL)
            """
            
            if symbol in P1_KEY_MAP:
                self.p1_pressed.discard(symbol)
            elif symbol in P2_KEY_MAP:
                self.p2_pressed.discard(symbol)