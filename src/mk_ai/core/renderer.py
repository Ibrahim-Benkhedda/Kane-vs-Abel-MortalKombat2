import pyglet
from .env_model import EnvModel

# =============================================================================
# Renderer (Responsible for Rendering the View)
# =============================================================================

class Renderer:
    """
    Responsible for rendering the game state using pyglet.
    """
    def __init__(self, window: pyglet.window.Window, env_model: EnvModel) -> None:
        """
        Initialize the Renderer with a pyglet Window and an EnvModel.

        Parameters:
            window (pyglet.window.Window): The window in which rendering will occur.
            env_model (EnvModel): The model encapsulating the environment's state and logic.
        """
        self.window = window
        self.game_model = env_model

    def render(self):
        """
        Render the current game state. If original_obs is found (an RGB numpy array),
        its centered and scaled within the window dimensions.

        - Clears the window.
        - Calculates scaling to maintain the aspect ratio.
        - Converts the raw numpy data to a pyglet ImageData.
        - Blits (draws) the image at the appropriate offset and scaled size.
        """

        # retrieves the raw (unprocessed) observation from the environment
        original_obs = getattr(self.game_model.env, 'original_obs', None)
        if original_obs is not None:
            # Clear the window before drawing
            self.window.clear()

            # original_obs is expected to have shape: (height, width, 3)
            game_height, game_width = original_obs.shape[0], original_obs.shape[1]

            # Determine scale factors for width and height, then pick the smaller to preserve aspect ratio
            scale_x = self.window.width / game_width
            scale_y = self.window.height / game_height
            scale = min(scale_x, scale_y)

            # Compute the scaled dimensions of the image
            scaled_width = int(game_width * scale)
            scaled_height = int(game_height * scale)

            # Center the image in the window
            x_offset = (self.window.width - scaled_width) // 2
            y_offset = (self.window.height - scaled_height) // 2

            # Create a pyglet image from the numpy array
            img = pyglet.image.ImageData(
                game_width,
                game_height,
                "RGB",
                original_obs.tobytes(),
                pitch=-game_width * 3 # pitch is negative to account for row order differences between pyglet and numpy
            )

            # Draw the image at the calculated offset with the scaled dimensions
            img.blit(x_offset, y_offset, width=scaled_width, height=scaled_height)