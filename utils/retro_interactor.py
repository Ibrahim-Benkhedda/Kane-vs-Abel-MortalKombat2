import retro
import pyglet
import gzip
from pyglet.window import key as keycodes

# Define key-to-action mapping
KEY_MAP = {
    keycodes.UP: "UP",
    keycodes.DOWN: "DOWN",
    keycodes.LEFT: "LEFT",
    keycodes.RIGHT: "RIGHT",
    keycodes.Z: "A",
    keycodes.X: "B",
    keycodes.C: "C",
    keycodes.ENTER: "START",
}

def keys_to_action(keys, buttons):
    """Convert pressed keys to action array."""
    action = [False] * len(buttons)
    for key in keys:
        if key in KEY_MAP:
            button = KEY_MAP[key]
            if button in buttons:
                action[buttons.index(button)] = True
    return action

env = retro.make(game="MortalKombatII-Genesis", state=None, render_mode="rgb_array", players=2)
buttons = env.buttons
obs = env.reset()

window = pyglet.window.Window(width=640, height=480)
pressed_keys = set()

@window.event
def on_key_press(symbol, modifiers):
    """Handle key presses."""
    pressed_keys.add(symbol)

@window.event
def on_key_release(symbol, modifiers):
    """Handle key releases."""
    pressed_keys.discard(symbol)

@window.event
def on_draw():
    """Render the game scaled to fit the window."""
    global obs
    if isinstance(obs, tuple):  # Handle tuple outputs
        obs = obs[0]

    window.clear()

    # Calculate scaling factors to fit the image into the window
    game_width = obs.shape[1]
    game_height = obs.shape[0]

    scale_x = window.width / game_width
    scale_y = window.height / game_height
    scale = min(scale_x, scale_y)  # Preserve aspect ratio

    scaled_width = int(game_width * scale)
    scaled_height = int(game_height * scale)

    # Center the image in the window
    x_offset = (window.width - scaled_width) // 2
    y_offset = (window.height - scaled_height) // 2

    img = pyglet.image.ImageData(
        game_width, game_height, "RGB", obs.tobytes(), pitch=-game_width * 3
    )

    # Draw the image scaled and centered
    img.blit(x_offset, y_offset, width=scaled_width, height=scaled_height)

def save_state(env, filename):
    """Save the current state of the environment to a gzipped file."""
    state = env.em.get_state()  # Get the current state as a byte string
    with gzip.open(f"{filename}.state", "wb") as f:
        f.write(state)  # Write the state to a gzipped file
    print(f"State saved to {filename}.state")

def update(dt):
    """Game update loop."""
    global obs
    action = keys_to_action(pressed_keys, buttons)
    obs, reward, done, truncated, info = env.step(action)

    if keycodes.F5 in pressed_keys:
        save_state(env, "saved_state.state")

    if done:
        obs = env.reset()

pyglet.clock.schedule_interval(update, 1 / 60)  # 60 FPS
pyglet.app.run()

env.close()
