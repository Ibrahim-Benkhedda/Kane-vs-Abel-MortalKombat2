import argparse
import pyglet

from typing import List, Tuple, Any
from stable_baselines3 import DQN
from mk_ai.agents import DQNAgent, HumanAgent, DoubleDQN, DuelingDoubleDQN, BTAgent, Agent
from mk_ai.core import InputHandler, EnvModel, Renderer, ArenaConfig, AgentConfig, AgentFactory
from mk_ai.wrappers import MultiAgentMkEnvWrapper
from mk_ai.utils import EloManager


# =============================================================================
# Arena (Coordinates Model, View, and Input)
# =============================================================================

class MortalKombatArena:
    """
    Main game controller that ties the model, view, and input handler together.
    """
    def __init__(
        self,
        config: ArenaConfig,
        env_model: EnvModel,
        renderer: Renderer,
        input_handler: InputHandler,
        agent_factory: AgentFactory
    ):
        self.config = config
        self.game_model = env_model
        self.renderer = renderer
        self.input_handler = input_handler

        # Create agents using the factory and dependencies from the game model.
        self.agents: List[Agent] = [
            agent_factory.create(
                self.config.p1_agent,
                self.game_model.env._action_mapping, 
                self.game_model.env.buttons
            ),
            agent_factory.create(
                self.config.p2_agent,
                self.game_model.env._action_mapping,
                self.game_model.env.buttons
            )
        ]

        # Schedule the update loop using pyglet's clock.
        pyglet.clock.schedule_interval(self.update, 1 / self.config.fps)
        # Register the on_draw event to use our renderer.
        self.renderer.window.push_handlers(self.on_draw)
    
    def on_draw(self):
        """Delegate the drawing to the renderer."""
        self.renderer.render()

    def update(self, dt):
        """
        Main update loop:
          - Updates human-agent key states.
          - Retrieves actions from each agent.
          - Steps the game model.
        """
        for idx, agent in enumerate(self.agents):
            if isinstance(agent, HumanAgent):
                if idx == 0:
                    agent.update_keys(self.input_handler.p1_pressed)
                else:
                    agent.update_keys(self.input_handler.p2_pressed)

        actions = [
            agent.select_action(self.game_model.obs, self.game_model.info)
            for agent in self.agents
        ]
        self.game_model.step(actions)

    def run(self):
        pyglet.app.run()
        self.game_model.close()


# =============================================================================
# Command-Line Argument Parsing & Main Entrypoint
# =============================================================================

def parse_args():
    """
    Parse command-line arguments to override default configuration values.
    """
    default_config = ArenaConfig()
    
    parser = argparse.ArgumentParser(description='Mortal Kombat AI Arena')
    
    # Player 1 arguments
    parser.add_argument('--p1-type', 
        choices=['human', 'dqn', 'double_dqn', 'dueling_ddqn', 'bt'],
        default=default_config.p1_agent.agent_type,
        help=f'Agent type for Player 1 (default: {default_config.p1_agent.agent_type})'
    )
    parser.add_argument('--p1-model',
        default=default_config.p1_agent.model_path,
        help=f'Path to model file for Player 1 (default: {default_config.p1_agent.model_path})'
    )
    parser.add_argument('--p1-bt-file',
        type=str,
        default=None,
        help='Path to behavior tree YAML file for Player 1'
    )

    # Player 2 arguments
    parser.add_argument('--p2-type',
        choices=['human', 'dqn', 'double_dqn', 'dueling_ddqn', 'bt'],
        default=default_config.p2_agent.agent_type,
        help=f'Agent type for Player 2 (default: {default_config.p2_agent.agent_type})'
    )
    parser.add_argument('--p2-model',
        default=default_config.p2_agent.model_path,
        help=f'Path to model file for Player 2 (default: {default_config.p2_agent.model_path})'
    )
    parser.add_argument('--p2-bt-file',
        type=str,
        default=None,
        help='Path to behavior tree YAML file for Player 2'
    )

    # Human player arguments
    parser.add_argument('--p1-username',
        type=str,
        default="HumanPlayer1",
        help='Username for Player 1 if human'
    )
    parser.add_argument('--p2-username',
        type=str,
        default="HumanPlayer2",
        help='Username for Player 2 if human'
    )

    # General configuration
    parser.add_argument('--window-size',
        nargs=2,
        type=int,
        default=list(default_config.window_size),
        help=f'Window dimensions (default: {default_config.window_size})'
    )
    parser.add_argument('--fps',
        type=int,
        default=default_config.fps,
        help=f'Target frames per second (default: {default_config.fps})'
    )
    parser.add_argument('--switch',
        action='store_true',
        help='Swap player sides (P1 and P2 positions)'
    )
    

def main():
    # Parse command-line arguments and construct the configuration.
    args = parse_args()

    # if switch is set, swap the player types, models, bt files, and usernames
    if args.switch:
        # Swap agent types
        args.p1_type, args.p2_type = args.p2_type, args.p1_type
        # Swap model paths
        args.p1_model, args.p2_model = args.p2_model, args.p1_model  
        # Swap behavior tree files
        args.p1_bt_file, args.p2_bt_file = args.p2_bt_file, args.p1_bt_file  
        # Swap usernames
        args.p1_username, args.p2_username = args.p2_username, args.p1_username
    
    config = ArenaConfig(
        window_size=tuple(args.window_size),
        fps=args.fps,
        p1_agent=AgentConfig(
            agent_type=args.p1_type,
            model_path=args.p1_model,
            bt_file_path=args.p1_bt_file,
            player_num=1,
        ),
        p2_agent=AgentConfig(
            agent_type=args.p2_type,
            model_path=args.p2_model,
            bt_file_path=args.p2_bt_file,
            player_num=2
        )
    )

    # Dependency injection: create the window, model, renderer, input handler, and controller.
    window = pyglet.window.Window(
        width=config.window_size[0],
        height=config.window_size[1],
        caption="Mortal Kombat AI Arena"
    )

    # Creates game environment wrapper and inject it into EnvModel.
    env_wrapper = MultiAgentMkEnvWrapper(
        game=config.game,
        state=config.state,
        players=config.players,
        render_mode="none",
        record="data/replays/arena"
    )

    # Create the EloManager
    elo_manager = EloManager()

    # Updated agent IDs to distinguish between DQN variants
    def get_agent_id(agent_type, player_num):
        # Return the username for human players
        if agent_type == "human":
            return args.p1_username if player_num == 1 else args.p2_username
        else:
            # Capitalize the agent type for better readability in ELO ratings
            return agent_type.upper()
        
    # Set agent IDs for ELO tracking
    agent_ids = [
        get_agent_id(args.p1_type, 1),
        get_agent_id(args.p2_type, 2)
    ]

    # Create the EnvModel, Renderer, InputHandler, and Controller.
    env_model = EnvModel(env_wrapper, elo_manager, agent_ids)
    renderer = Renderer(window, env_model)
    input_handler = InputHandler(window)
    controller = MortalKombatArena(
        config,
        env_model,
        renderer,
        input_handler,
        AgentFactory()
    )
    
    # Run the application.
    controller.run()

if __name__ == "__main__":
    main()