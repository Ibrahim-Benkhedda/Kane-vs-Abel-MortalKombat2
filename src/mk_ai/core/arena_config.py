from mk_ai.agents import DQNAgent, HumanAgent, DoubleDQN, DuelingDoubleDQN, BTAgent, Agent
from typing import Tuple, Optional, List
from dataclasses import dataclass
from stable_baselines3 import DQN

@dataclass
class AgentConfig:
    """
    Configuration for the individual agent
    """
    agent_type: str # "human", "dqn, double_dqn, dueling_ddqn", "bt"
    model_path: Optional[str] = None 
    bt_file_path: Optional[str] = None
    player_num: int = 1

@dataclass
class ArenaConfig:
    """
    Main configuration container for Arena
    """
    game: str = "MortalKombatII-Genesis"
    state: str = "Level1.LiuKangVsJax.2P"
    players: int = 2
    window_size: Tuple[int, int] = (640, 480)
    fps: int = 60
    p1_agent: AgentConfig = AgentConfig(agent_type="human")
    p2_agent: AgentConfig = AgentConfig(
        agent_type="double_dqn",
        model_path="models/kane/DDQN_4M_with_lrSchedule_frameskip_24_actions"
    )


class AgentFactory: 
    """
    Creates agent instances based on the configuration of Arena and Agent. 
    """

    @staticmethod
    def create(
        config: AgentConfig,
        action_mapping,
        buttons: List[str],
        frame_stack: int = 4
    ) -> Agent:
        if config.agent_type == "human":
            return HumanAgent(
                action_mapping=action_mapping,
                buttons=buttons,
                player_num=config.player_num
            )
        
        elif config.agent_type == "bt":
            return BTAgent(
                buttons=buttons,
                bt_file_path=config.bt_file_path
            )
        
        elif config.agent_type == "dqn":
            return DQNAgent(
                DQN.load(
                    config.model_path,
                    device="cuda"
                ),
                frame_stack=frame_stack
            )

        elif config.agent_type == "double_dqn":
            return DQNAgent(
                DoubleDQN.load(
                    config.model_path,
                    device="cuda"
                ),
                frame_stack=frame_stack
            )
        
        elif config.agent_type == "dueling_ddqn":
            return DQNAgent(
                DuelingDoubleDQN.load(
                    config.model_path,
                    device="cuda"
                ),
                frame_stack=frame_stack
            )
        
        else:
            raise ValueError(f"Unknown agent type: {config.agent_type}")