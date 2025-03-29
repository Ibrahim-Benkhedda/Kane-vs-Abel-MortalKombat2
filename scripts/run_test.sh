python test.py \
    --model_type=DUELINGDDQN \
    --model_path=models/kane/DuellingDDQN_curriculum_16M_VeryEasy_3_Tiers.zip \
    --num_episodes=50 \
    --individual_eval \
    --states "Level1.LiuKangVsJax,VeryEasy.LiuKang-02,VeryEasy.LiuKang-03,VeryEasy.LiuKang-04,VeryEasy.LiuKang-05,VeryEasy.LiuKang-06,VeryEasy.LiuKang-07,VeryEasy.LiuKang-08"
