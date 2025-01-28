# FlappyBird_RL

This project is an implementation of **Reinforcement Learning (RL)** applied to the classic **Flappy Bird** game. The goal is to train an agent (a neural network) to play the game and make decisions like whether the bird should flap or not in order to avoid obstacles and maximize the score.

## Files

### `FlappyBirdRL.py`
This file contains the game environment. It defines the `FlappyBirdGame` class that simulates the game, including mechanics such as bird movement, pipe spawning, collision detection, and score calculation. This file also handles the game resets and the game's visual assets.

### `Agent.py`
The agent responsible for training the Reinforcement Learning model. It defines the `Agent` class, which contains functions for memory, training, and making decisions (actions) based on the state of the game. It uses a deep Q-learning approach with a neural network model (`Linear_QNet`).

### `flappy-bird.py`
This file is the main script for running the game with the RL agent. It initializes the game, runs the training loop, and updates the agent’s behavior over time.

### `model.py`
Defines the `Linear_QNet` and `QTrainer` classes. These classes are responsible for creating and training the neural network that powers the RL agent's decision-making process.

### `helper.py`
Contains utility functions such as plotting the scores over time, which helps visualize the agent's performance during training.

## Game Description

In the game, the bird moves upwards when it flaps and falls due to gravity. The goal of the RL agent is to decide whether to flap or not at each time step in order to avoid colliding with pipes and maximize the score.

### State Representation
The game state is represented as a feature vector with the following elements:
- Bird's vertical position
- Bird's vertical velocity
- Distance to the nearest pipe
- Position of the top and bottom of the nearest pipe
- Gap center of the pipe
- Bird's position relative to the gap
- Score
- Distance to the nearest pipe
- Bird's distance from the top and bottom of the screen

### Action Space
The action space consists of:
- **Action 0**: Do nothing
- **Action 1**: Flap (upward movement)

### Rewards
- Positive rewards are given for staying alive, passing pipes, or moving closer to the pipe gap.
- Negative rewards are given for collisions with pipes or boundaries (top/bottom of the screen).

## Dependencies

- **pygame**: For game rendering and mechanics.
- **torch**: For implementing the neural network.
- **numpy**: For handling arrays and game state data.
- **matplotlib** (optional): For plotting the training scores.

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/FlappyBird_RL.git
   cd FlappyBird_RL
   ```
2.	Install dependencies:

     ```bash
      pip install pygame torch numpy
     ```
3.	Run the base game:

     ```bash
      python flappy-bird.py
    ```

4. Run the RL Version:

     ```bash
     python Agent.py
     ```
## View the Agent’s Progress and Scores

As the agent trains, you can track its progress and performance. The training loop logs the agent's scores and shows the mean scores over time, providing insights into the agent's improvement. The scores are displayed in real-time during training, and you can visualize the training history using the utility functions in the `helper.py` file.

## Training

The agent uses a **Deep Q-Network (DQN)** for training. The agent learns from its experiences by remembering previous states, actions, rewards, and next states. This data is stored in memory and sampled in mini-batches during training. The neural network predicts Q-values for each action based on the current state.

Training involves:
- The agent playing multiple games, making decisions, and learning from the results.
- The agent’s behavior improves over time as the neural network’s weights are updated using backpropagation and a reward-based loss function.

## Saving & Loading Models

The trained model is saved periodically during training whenever the agent achieves a new high score. The model is saved in the `./model/` directory.

To load a previously saved model, ensure that a `model.pth` file exists in the `./model/` directory. The agent will load the model before continuing training.

## License

This project is licensed under the **MIT License**.
