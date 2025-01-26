# WarpRL
This repository provides a custom **Gym-like reinforcement learning (RL) environment** built using **[NVIDIA Warp](https://github.com/NVIDIA/warp)**, a highly efficient framework for GPU-accelerated simulations. The environment is designed for fast, scalable, and high-performance training of RL agents in physics-based tasks.

## Setup & Running Sample Code
1. Clone the WarpRL repository.
To clone the repository, run the following command: ```git clone git@github.com:makolon/WarpRL.git```

2. Create a `.env` file under the `docker` directory with the following content.
```
###
# General settings
###
# WarpRL version
WARPRL_VERSION=0.1.0
# WarpRL default path
WARPRL_PATH=/workspace/warprl
# Docker user directory - by default this is the root user's home directory
DOCKER_USER_HOME=/root
```

3. Start the Docker container. To start the Docker container, run: ```docker-compose -p warprl_docker run warprl```

4. Install Python packages using uv. Inside the container, install the required Python packages: ```uv sync```

5. Install the pwm package. Still inside the container, install the pwm package with: ```uv pip install -e .```

6. Run the sample code.
You can run a sample code using the following command: ```uv run python -m warp.examples.optim.example_bounce```

7. Run the training code.
To execute the training script, use the following command: ```uv run scripts/train_warp.py env=warp_ant alg=shac```

## License
This software includes components derived from NVIDIA Warp, which are licensed under the NVIDIA Software License Agreement. Users must comply with NVIDIA's licensing terms for any redistribution or modification of this software. 

## Acknowledgments
WarpRL's development has been made possible thanks to these open-source projects:
- [AHAC](https://github.com/imgeorgiev/DiffRL): An Adaptive Horizon Actor-Critic algorithm designed to optimize policies in contact-rich environments by dynamically adjusting model-based horizons, outperforming SHAC and PPO on high-dimensional locomotion tasks.
- [SHAC](https://github.com/NVlabs/DiffRL): A GPU-accelerated policy learning method using differentiable simulation to solve robotic control tasks, enabling faster and more effective learning through parallelized simulation.
- [PWM](https://github.com/imgeorgiev/PWM): A Model-Based RL algorithm leveraging large multi-task world models to efficiently learn continuous control policies with first-order gradients, achieving superior performance on complex locomotion tasks without relying on ground-truth simulation dynamics.
