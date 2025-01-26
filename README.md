# WarpRL
This repository provides a custom **Gym-compatible reinforcement learning (RL) environment** built using **[NVIDIA Warp](https://github.com/NVIDIA/warp)**, a highly efficient framework for GPU-accelerated simulations. The environment is designed for fast, scalable, and high-performance training of RL agents in physics-based tasks.

## Setup & Running Training/Inference Code
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
