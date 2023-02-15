# Resource Management in Cloud Servers Using end-to-end Deep Reinforcement Learning

The importance of resource management in cloud servers cannot be overstated. With a vast array of resources at their disposal, efficient management can help reduce costs, improve performance, and ensure optimal resource utilization. Reinforcement learning has emerged as a promising approach to solving this complex optimization problem, allowing systems to learn how to allocate resources efficiently through experience. By leveraging techniques such as proximal policy optimization and deep reinforcement learning, researchers have been able to achieve impressive results in optimizing resource allocation in cloud environments.

In this project, we simulated the process of a cloud server with python and wrapped it with OpenAI's Gym Wrapper. Then, We utilized Stable-baselines3 framework to train an agent with Proximal Policy Optimization (PPO) algorithm to achieve optimal performance.

## Install prerequisites
```
sudo apt-get update
sudo apt-get install python3.10
python -m pip install pip
pip install stable-baselines3==1.6.2
pip install gym==0.21.0
pip install tensorboard==2.10.1
```

## Usage
```
git clone https://github.com/AhmadrezaHadi/deep-css.git
cd deep-css
```
## Training
```
python main.py train ppo -n MODEL_NAME -t TIMESTEPS -e ENVIRONMENT_ID -c CPU_COUNTS
```
Below is the list of more arguments that can be set:
```
--timesteps <timesteps for training. default=150_000_000>
--name <name of the training model>
--load <model path to load from>
--environment <Environment ID for the training process. default=deepcss-v2>
--iters <iterations for evaluation process>
--cpu <number of cpus to use multiprocessing for the learning process>
--cliprange <Clip range parameter for PPO algorithm>
--batchsize <Mini-batch size for learning process>
--render <Render the output of environment or not. default=False>
```
More parameters related to the simulation process can be set in model_name/parameters.py

## Evaluation
```
python main.py eval -l MODEL_PATH -i ITERATIONS -e ENVIRONMENT_ID
```

## Simulation Details
![Simulation](/assets/images/simulation.png "Simulation")

We have n (=3) Servers and a buffer with size m (=10) for the incoming jobs. At each timestep, a number of jobs arrive to the buffer and are ready to be scheduled by the agent. If the buffer is full, the jobs will be added to the job backlog (default size is 60). The agent only can see the state of servers, buffer and number of jobs in the backlog. The number of incoming jobs at each timestep are sampled from poisson distribution with $\lambda = 3$. The length of each job is sampled from exponential distribution with $\lambda = 0.2$. <br>
 