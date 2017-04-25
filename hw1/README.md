# CS294-112 HW 1: Imitation Learning

Dependencies: TensorFlow, MuJoCo version 1.31, OpenAI Gym

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v1.pkl
* HalfCheetah-v1.pkl
* Hopper-v1.pkl
* Humanoid-v1.pkl
* Reacher-v1.pkl
* Walker2d-v1.pkl

The name of the pickle file corresponds to the name of the gym environment.

## Instruction for running my answer

First, you should read the hw1.pdf to get some knowledge of the prerequisite and the purpose of hw1. 

How to run my answer ?:

1. cd into hw1 and run `run_expert.py` to get the data for behaviour cloning, e.g., `python run_expert.py experts/Hopper-v1.pkl Hopper-v1 --num_rollouts 100`. You can replace Hopper-v1 with other name of the gym environment. You can add `--render` to render the image of running process. The data is written into `expert_data/` file.
2. run `python bc_train Hopper-v1` to train the neuro network of behaviour cloning. 
3. run `python bc_test experts/Hopper-v1.pkl Hopper-v1 behaviour_cloning --render` to see the result of training

run DAgger method:

1. install easydict `pip install easydict`
2. run `python da_train Hopper-v1` to train the NN of DAgger on the base of behaviour cloning. The data of DAgger is stored into '/tmp/DAgger_data'. 
3. run `python bc_test experts/Hopper-v1.pkl Hopper-v1 behaviour_cloning --render` to see the result.

Note: you are expected to run behaviour cloning firstly, and run DAgger after that. Because DAgger method restore the checkpoint of behaviour cloning. 

Some information: the graph of neuro network is constructed in `behaviour_cloning.py`. I only use two hidden layers with 10 nodes in each layers, and only max_steps = 10000 for training. Thus if you want to enhance the result, you can add more hidden layers, more node, or increase max_step in `bc_train.py`.