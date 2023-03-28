# gym-gmazes
Maze environments with Dubins' car dynamics for reinforcement learning, with Gym(nasium) interface.

See this [*xpag*](https://github.com/perrin-isir/xpag) tutorial for an example of policy 
training in a gym-gmazes goal-based environment (GMazeGoalDubins-v0):  
[https://colab.research.google.com/github/perrin-isir/xpag-tutorials/blob/main/train_gmazes.ipynb](https://colab.research.google.com/github/perrin-isir/xpag-tutorials/blob/main/train_gmazes.ipynb)

## Installation

<details><summary>Option 1: pip</summary>
<p>

    pip install git+https://github.com/perrin-isir/gym-gmazes

</p>
</details>

<details><summary>Option 2: conda</summary>
<p>

    git clone https://github.com/perrin-isir/gym-gmazes.git
    cd gym-gmazes

Choose a conda environmnent name, for instance `gmazeenv`.  
The following command creates the `gmazeenv` environment with the requirements listed in [environment.yaml](environment.yaml):

    conda env create --name gmazeenv --file environment.yaml

If you prefer to update an existing environment (`existing_env`):

    conda env update --name existing_env --file environment.yml

To activate the `gmazeenv` environment:

    conda activate gmazeenv

Finally, to install the *gym-gmazes* library in the activated virtual environment:

    pip install -e .

</p>
</details>

Once the installation is complete, you can import the environment in python with:  
```import gym_gmazes```  
This directly registers the environments *GMazeDubins-v0* and *GMazeGoalDubins-v0* in gymnasium.
