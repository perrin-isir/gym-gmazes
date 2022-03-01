# gym-gmazes
Mazes environments for reinforcement learning (with OpenAI Gym interface)

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