from setuptools import setup, find_packages

# Install with 'pip install -e .'

setup(
    name="gym_gmazes",
    version="0.1.0",
    author="Nicolas Perrin-Gilbert",
    description="Mazes environments for RL (with OpenAI Gym interface)",
    url="https://github.com/perrin-isir/gym-gmazes",
    packages=find_packages(),
    install_requires=[
        "gym>=0.22.0",
        "torch>=1.10.0",
        "matplotlib>=3.1.3",
    ],
    license="LICENSE",
)
