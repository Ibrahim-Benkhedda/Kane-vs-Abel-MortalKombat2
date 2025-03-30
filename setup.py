from setuptools import setup, find_packages

setup(
    name="mk_ai",
    version="1.0.0",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    author="Ibrahim Benkhedda",
    description="AI agents evaluation platform in Mortal Kombat II: Genesis",
    install_requires=[
        "stable_baselines3==2.4.0",
        "stable-retro==0.9.2",
        "gymnasium==1.0.0",
        "optuna==4.1.0", 
        "torch==2.5.1",
        "tensorboard==2.18.0",
    ],
)

