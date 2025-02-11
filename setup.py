from setuptools import setup, find_packages

setup(
    name="mk_ai",
    version="0.9.1",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    author="Ibrahim Benkhedda",
    description="AI agents for Mortal Kombat II: Genesis",
)

