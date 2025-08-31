from setuptools import setup, find_packages

setup(
    name="soccerchain_wrap",
    version="0.1.0",
    description="Soccer Sequence pattern identification",
    author="Youssef Benallal",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "soccerchain_wrap=soccerchain_wrap.cli:main",
        ],
    },
)