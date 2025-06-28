from setuptools import setup, find_packages

setup(
    name='RLSkullTrophy',
    version = '0.1',
    packages = find_packages(),
    author = 'Ahmed Ashraf',
    description = 'Reinforcement Learning Game Environment',
    python_requires='>=3.7',
    include_package_data=True,
    package_data={'RLSkullTrophy.data': ['Environment_Variables.pkl']},
)