from setuptools import setup, find_packages

setup(
    name='heuristic_strategy',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'heuristic_strategy=app.main:main'
        ],
        'heuristic_strategy.plugins': [
            'default=app.plugins.plugin_long_short_predictions:Plugin',
            'ls_pred_strategy=app.plugins.plugin_long_short_predictions:Plugin'
        ]
    },
    install_requires=[
        
    ],
    author='Harvey Bastidas',
    author_email='your.email@example.com',
    description='A trading strategy tester with backtrader.'
)
