from setuptools import setup, find_packages

setup(
    name='heuristic_strategy',
    version='0.1.0',
    packages=find_packages(where='.', include=['app', 'app.*', 'heuristic_strategy_plugins', 'heuristic_strategy_plugins.*']),
    package_dir={'': '.'},
    entry_points={
        'console_scripts': [
            'heuristic_strategy=app.main:main'
        ],
        'heuristic_strategy.plugins': [
            'default=heuristic_strategy_plugins.plugin_long_short_predictions:Plugin',
            'ls_pred_strategy=heuristic_strategy_plugins.plugin_long_short_predictions:Plugin'
        ],
        'heuristic_strategy.optimizer_plugins': [
            'ga_optimizer=app.optimizer:Plugin'
        ]
    },
    install_requires=[
        # your dependencies here
    ],
    author='Harvey Bastidas',
    author_email='your.email@example.com',
    description='A trading strategy tester with backtrader.'
)
