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
            'default=app.plugins.predictor_plugin_ann:Plugin',
            'ann=app.plugins.predictor_plugin_ann:Plugin',
            'cnn=app.plugins.predictor_plugin_cnn:Plugin',
            'lstm=app.plugins.predictor_plugin_lstm:Plugin',
            'transformer=app.plugins.predictor_plugin_transformer:Plugin',
            'sarimax=app.plugins.predictor_plugin_sarimax:Plugin',
            'ls_pred_strategy=app.plugins.plugin_long_short_predictions:Plugin'
        ]
    },
    install_requires=[
        
    ],
    author='Harvey Bastidas',
    author_email='your.email@example.com',
    description='A timeseries prediction system that supports dynamic loading of predictor plugins for processing time series data.'
)
