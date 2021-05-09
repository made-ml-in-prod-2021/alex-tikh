from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='MADE ML in production course HW1',
    author='Alexander Tikhomirov',
    install_requires=[
        'marshmallow-dataclass==8.4.1',
        'matplotlib==3.3.2',
        'numpy==1.20.1',
        'pandas==1.2.4',
        'PyYAML==5.4.1',
        'scikit-learn==0.24.2',
        'seaborn==0.11.1',
    ],
    license='MIT',
)