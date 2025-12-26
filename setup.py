from setuptools import setup, find_packages

setup(
    name='pathwas',
    version='0.2.0',
    description='Toolbox for pathway-level association studies',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'scikit-allel',
        'mygene',
        'gseapy',
    ],
    extras_require={
        'robust': ['astropy'],
        'dev': ['pytest', 'pytest-cov'],
    },
    entry_points={
        'console_scripts': [
            'pathwas=pathwas.cli:main'
        ]
    },
)