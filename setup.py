from setuptools import setup, find_packages

setup(
    name='pathwas',
    version='0.2.0',
    description='Toolbox for pathway-level association studies',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Yousef Mustafa',
    author_email='',
    url='https://github.com/yousef-mustafa/PathWAS',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0,<2.0.0',
        'pandas>=1.3.0,<3.0.0',
        'scipy>=1.7.0,<2.0.0',
        'scikit-allel>=1.3.5,<2.0.0',
        'mygene>=3.2.2,<4.0.0',
        'gseapy>=1.0.0,<2.0.0',
        'pyyaml>=6.0,<7.0',
        'markdown>=3.4.0,<4.0.0',
    ],
    extras_require={
        'robust': ['astropy>=5.0,<7.0'],
        'viz': ['matplotlib>=3.5.0,<4.0.0', 'seaborn>=0.12.0,<1.0.0'],
        'dev': ['pytest>=7.0.0', 'pytest-cov>=4.0.0'],
        'all': [
            'astropy>=5.0,<7.0',
            'matplotlib>=3.5.0,<4.0.0',
            'seaborn>=0.12.0,<1.0.0',
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'pathwas=pathwas.cli:main'
        ]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)
