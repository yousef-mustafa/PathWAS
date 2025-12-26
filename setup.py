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
        'all': ['astropy', 'pytest', 'pytest-cov'],
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
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)