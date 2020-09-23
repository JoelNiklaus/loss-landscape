import setuptools

setuptools.setup(
    name='loss_landscape',
    version='0.0.6.dev5',
    install_requires=['h5py>=2.10.0', 'numpy>=1.18.4', 'scipy>=1.4.1',
                      'scikit-learn>=0.22.2.post1', 'matplotlib>=3.2.1', 'seaborn>=0.10.1'],
    python_requires='>=3',
    author="Joel Niklaus",
    author_email="me@joelniklaus.ch",
    description="A library for computing loss landscapes for neural networks",
    license='MIT License',
    long_description=open('README.md', "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JoelNiklaus/loss_landscape",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
