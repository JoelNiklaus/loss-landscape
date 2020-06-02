import setuptools

setuptools.setup(
    name='loss_landscape',
    version='0.0.1',
    install_requires=['pytorch==1.5.0', 'openmpi==4.0.3', 'mpi4py==3.0.3', 'numpy==1.18.4', 'h5py==2.10.0',
                      'matplotlib==3.2.1', 'scipy==1.4.1', 'scikit-learn==0.22.2.post1', 'seaborn==0.10.1'],
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
