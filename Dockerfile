FROM nvidia/cuda:10.1-runtime
RUN groupadd -g 1001 user && \
    useradd -u 1001 -g 1001 -ms /bin/bash user && \
    mkdir /loss_landscape && \
    chown -R user:user /loss_landscape

RUN apt-get update && apt-get install -y wget

# Get miniconda and its binaries to the PATH
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash ./miniconda.sh -b -p /opt/conda && \
    rm ./miniconda.sh
ENV PATH /opt/conda/bin:$PATH
ENV PYTHONPATH /loss_landscape:$PYTHONPATH

# Create loss_landscape conda environment (like cd loss_landscape)
WORKDIR /loss_landscape
ADD env.yml .
RUN conda env create -f env.yml && conda clean -a -y
RUN conda init bash

# Add the path of the python interpreter (like source activate loss_landscape)
ENV PATH /opt/conda/envs/loss_landscape/bin/:$PATH

# Copy content over
ADD . .
