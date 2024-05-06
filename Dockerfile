# use nvidia image as base image
FROM nvcr.io/nvidia/pytorch:22.07-py3

ARG DEBIAN_FRONTEND=noninteractive

# install dependency
RUN apt-get update && apt-get install -y \
    libgl1 \
    dvipng \
    texlive-latex-extra \
    texlive-fonts-recommended \
    cm-super

# create user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# clone repository
RUN git clone https://github.com/ciass-konstanz/protest-segments ./ --recurse-submodules

CMD ["/bin/bash", "-c", "source scripts/install.sh"]
