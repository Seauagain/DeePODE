FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu18.04

# Mute the output in nvidia/cuda image
ENTRYPOINT []

CMD ["bash"]

LABEL maintainer=DeePCK author=Joey@sjtu

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install packages for repositories and compilation (g++7.5 cmake)
# Install general packages for OpenFOAM
RUN apt-get update --fix-missing && \
	apt-get install -y vim cmake make ssh wget unzip gcc-7 g++-7 sudo && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7 && \
	apt-get install -y flex libfl-dev bison zlib1g-dev libboost-system-dev \
                       libboost-thread-dev libopenmpi-dev openmpi-bin && \
    rm -rf /var/lib/apt/lists/ && rm -rf /usr/share/doc/ && \
    rm -rf /usr/share/man/ && rm -rf /usr/share/locale/ && \
    apt-get clean

# Download OpenFOAM5.x source code and compile	  
RUN mkdir -p /opt/OpenFOAM && wget -O /opt/OpenFOAM/OFzip https://github.com/Seauagain/OpenFOAM-5.x/archive/refs/tags/version-5.x.tar.gz && \
    cd /opt/OpenFOAM && tar -xvf OFzip && rm OFzip && mv /opt/OpenFOAM/OpenFOAM-5.x-version-5.x /opt/OpenFOAM/OpenFOAM-5.x && \
    wget -O /opt/OpenFOAM/TPzip https://github.com/Seauagain/ThirdParty-5.x/archive/refs/tags/version-5.x.tar.gz && tar -xvf TPzip && rm TPzip && \
    mv /opt/OpenFOAM/ThirdParty-5.x-version-5.x /opt/OpenFOAM/ThirdParty-5.x && \
    cd /opt/OpenFOAM/OpenFOAM-5.x && \
    /bin/bash -c "source etc/bashrc; ./Allwmake -j"

# Source openfoam env in bashrc and set WM_PROJECT_USER_DIR=$HOME
RUN /bin/bash -c "source /opt/OpenFOAM/OpenFOAM-5.x/etc/bashrc" && \
	sudo sed -i 's/\(WM_PROJECT_USER_DIR=$HOME\).*$/\1/g' /opt/OpenFOAM/OpenFOAM-5.x/etc/bashrc

# Install miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/miniconda3 && \
    rm miniconda.sh

# Set PATH for conda
ENV PATH=/opt/miniconda3/bin:$PATH


# Install libtorch1.13.0-cuda11.6 and set ENV
RUN wget https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcu116.zip -O /opt/libtorch.zip && \
    cd /opt && unzip libtorch*.zip && rm libtorch*.zip

ENV LIBTORCH_HOME=/opt/libtorch

ENV LD_LIBRARY_PATH=$LIBTORCH_HOME/lib:$LD_LIBRARY_PATH

# Install Sundials and set ENV
RUN mkdir -p /opt/sundials-2.7 && cd /opt/sundials-2.7 && \
    wget https://github.com/LLNL/sundials/releases/download/v2.7.0/sundials-2.7.0.tar.gz && \
    tar xf sundials-2.7.0.tar.gz && \
    mkdir -p sundials-2.7.0_install && \
    mkdir -p sundials-2.7.0_tmp && \
    cd sundials-2.7.0_tmp && \
    compilerg=`which gcc` && \
    compilergpp=`which g++` && \
    cmake -DCMAKE_C_COMPILER=$compilerg -DCMAKE_CXX_COMPILER=$compilergpp \
        -DBUILD_STATIC_LIBS=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_C_FLAGS='-O3 -DNDEBUG' \
        -DLAPACK_ENABLE=OFF -DSUNDIALS_PRECISION=double -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/sundials-2.7/sundials-2.7.0_install \
        -DEXAMPLES_INSTALL_PATH=/opt/sundials-2.7/sundials-2.7.0_install/examples \
        ../sundials-2.7.0 && \ 
    make  && \ 
    make install

ENV SUNDIALS_HOME=/opt/sundials-2.7/sundials-2.7.0_install

ENV LD_LIBRARY_PATH=$SUNDIALS_HOME/lib:$LD_LIBRARY_PATH

# Set CUDA PATH
ENV CUDA_HOME=/usr/local/cuda

# Allow MPI usage in root mode
ENV OMPI_ALLOW_RUN_AS_ROOT=1

ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Create user: ebi-dnn
ENV user=ebi-dnn uid=6666 
ENV HOME=/home/${user}

# Set sudo right and initial password: 666
RUN useradd -u ${uid} -ms /bin/bash ${user} && \
    echo "${user} ALL=(ALL:ALL) ALL" >> /etc/sudoers && \
    echo "${user}:666" | chpasswd

# Set display color
RUN echo "PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> $HOME/.bashrc

RUN echo "PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> /root/.bashrc

RUN echo "source /opt/OpenFOAM/OpenFOAM-5.x/etc/bashrc" >> $HOME/.bashrc

USER ${uid}

WORKDIR /home/${user}

CMD ["/bin/bash"]