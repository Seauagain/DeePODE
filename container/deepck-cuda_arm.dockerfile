## 指定arm64架构

FROM --platform=arm64 nvcr.io/nvidia/pytorch:22.03-py3

CMD ["bash"]

LABEL maintainer=DeePCK author=Joey@sjtu

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV DEBIAN_FRONTEND=noninteractive

# Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone

# Install packages for repositories and compilation (g++7.5 cmake)
RUN chmod 1777 /tmp && \
    apt-get update --fix-missing && \
    apt-get install -y vim wget ssh rsync sudo && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt install -y gcc-11 g++-11 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 60 --slave /usr/bin/g++ g++ /usr/bin/g++-11 && \
    apt-get install -y texlive-base texlive-latex-base dvipng texlive-latex-extra texlive-fonts-recommended cm-super fontconfig && \
    rm -rf /var/lib/apt/lists/ && rm -rf /usr/share/doc/ && \
    rm -rf /usr/share/man/ && rm -rf /usr/share/locale/ && \
    apt-get clean


# Install python library
RUN conda install numpy matplotlib seaborn scikit-learn pandas -y && \
    conda install --channel cantera/label/dev cantera -y && \
    pip install easydict scienceplots meshio -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN conda install -c conda-forge mpi4py openmpi -y

ENV MPLCONFIGDIR=/tmp/matplotlib

RUN mkdir -p /var/cache/fontconfig

RUN chmod 755 /var/cache/fontconfig
    

# Allow MPI usage in root mode
ENV OMPI_ALLOW_RUN_AS_ROOT=1

ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

ENV OMPI_MCA_opal_cuda_support=true


# Create user: ebi-dnn
ENV user=deepck uid=6666 
ENV HOME=/home/${user}

# Set sudo right and initial password: 666
RUN useradd -u ${uid} -ms /bin/bash ${user} && \
    echo "${user} ALL=(ALL:ALL) ALL" >> /etc/sudoers && \
    echo "${user}:666" | chpasswd

# Set display color
RUN echo "PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> $HOME/.bashrc

RUN echo "PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> /root/.bashrc


USER ${uid}

WORKDIR /home/${user}

CMD ["/bin/bash"]