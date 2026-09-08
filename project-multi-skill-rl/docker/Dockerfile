ARG from

FROM ${from}

# Timezone Configuration
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y apt-utils \
                                         lsb-release \
                                         mesa-utils \
                                         gnupg2 \
                                         net-tools \
                                         build-essential \
                                         wget \
                                         unzip \
                                         curl \
                                         git \
                                         mc \
                                         vim \
                                         ffmpeg

RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install pathlib statistics scipy
RUN python3 -m pip install --upgrade pip

#RUN pip3 install pybullet
RUN pip3 install torch torchvision torchaudio
RUN pip3 install stable-baselines3

## GUI tunnel from docker
RUN dpkg --add-architecture i386 && \
    apt-get update && apt-get install -y --no-install-recommends \
        libxau6 libxau6:i386 \
        libxdmcp6 libxdmcp6:i386 \
        libxcb1 libxcb1:i386 \
        libxext6 libxext6:i386 \
        libx11-6 libx11-6:i386 && \
    rm -rf /var/lib/apt/lists/*

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
        ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
        ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compat32,utility

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglvnd0 libglvnd0:i386 \
        libgl1 libgl1:i386 \
        libglx0 libglx0:i386 \
        libegl1 libegl1:i386 \
        libgles2 libgles2:i386 && \
    rm -rf /var/lib/apt/lists/*

COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
# Required for non-glvnd setups.
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# very important libraries :)
RUN apt update && apt install -y python3-opengl
RUN pip3 install pyglet
RUN pip3 install tensorboard
RUN pip3 install wandb
RUN pip3 install optuna
RUN pip3 install plotly
#additional libraries
RUN pip3 install gymnasium-robotics
RUN pip3 install rl_zoo3
RUN pip3 install moviepy
RUN pip3 install mujoco
RUN wget https://github.com/google-deepmind/mujoco/releases/download/3.1.4/mujoco-3.1.4-linux-aarch64.tar.gz && tar -xzvf mujoco-3.1.4-linux-aarch64.tar.gz
RUN pip3 install gpustat
RUN apt-get update && apt-get install -y python3.10-venv
RUN pip3 install --force-reinstall protobuf==3.20.0
