# Use an official nvidia runtime as a parent image
FROM tensorflow:base

# CMD ["bash"]

SHELL ["/bin/bash", "-c"]

# nvidia-container-runtime for nvidia-docker2
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# RUN apt update

# this is to be able to use graphics from the container
# Replace 1000 with your user / group id (if needed)
RUN export uid=1000 gid=1000 && \
  mkdir -p /home/developer && \
  mkdir -p /etc/sudoers.d && \
  echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
  echo "developer:x:${uid}:" >> /etc/group && \
  echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
  chmod 0440 /etc/sudoers.d/developer && \
  chown ${uid}:${gid} -R /home/developer && \
  adduser developer sudo

# Set the working directory to $HOME/bonnetal
ENV HOME /home/developer
WORKDIR $HOME


RUN git clone git://github.com/yyuu/pyenv.git $HOME/.pyenv

# ownership of directory
RUN chown -R developer:developer $HOME/.pyenv
RUN chmod 755 $HOME/.pyenv

# # clean the cache
# RUN apt update && \
#   apt autoremove --purge -y && \
#   apt clean -y

# RUN rm -rf /var/lib/apt/lists/*

# user stuff (and env variables)
USER developer
RUN cp /etc/skel/.bashrc $HOME/ && \
  echo 'export NO_AT_BRIDGE=1' >> $HOME/.bashrc && \
  # echo 'export PYENV_ROOT="$HOME/.pyenv"' >> $HOME/.bashrc && \
  # echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> $HOME/.bashrc && \
  echo 'eval "$(pyenv init -)"' >> $HOME/.bashrc

ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/bin:$PATH

# RUN eval "$(pyenv init -)"
RUN pyenv install 3.6.9 && pyenv global 3.6.9
# RUN python -V
RUN $HOME/.pyenv/versions/3.6.9/bin/pip install gast==0.2.2 tensorflow-gpu numpy keras pandas opencv-python ipython notebook matplotlib
# RUN $HOME/.pyenv/versions/3.6.9/bin/pip install scipy

RUN $HOME/.pyenv/versions/3.6.9/bin/pip install sklearn tqdm pillow pydot
RUN $HOME/.pyenv/versions/3.6.9/bin/pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

ENTRYPOINT ["/bin/bash","-c"]

