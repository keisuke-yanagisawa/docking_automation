FROM continuumio/miniconda3:4.12.0

RUN apt update
RUN conda install -c conda-forge mamba


# ====== installation of kalasanty ======

# aa912... is the latest commit as of July 7th, 2022
ENV KALASANTY_ROOT /kalasanty
RUN apt install -y libjpeg-dev
RUN git clone https://gitlab.com/cheminfIBB/kalasanty $KALASANTY_ROOT \
    && cd $KALASANTY_ROOT \
    && git checkout aa91525ff6d2540c6b14f560a184c44667eb410c
RUN mamba install -c openbabel -c cheminfIBB -c anaconda -c conda-forge \
        python">=3.5" h5py">=2.7" openbabel">=2.4" tfbio"=0.3" rdkit \
        scikit-image">=0.13" numpy">=1.12" scipy">=1" \
        keras">=2.1,<2.3" tensorflow">=1.8,<=1.11" \
        protobuf"=3.6.1" tqdm gridDataFormats
RUN cd $KALASANTY_ROOT \
    && pip install .


# ====== installation of autodock vina with python bindings ======

# prepare_ligand, prepare_receptor, AutoGrid4.2, AutoSite v1.0 and v1.1
ENV ADFR_ROOT /ADFRsuite-1.0
ENV TMPDIR /tmp/adfrtmp
RUN cd /tmp \
    && wget https://ccsb.scripps.edu/adfr/download/1028/ADFRsuite_Linux-x86_64_1.0_install \
    && chmod +x ADFRsuite_Linux-x86_64_1.0_install \
    && ./ADFRsuite_Linux-x86_64_1.0_install --prefix $TMPDIR \
    && cd $TMPDIR \
    && tar -zxvf ADFRsuite_x86_64Linux_1.0.tar.gz \
    && cd ADFRsuite_x86_64Linux_1.0 \
    && ./install.sh -d $ADFR_ROOT
ENV PATH $PATH:$ADFR_ROOT/bin

# AutoDock Vina and its python bindings
ENV ADVINA_ROOT /AutoDock-Vina
RUN apt install -y build-essential libboost-all-dev swig
RUN git clone https://github.com/ccsb-scripps/AutoDock-Vina $ADVINA_ROOT \
    && cd $ADVINA_ROOT \
    && git checkout v1.2.3
RUN cd $ADVINA_ROOT/build/linux/release \
    && make -j
RUN cd $ADVINA_ROOT/build/python \
    && python setup.py build install
ENV PATH $PATH:$ADVINA_ROOT/build/linux/release
ENV PATH $PATH:$ADVINA_ROOT/example/autodock_scripts

# ====== installation of Gypsum-DL ======
ENV GYPSUMDL_ROOT /gypsum_dl
RUN mamba install mpi4py
RUN git clone https://git.durrantlab.pitt.edu/jdurrant/gypsum_dl.git $GYPSUMDL_ROOT \
    && cd $GYPSUMDL_ROOT \
    && git checkout 1.2.0 

# ====== installation of di
ENV DIMORPHITEDL_ROOT /dimorphite_dl
RUN git clone https://git.durrantlab.pitt.edu/jdurrant/dimorphite_dl.git $DIMORPHITEDL_ROOT \
    && cd $DIMORPHITEDL_ROOT \
    && git checkout 1.2.4

RUN echo "alias ls='ls --color=auto'" >> /etc/bash.bashrc