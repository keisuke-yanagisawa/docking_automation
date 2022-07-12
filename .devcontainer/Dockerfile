FROM continuumio/miniconda3:4.12.0

RUN apt update
RUN apt install -y libjpeg-dev

# ====== installation of kalasanty ======

# aa912... is the latest commit as of July 7th, 2022
# kalasanty will be installed in /kalasanty
RUN git clone https://gitlab.com/cheminfIBB/kalasanty \
    && cd kalasanty \
    && git checkout aa91525ff6d2540c6b14f560a184c44667eb410c
RUN conda install -c conda-forge mamba
RUN mamba install -c openbabel -c cheminfIBB -c anaconda -c conda-forge \
        python">=3.5" h5py">=2.7" openbabel">=2.4" tfbio"=0.3" \
        scikit-image">=0.13" numpy">=1.12" scipy">=1" \
        keras">=2.1,<2.3" tensorflow">=1.8,<=1.11" \
        protobuf"=3.6.1" tqdm gridDataFormats
RUN cd /kalasanty \
    && pip install .
ENV KALASANTY_ROOT /kalasanty


# ====== installation of autodock vina with python bindings ======

# prepare_ligand, prepare_receptor, AutoGrid4.2, AutoSite v1.0 and v1.1
RUN cd /tmp \
    && wget https://ccsb.scripps.edu/adfr/download/1028/ADFRsuite_Linux-x86_64_1.0_install \
    && chmod +x ADFRsuite_Linux-x86_64_1.0_install \
    && ./ADFRsuite_Linux-x86_64_1.0_install --prefix /tmp/ADFRsuite-1.0 \
    && cd ADFRsuite-1.0 \
    && tar -zxvf ADFRsuite_x86_64Linux_1.0.tar.gz \
    && cd ADFRsuite_x86_64Linux_1.0 \
    && ./install.sh -d /ADFRsuite-1.0
RUN pip install meeko==0.3.3
ENV PATH $PATH:/ADFRsuite-1.0/bin
# executable file of AutoDock Vina
RUN mkdir -r /autodock_vina \
    && cd /autodock_vina \
    && wget https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.3/vina_1.2.3_linux_x86_64 \
    && chmod +x vina_1.2.3_linux_x86_64 \
    && ln -s vina_1.2.3_linux_x86_64 vinad    
# python bindings
RUN mamba install -c conda-forge swig boost-cpp sphinx sphinx_rtd_theme
RUN pip install vina==1.2.3