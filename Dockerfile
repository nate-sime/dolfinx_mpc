# This is a Dockerfile used to install dolfinx_mpc.
#
# Author:
# Jørgen S. Dokken <jsd55@cam.ac.uk>

FROM dolfinx/real as dolfinx-mpc-real
LABEL description="DOLFIN-X in real mode with MPC"
USER root

ONBUILD WORKDIR /tmp

ONBUILD RUN git clone https://gitlab.asimov.cfms.org.uk/wp2/mpc.git && \
    cd mpc && \
    mkdir build && \
    cd build && \
    cmake -G Ninja ../cpp && \
    ninja install && \
    cd ../python && \
    pip3 install . && \
    rm -rf /tmp/*

ONBUILD WORKDIR /root


FROM dolfinx/complex as dolfinx-mpc-complex

LABEL description="DOLFIN-X in complex mode with MPC"
USER root

ONBUILD WORKDIR /tmp

ONBUILD RUN git clone https://gitlab.asimov.cfms.org.uk/wp2/mpc.git && \
    cd mpc && \
    mkdir build && \
    cd build && \
    cmake -G Ninja ../cpp && \
    ninja install && \
    cd ../python && \
    pip3 install . && \
    rm -rf /tmp/*

ONBUILD WORKDIR /root
