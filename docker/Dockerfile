FROM vnmd/vesselboost_1.0.1

# If building from the GitHub repo, uncomment the below section, open a command
# prompt in the folder containing this Dockerfile and run the command:
#    docker build --no-cache -t kspacekelvin/fire-python ./
# RUN cd /opt/code && \
#     git clone https://github.com/kspaceKelvin/python-ismrmrd-server.git

# If doing local development, use this section to copy local code into Docker
# image. From the folder containing the python-ismrmrd-server repo, uncomment
# the COPY line below and run the command:
#    docker build --no-cache -t fire-python-custom -f python-ismrmrd-server/docker/Dockerfile ./
COPY invertcontrast.py  /opt/code/python-ismrmrd-server

WORKDIR /opt/code/python-ismrmrd-server

# CMD [ "python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-r", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log", "-s", "-S=/tmp/share/saved_data"]
# python3 /opt/code/python-ismrmrd-server/main.py -v -r -H=0.0.0.0 -p=9002 -s -S /tmp/share/saved_data

# Replace the above CMD with this ENTRYPOINT to allow allow "docker stop"
# commands to be passed to the server.  This is useful for deployments, but
# more annoying for development
# ENTRYPOINT [ "python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log"]