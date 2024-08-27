#convert data
python dicom2mrd.py data/014-TOF_FCY_500um/ -o tof.h5

containerName=vesselboost1

docker build -t stebo85/$containerName -f docker/Dockerfile ./


# docker run -it stebo85/$containerName bash
docker run --rm -it --add-host=host.docker.internal:host-gateway -v /tmp:/tmp -v /home/ubuntu/github/python-ismrmrd-server:/data stebo85/$containerName /bin/bash
python3 /opt/code/python-ismrmrd-server/main.py -v -r -H=0.0.0.0 -p=9002 -s -S=/tmp/share/saved_data &


# python client.py -G dataset -o phantom_img.h5 phantom_raw.h5
python client.py -G dataset -o /data/tof_oprenrecon.h5 /data/tof.h5




docker push stebo85/$containerName
