#convert data
# python dicom2mrd.py -o brain.h5 data/dicom_interop/[complete with path to dicoms]

containerName=afib1

docker login
docker build -t kpinecbs/$containerName -f docker/Dockerfile ./


# docker run -it stebo85/$containerName bash
docker run --rm -it --add-host=host.docker.internal:host-gateway -v /tmp:/tmp -v $PWD:/data kpinecbs/$containerName /bin/bash
python3 /opt/code/python-ismrmrd-server/main.py -v -r -H=0.0.0.0 -p=9002 -s -S=/tmp/share/saved_data &


# python client.py -G dataset -o phantom_img.h5 phantom_raw.h5
python client.py -G dataset -o /data/brain_oprenrecon.h5 /data/brain.h5




docker push kpinecbs/$containerName
