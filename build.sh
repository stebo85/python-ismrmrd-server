#convert data
cd data
tar xf 012-TOF_FCY_500um.tar.gz
cd ..
rm tof.h5
python dicom2mrd.py data/012-TOF_FCY_500um/ -o tof.h5

containerName=vesselboost1

docker build -t $containerName -f docker/Dockerfile ./


# docker run -it $containerName bash
docker run --rm -it --add-host=host.docker.internal:host-gateway -v /tmp:/tmp -v .:/data $containerName
python3 /opt/code/python-ismrmrd-server/main.py -v -r -H=0.0.0.0 -p=9002 -s -S=/tmp/share/saved_data &

rm /data/tof_openrecon.h5
python client.py -G dataset -o /data/tof_oprenrecon.h5 /data/tof.h5

rm /data/tof.nii
cp tof.nii /data

docker tag $containerName stebo85/$containerName
docker tag $containerName kmarshallx/$containerName

docker push stebo85/$containerName
