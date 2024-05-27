docker build -t stebo85/niimath-test -f docker/Dockerfile ./
docker run -it stebo85/niimath-test bash
docker push stebo85/niimath-test
