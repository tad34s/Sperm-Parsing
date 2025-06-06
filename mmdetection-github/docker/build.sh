# sudo docker rm $(sudo docker ps -a -q) ; sudo docker image rm -f $(sudo docker images -aq)
sudo docker build -t sperm-parsing .
