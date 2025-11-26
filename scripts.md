# Scripts
## Append to `/etc/hosts` to forward URLs:
```
127.0.0.1 kserve-gateway.local 
127.0.0.1 ml-pipeline-ui.local 
127.0.0.1 mlflow-server.local 
127.0.0.1 mlflow-minio-ui.local 
127.0.0.1 mlflow-minio.local 
127.0.0.1 prometheus-server.local 
127.0.0.1 grafana-server.local 
127.0.0.1 evidently-monitor-ui.local 
```
## SSH stuff
Grafana forwarding
```
ssh -N -L 3001:localhost:3001 local-mlops
```
All the web services forwarding
```
ssh -N \
  -L 8080:192.168.1.132:80 \
  -L 8443:192.168.1.132:443 \
  remote-mlops-jump
```