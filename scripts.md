# Scripts
## Add to `~/.ssh/config`
```
Host local-mlops
        HostName 195.148.22.105 # floating ip of the 'local' machine
        user ubuntu
    
Host remote-mlops
        HostName 195.148.23.141 # floating ip of the remote machine
        user ubuntu
    
# if we want to simulate going to the remote machine via the local machine
# though we can just connect to the remote server directly.
Host remote-mlops-jump 
        HostName 195.148.23.141
        User ubuntu
        ProxyJump local-mlops
```

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


Directly forward so that localhost:80 on the remote-mlops machine is routed to 8080
ssh -L [port on your machine]:[address from remote machine's perspective that you want to forward]:[port from remote perspective] [remote machine address from your perspective (or alias)]

```
ssh -N -L 8080:localhost:80 remote-mlops
```

If you want to simulate connecting to remote VIA the 'local' machine
```
ssh -N \
  -L 8080:192.168.1.132:80 \
  -L 8443:192.168.1.132:443 \
  remote-mlops-jump
```