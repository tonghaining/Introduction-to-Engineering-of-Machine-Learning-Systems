#!/usr/bin/env bash

echo "==============================="
echo " MLflow Remote Connection Setup"
echo "==============================="
echo ""

read -p "Enter remote host IP: " REMOTE_HOST
read -p "Enter SSH username: " REMOTE_USER
read -p "Enter path to SSH private key (e.g. ~/.ssh/id_rsa): " SSH_KEY

if [ ! -f "$SSH_KEY" ]; then
    echo "ERROR: SSH key not found at $SSH_KEY"
    exit 1
fi

echo ""
echo "Connecting to $REMOTE_USER@$REMOTE_HOST ..."
echo "Forwarding ports:"
echo "  MLflow → localhost:5000"
echo "  MinIO UI → localhost:9001"
echo ""

ssh -i "$SSH_KEY" \
    -L 5000:localhost:5000 \
    -L 9001:localhost:9001 \
    "$REMOTE_USER@$REMOTE_HOST"
