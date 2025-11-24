#!/bin/bash
#Connect to vast.ai machine
ssh -p port -i ~/.ssh/id_ed25519 root@your-ip

#Sync files to vast.ai machine
rsync -avz -e "ssh -p port -i ~/.ssh/id_ed25519" \
                                 --exclude='__pycache__' \
                                 --exclude='*.pyc' \
                                 --exclude='checkpoints' \
                                 --exclude='.git' \
                                 --exclude='.venv' \
                                 ./ root@your-ip:/root/BasicGPT/