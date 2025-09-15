# README


## Notes
```
GIT_SSH_COMMAND='ssh -i /home/buiducnhan/Project/vtscada/.ssh/id_ed25519 -o IdentitiesOnly=yes' git clone git@github.com:beiteman/vtscada.git
```

### Versions

Node v20.19.4
NPM 10.8.2
Python 3.12.3 or 3.10.10


## Server

### Install

```
python3 -m venv venv
venv/bin/python -m ensurepip --upgrade
venv/bin/python -m pip install -r requirements.txt
ngrok config add-authtoken 32XI5ZutjH6CWZIcWuEFFMojuUO_2wB6UCwRVFsyCEBMAKY4V
```

### Run

Server
```
cd server
rm -rf ngrok.log && ngrok tcp 2087 --log=stdout > ngrok.log &
CUDA_VISIBLE_DEVICES=4 ../venv/bin/python server.py
```

## Client

### Install

```
npm install
npm run compile
```

