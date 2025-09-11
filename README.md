# README


## Notes
```
GIT_SSH_COMMAND='ssh -i /home/buiducnhan/Project/vtscada/.ssh/id_ed25519 -o IdentitiesOnly=yes' git clone git@github.com:beiteman/vtscada.git
```

## Install

### Server

```
cd vtscada
python3 -m venv venv
venv/bin/python -m ensurepip --upgrade
venv/bin/python -m install -r requirements.txt
```
