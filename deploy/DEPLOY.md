# Deploying to bixops-prod-1

No pipeline. The repo is cloned on the droplet and run by systemd as `deploy`.

What is already on the droplet:

- `kit-pg`: pgvector Postgres on 127.0.0.1:5433, database `kit_assistant`, with the `vector` extension.
- `/home/deploy/kit-assistant.env`: production settings and keys (deploy, mode 600). It is never in git.
- `/etc/systemd/system/kit-assistant.service` (copy of `kit-assistant.service` here): enabled, not started yet.
- The Caddy block for `kit.bixops.in` (copy of `kit.bixops.in.caddy` here), proxying to 127.0.0.1:5210.

## First deploy

```sh
ssh -i ~/.ssh/id_bixops_droplet -o IdentitiesOnly=yes root@64.227.132.134
sudo -u deploy -H bash -lc 'cd ~ && git clone https://github.com/<owner>/kit-voice-assistant.git && cd kit-voice-assistant && uv sync --no-dev'
systemctl restart kit-assistant
curl -s http://127.0.0.1:5210/healthz
```

If the env file changes, copy it from your machine over ssh stdin:

```sh
ssh -i ~/.ssh/id_bixops_droplet -o IdentitiesOnly=yes root@64.227.132.134 \
  'umask 077; cat > /home/deploy/kit-assistant.env && chown deploy:deploy /home/deploy/kit-assistant.env' < prod.env
```

## Update

```sh
ssh -i ~/.ssh/id_bixops_droplet -o IdentitiesOnly=yes root@64.227.132.134
sudo -u deploy -H bash -lc 'cd ~/kit-voice-assistant && git pull && uv sync --no-dev'
systemctl restart kit-assistant
curl -s http://127.0.0.1:5210/healthz
```

Logs: `journalctl -u kit-assistant -f`.

## Solo Lite and rollback

The Solo Lite Rails API container was stopped to free memory. `bash /root/restore-bixops.sh`
(copy: `restore-bixops.sh` here) starts it again. That script's header lists how to remove the KIT pieces.
