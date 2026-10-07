#!/usr/bin/env bash
# Run ON the EC2 server, from the project folder (~/app):
#   S3_BUCKET=your-bucket-name bash deploy/setup_ec2.sh
set -euo pipefail

: "${S3_BUCKET:?Set S3_BUCKET, e.g. S3_BUCKET=xai-med-mlflow-artifacts bash deploy/setup_ec2.sh}"
APP_DIR="$(pwd)"
RUN_USER="$(whoami)"

echo "== Installing system packages =="
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip nginx apache2-utils libgomp1 unzip

echo "== Creating Python environment (this takes several minutes) =="
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements-ec2.txt

mkdir -p "$HOME/mlflow-data"

echo "== Installing services =="
for svc in xai-mlflow xai-api xai-streamlit; do
  sed "s|__USER__|$RUN_USER|g; s|__APP_DIR__|$APP_DIR|g; s|__S3_BUCKET__|$S3_BUCKET|g" \
    "deploy/$svc.service" | sudo tee "/etc/systemd/system/$svc.service" >/dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable --now xai-mlflow

echo "== Configuring nginx (you will be asked to choose a UI password) =="
sudo htpasswd -c /etc/nginx/.htpasswd admin
sudo cp deploy/nginx-xai.conf /etc/nginx/sites-available/xai
sudo ln -sf /etc/nginx/sites-available/xai /etc/nginx/sites-enabled/xai
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

echo
echo "Done. MLflow is running. Next:"
echo "  1) MLFLOW_TRACKING_URI=http://127.0.0.1:5000 ./venv/bin/python register_models.py"
echo "  2) sudo systemctl enable --now xai-api xai-streamlit"
