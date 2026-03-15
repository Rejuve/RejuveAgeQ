# Age Estimator — Deployment Guide

Deployment flow: build image → push to AWS ECR → run on EC2.

---

## Prerequisites

- Docker installed locally
- AWS CLI configured
- EC2 key pair (`age-estimator.pem`) for SSH
- ECR repository and EC2 instance in `us-east-1`

---

## 1. Push Image to ECR

From the project root:

```bash
cd ~/../RejuveAgeQ

# Build for linux/amd64 (EC2)
docker build --platform linux/amd64 -t age-estimator .

# Tag for ECR
docker tag age-estimator:latest 782988627614.dkr.ecr.us-east-1.amazonaws.com/age-estimator:latest

# Log in to ECR
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin 782988627614.dkr.ecr.us-east-1.amazonaws.com

# Push
docker push 782988627614.dkr.ecr.us-east-1.amazonaws.com/age-estimator:latest
```

---

## 2. Deploy to EC2

```bash
# SSH into EC2
ssh -i "age-estimator.pem" ec2-user@ec2-98-84-13-73.compute-1.amazonaws.com
```

On the EC2 instance:

```bash
# Ensure Docker is running
sudo systemctl start docker

# Log in to ECR (if not already)
aws ecr get-login-password --region us-east-1 \
  | sudo docker login --username AWS --password-stdin 782988627614.dkr.ecr.us-east-1.amazonaws.com

# Pull latest image
docker pull 782988627614.dkr.ecr.us-east-1.amazonaws.com/age-estimator:latest

# Run (replace existing container if redeploying — see §3)
docker run -d \
  --restart always \
  -p 8000:8000 \
  782988627614.dkr.ecr.us-east-1.amazonaws.com/age-estimator:latest \
  gunicorn app:app --bind 0.0.0.0:8000 --workers 4
```

Service is available at `http://3.234.253.55:8000/predict`.

---

## 3. Stop Previous Container (Redeploy)

```bash
docker ps
docker stop <container_id_or_name>
docker rm <container_id_or_name>
```

Then run the `docker run` command from §2 again.

---

## 4. View Logs

```bash
docker ps
docker logs <container_id_or_name>
```

Follow logs: `docker logs -f <container_id_or_name>`

---

## 5. Clean Up (Remove All Containers & Images)

**Use with caution** — removes all containers, images, volumes, and networks:

```bash
docker system prune -a --volumes -f
```

---

## Quick Reference

| Action           | Command / step                          |
|-----------------|-----------------------------------------|
| Build           | `docker build --platform linux/amd64 -t age-estimator .` |
| Push to ECR     | Tag → ECR login → `docker push`         |
| SSH to EC2      | `ssh -i "age-estimator.pem" ec2-user@ec2-98-84-13-73.compute-1.amazonaws.com` |
| Run app         | `docker run -d --restart always -p 8000:8000 <image> gunicorn app:app --bind 0.0.0.0:8000 --workers 4` |
| Stop container  | `docker stop` → `docker rm`             |
| Logs            | `docker logs <container_id>`            |
| Full cleanup    | `docker system prune -a --volumes -f`   |
