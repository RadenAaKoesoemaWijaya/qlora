# Production Deployment Guide for QLoRA Application

## Overview

This guide provides comprehensive instructions for deploying the QLoRA fine-tuning platform in a production environment. The application supports distributed training, real-time monitoring, and enterprise-grade security features.

## Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+ recommended) or Windows with WSL2
- **Memory**: Minimum 16GB RAM (32GB+ recommended for large models)
- **Storage**: 100GB+ available disk space
- **Network**: Stable internet connection for model downloads
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3080/4070 or better recommended)

### Software Dependencies
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+
- Git
- NVIDIA Docker Runtime (for GPU support)

## Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd qlora
cp .env.example .env
# Edit .env with your configuration
```

### 2. Environment Configuration
```bash
# Validate environment configuration
python backend/core/environment_validator.py --env-file .env

# Generate template if needed
python backend/core/environment_validator.py --generate-template
```

### 3. Deploy Application
```bash
# Make deployment script executable
chmod +x deploy.sh

# Full setup and deployment
./deploy.sh setup

# Or deploy with monitoring stack
./deploy.sh deploy-monitoring
```

### 4. Access Application
- **Frontend**: http://localhost
- **Backend API**: http://localhost:8000
- **Grafana** (if monitoring enabled): http://localhost:3000 (admin/admin)
- **Prometheus** (if monitoring enabled): http://localhost:9090

## Detailed Deployment Options

### Option 1: Basic Deployment
Deploys core services without monitoring:
```bash
./deploy.sh deploy
```

Services included:
- MongoDB database
- Redis cache
- Backend API (port 8000)
- Frontend (port 80)

### Option 2: Full-Stack Deployment
Deploys combined backend and frontend:
```bash
./deploy.sh deploy-fullstack
```

Access points:
- Application: http://localhost:8080
- API: http://localhost:8081

### Option 3: Deployment with Monitoring
Deploys with Prometheus and Grafana monitoring:
```bash
./deploy.sh deploy-monitoring
```

Additional services:
- Prometheus metrics collection
- Grafana dashboards
- System and application monitoring

## Configuration

### Environment Variables

#### Required Variables
```bash
# Database
DATABASE_URL=mongodb://admin:password@localhost:27017/qlora_db?authSource=admin
MONGO_PASSWORD=your-secure-mongodb-password
REDIS_PASSWORD=your-secure-redis-password

# Security
SECRET_KEY=your-32-character-secret-key-here-change-this-in-production
JWT_SECRET_KEY=your-32-character-jwt-secret-key-here-change-this-in-production
```

#### Optional Variables
```bash
# Application Settings
ENVIRONMENT=production
LOG_LEVEL=INFO
MAX_WORKERS=4

# GPU Configuration
GPU_MEMORY_THRESHOLD=0.9
ENABLE_GPU_MONITORING=true

# ML Platform Integration
WANDB_API_KEY=your-wandb-api-key
HUGGINGFACE_TOKEN=your-huggingface-token
```

### Security Configuration

1. **Change Default Passwords**: Update all default passwords in `.env`
2. **SSL/TLS**: Configure SSL certificates for production
3. **Firewall**: Configure firewall rules for exposed ports
4. **Access Control**: Set up proper user authentication

## GPU Support

### NVIDIA GPU Setup
```bash
# Install NVIDIA Docker Runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Verify GPU Access
```bash
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

## Monitoring and Maintenance

### Health Checks
```bash
# Check service health
./deploy.sh health

# View service status
./deploy.sh status

# Check logs
./deploy.sh logs backend
./deploy.sh logs frontend
```

### Performance Monitoring
- **Grafana Dashboards**: Pre-configured dashboards for system and application metrics
- **Prometheus Metrics**: Real-time performance data collection
- **Custom Alerts**: Set up alerts for critical metrics

### Backup and Recovery

#### Database Backup
```bash
# MongoDB backup
docker-compose exec mongodb mongodump --out /backup --gzip
docker cp qlora-mongodb:/backup ./mongodb-backup-$(date +%Y%m%d)

# MongoDB restore
docker cp ./mongodb-backup-20240101 qlora-mongodb:/backup
docker-compose exec mongodb mongorestore /backup --gzip
```

#### Model and Data Backup
```bash
# Backup models and checkpoints
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/ checkpoints/

# Backup training data
tar -czf data-backup-$(date +%Y%m%d).tar.gz data/
```

## Scaling and Optimization

### Horizontal Scaling
- Deploy multiple backend instances behind load balancer
- Use external MongoDB cluster for database scaling
- Configure Redis cluster for caching

### Performance Optimization
- Adjust `MAX_WORKERS` based on CPU cores
- Configure `GPU_MEMORY_THRESHOLD` for optimal GPU usage
- Enable GPU monitoring for resource tracking
- Use SSD storage for model checkpoints

### Resource Limits
Configure Docker resource limits in `docker-compose.yml`:
```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '2.0'
        reservations:
          memory: 4G
          cpus: '1.0'
```

## Troubleshooting

### Common Issues

1. **GPU Not Detected**
   - Verify NVIDIA Docker Runtime installation
   - Check GPU driver compatibility
   - Ensure GPU memory is available

2. **Database Connection Failed**
   - Verify MongoDB credentials
   - Check network connectivity
   - Review database logs

3. **Out of Memory Errors**
   - Reduce batch size in training configuration
   - Adjust GPU memory threshold
   - Monitor system memory usage

4. **Model Download Issues**
   - Check Hugging Face token validity
   - Verify internet connectivity
   - Review disk space availability

### Log Analysis
```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs backend
docker-compose logs mongodb

# Follow logs in real-time
docker-compose logs -f backend
```

### Performance Debugging
- Use built-in performance monitoring
- Check Grafana dashboards for bottlenecks
- Review application metrics in real-time
- Analyze training logs for optimization opportunities

## Production Checklist

### Pre-Deployment
- [ ] Environment variables configured
- [ ] Security settings updated
- [ ] SSL certificates installed
- [ ] Backup strategy implemented
- [ ] Monitoring configured
- [ ] Health checks verified

### Post-Deployment
- [ ] Application accessible
- [ ] Database connectivity verified
- [ ] GPU acceleration working
- [ ] Monitoring dashboards accessible
- [ ] Log collection functioning
- [ ] Performance metrics tracking

### Security Review
- [ ] Default passwords changed
- [ ] Access controls configured
- [ ] Network security implemented
- [ ] Audit logging enabled
- [ ] Vulnerability scanning completed

## Support and Maintenance

### Regular Maintenance Tasks
- Monitor system performance
- Review security logs
- Update dependencies
- Backup critical data
- Test disaster recovery procedures

### Getting Help
- Check application logs for errors
- Review monitoring dashboards
- Consult troubleshooting guide
- Contact support team if needed

## Advanced Configuration

### Custom Model Repository
Configure custom model repository access:
```bash
# Add to .env
CUSTOM_MODEL_REPO_URL=https://your-model-repo.com
CUSTOM_MODEL_REPO_TOKEN=your-repo-token
```

### External Database
Configure external MongoDB instance:
```bash
# Update DATABASE_URL
DATABASE_URL=mongodb://username:password@external-host:27017/database?authSource=admin
```

### Load Balancer Setup
For production deployments with multiple instances:
```bash
# Configure nginx upstream
upstream qlora_backend {
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}
```

This deployment guide ensures your QLoRA application runs optimally with proper monitoring, security, and scalability for production use.