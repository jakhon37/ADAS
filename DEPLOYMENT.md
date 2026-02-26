# ADAS Core - Production Deployment Guide

## Overview

This guide covers production deployment of the ADAS (Advanced Driver Assistance System) Core platform.

## System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (4+ cores recommended)
- **Memory**: Minimum 2GB RAM, 4GB+ recommended
- **GPU** (Optional): NVIDIA GPU with CUDA support for TensorRT acceleration
- **Storage**: 10GB+ available space

### Software Requirements
- Python 3.10 or higher
- Docker 20.10+ (for containerized deployment)
- Linux OS (Ubuntu 20.04+ recommended for production)

## Deployment Options

### Option 1: Docker Deployment (Recommended)

#### Build Container
```bash
docker build -t adas-core:latest .
```

#### Run with Docker Compose
```bash
# Copy example config
cp config.example.json config.json

# Edit configuration as needed
vim config.json

# Start service
docker-compose up -d

# View logs
docker-compose logs -f adas-core

# Stop service
docker-compose down
```

#### Run Standalone Container
```bash
docker run -d \
  --name adas-core \
  -v $(pwd)/config.json:/app/config.json:ro \
  -v $(pwd)/logs:/logs \
  --restart unless-stopped \
  adas-core:latest --frames 1000
```

### Option 2: Native Installation

#### Install Package
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install package
pip install -e .
```

#### Run Service
```bash
# Run with default config
adas-run --frames 100

# Run with custom config
adas-run --config config.json --frames 100

# Run with debug logging
adas-run --log-level DEBUG --frames 50
```

### Option 3: Systemd Service

Create `/etc/systemd/system/adas-core.service`:

```ini
[Unit]
Description=ADAS Core Service
After=network.target

[Service]
Type=simple
User=adas
WorkingDirectory=/opt/adas-core
Environment="PATH=/opt/adas-core/venv/bin"
ExecStart=/opt/adas-core/venv/bin/adas-run --config /etc/adas/config.json
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable adas-core
sudo systemctl start adas-core
sudo systemctl status adas-core
```

## Configuration

### Configuration File Structure

See `config.example.json` for complete configuration options.

### Key Configuration Parameters

#### Detector Configuration
- `confidence_threshold`: Detection confidence threshold (0.0-1.0)
- `iou_threshold`: IoU threshold for NMS (0.0-1.0)

#### Planner Configuration
- `cruise_speed_mps`: Target cruise speed in m/s
- `min_follow_distance_m`: Minimum safe following distance
- `time_gap_s`: Desired time gap to lead vehicle

#### Safety Configuration
- `max_speed_mps`: Maximum allowed speed
- `max_deceleration_mps2`: Maximum deceleration limit
- `min_following_distance_m`: Absolute minimum following distance

### Environment Variables

- `ADAS_LOG_LEVEL`: Override log level (DEBUG, INFO, WARNING, ERROR)
- `ADAS_CONFIG_PATH`: Path to configuration file

## Monitoring and Observability

### Logging

Logs are written to stdout and can be collected via:
- Docker logs: `docker logs adas-core`
- Systemd journal: `journalctl -u adas-core -f`

Log format:
```
YYYY-MM-DD HH:MM:SS - module - LEVEL - message
```

### Health Checks

Docker health check is built-in:
```bash
docker inspect --format='{{.State.Health.Status}}' adas-core
```

### Metrics

Pipeline metrics are logged periodically including:
- Frame processing rate (FPS)
- Detection counts
- Safety events
- Processing latency

## Production Optimizations

### GPU Acceleration

For production deployments with NVIDIA GPUs:

1. Install NVIDIA Container Toolkit
2. Use GPU-enabled base image
3. Replace mock detector with TensorRT implementation

```dockerfile
# Add to Dockerfile
FROM nvcr.io/nvidia/tensorrt:22.12-py3

# Install CUDA-enabled dependencies
RUN pip install tensorrt onnxruntime-gpu
```

Run with GPU:
```bash
docker run --gpus all adas-core:latest
```

### Performance Tuning

1. **Increase FPS**: Adjust `fps` in config
2. **Reduce Latency**: Optimize detector model (quantization, pruning)
3. **Memory**: Adjust tracker `max_missed_frames` to control track memory

### Resource Limits

In production, set resource limits:

```yaml
# docker-compose.yml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 4G
```

## Safety Considerations

⚠️ **IMPORTANT**: This is a reference implementation for development and testing.

For production automotive deployment:
1. Implement ISO 26262 functional safety requirements
2. Add redundancy and fail-safe mechanisms
3. Perform extensive validation and testing
4. Integrate with vehicle CAN bus and safety systems
5. Implement watchdog timers and health monitoring
6. Add data recording for incident analysis

## Troubleshooting

### Common Issues

#### High CPU Usage
- Reduce `fps` in configuration
- Optimize detector model
- Use GPU acceleration

#### Memory Leaks
- Monitor with `docker stats`
- Check tracker pruning settings
- Restart service periodically

#### Missing Detections
- Lower `confidence_threshold`
- Verify camera calibration
- Check lighting conditions

### Debug Mode

Enable debug logging:
```bash
adas-run --log-level DEBUG --frames 10
```

## Backup and Recovery

### Configuration Backup
```bash
# Backup config
cp /etc/adas/config.json /backup/config.json.$(date +%Y%m%d)

# Restore config
cp /backup/config.json.20260226 /etc/adas/config.json
```

### Data Backup
```bash
# Backup logs and data
tar -czf adas-backup-$(date +%Y%m%d).tar.gz logs/ data/
```

## Support and Maintenance

### Updates

```bash
# Pull latest image
docker pull adas-core:latest

# Restart service
docker-compose down
docker-compose up -d
```

### Maintenance Schedule

- **Daily**: Check logs for errors
- **Weekly**: Review metrics and performance
- **Monthly**: Update dependencies and security patches
- **Quarterly**: Full system audit and testing

## License and Compliance

Ensure compliance with:
- Automotive software standards (MISRA, AUTOSAR)
- Data privacy regulations (GDPR, CCPA)
- Safety certifications (ISO 26262)

---

For technical support, see README.md or open an issue on the project repository.
