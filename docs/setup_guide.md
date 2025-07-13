# Installation and Setup Guide

## System Requirements

### Minimum Requirements
- **Operating System**: Linux, Windows, or macOS
- **Python**: 3.8 or higher
- **RAM**: 8GB (6GB for framework + 2GB system overhead)
- **CPU**: 4+ cores (8+ recommended)
- **Storage**: 2GB free space (SSD recommended)

### Recommended Requirements
- **RAM**: 16GB or higher
- **CPU**: 8+ cores with high clock speed
- **GPU**: CUDA-compatible GPU for quantum processing
- **Storage**: NVMe SSD for optimal performance

## Installation Steps

### 1. Python Environment Setup
```bash
# Check Python version
python --version  # Should be 3.8+

# Create virtual environment (recommended)
python -m venv agi_framework_env

# Activate virtual environment
# On Linux/macOS:
source agi_framework_env/bin/activate
# On Windows:
agi_framework_env\Scripts\activate
```

### 2. Core Dependencies
```bash
# Install core dependencies
pip install numpy>=1.21.0
pip install scipy>=1.7.0
pip install torch>=1.9.0
pip install psutil>=5.8.0
pip install GPUtil>=1.4.0
```

### 3. Optional Dependencies

#### Quantum Processing (Recommended)
```bash
# For quantum circuit optimization
pip install qiskit>=0.30.0
pip install pennylane>=0.20.0
```

#### Machine Learning (Recommended)
```bash
# For anomaly detection
pip install scikit-learn>=1.0.0
pip install pyod>=0.9.0
```

#### Time Series Analysis (Optional)
```bash
# For predictive analysis
pip install statsmodels>=0.13.0
```

#### Visualization (Optional)
```bash
# For system monitoring
pip install matplotlib>=3.4.0
pip install plotly>=5.0.0
```

### 4. Framework Setup
```bash
# Navigate to framework directory
cd project/final

# Verify configuration files
ls config/
# Should show: system_config.json, ariel_config.json, debate_config.json, warp_config.json, helix_cortex_config.json

# Test framework initialization
python -c "from framework_orchestrator import FrameworkOrchestrator; print('Framework import successful')"
```

## Configuration

### 1. System Configuration
Edit `config/system_config.json` to match your system:

```json
{
  "system": {
    "benchmark_target": 0.95
  },
  "performance": {
    "monitoring_interval": 1.0,
    "benchmark_frequency": 300
  },
  "logging": {
    "level": "INFO",
    "file_enabled": true,
    "console_enabled": true
  }
}
```

### 2. Hardware-Specific Settings

#### For Systems with Limited RAM (8GB)
```json
{
  "memory": {
    "team_memory_size": 1073741824,
    "compression_enabled": true
  }
}
```

#### For Systems without GPU
```json
{
  "performance": {
    "quantum_processing_enabled": false
  }
}
```

### 3. Team-Specific Configuration
Adjust individual team configurations in their respective config files:
- `ariel_config.json`: ARIEL team settings
- `debate_config.json`: DEBATE team settings  
- `warp_config.json`: WARP team settings
- `helix_cortex_config.json`: HeliX CorteX settings

## First Run

### 1. Basic Test
```bash
# Run framework test
python launch_framework.py
```

Expected output:
```
================================================================================
Four-Team AGI Framework
ARIEL | DEBATE | WARP | HeliX CorteX
Version: 1.0.0
================================================================================

Initializing framework...
✅ Framework initialized successfully

Starting framework...
✅ Framework started successfully

📊 Framework Status:
   Active: True
   Teams Active: 4/4
   System Performance: 0.XX
   Memory Utilization: 0.XX
   Core Utilization: 0.XX

🎯 Running initial benchmark...
   Benchmark Score: 0.XXX (Target: 0.950)

🔄 Framework is now operational. Press Ctrl+C to stop.
```

### 2. Verify Team Status
```python
from framework_orchestrator import FrameworkOrchestrator

orchestrator = FrameworkOrchestrator()
orchestrator.initialize_framework()
orchestrator.start_framework()

# Check team health
health_reports = orchestrator.get_team_health_reports()
for report in health_reports:
    print(f"{report.team_name}: {report.status} (Performance: {report.performance_score:.2f})")

orchestrator.stop_framework()
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```
ImportError: No module named 'numpy'
```
**Solution**: Install missing dependencies
```bash
pip install numpy scipy torch psutil GPUtil
```

#### 2. Memory Errors
```
MemoryError: Unable to allocate array
```
**Solution**: Reduce memory allocation or increase system RAM
```json
{
  "memory": {
    "team_memory_size": 536870912
  }
}
```

#### 3. Configuration Errors
```
ConfigurationError: Invalid configuration file
```
**Solution**: Validate JSON syntax
```bash
python -m json.tool config/system_config.json
```

#### 4. Permission Errors
```
PermissionError: [Errno 13] Permission denied: 'framework.log'
```
**Solution**: Ensure write permissions
```bash
chmod 755 project/final
touch framework.log
chmod 644 framework.log
```

### Debug Mode
Enable detailed logging:
```json
{
  "logging": {
    "level": "DEBUG"
  }
}
```

### Performance Issues

#### High CPU Usage
- Reduce monitoring frequency
- Disable unnecessary teams
- Lower benchmark frequency

#### High Memory Usage
- Enable compression
- Reduce team memory allocation
- Monitor for memory leaks

#### Slow Performance
- Check system resources
- Optimize configuration
- Enable performance tracking

## Advanced Configuration

### 1. Custom Team Configuration
Create custom team configurations by modifying team config files:

```json
{
  "team": {
    "priority": "HIGH",
    "memory_size": 1610612736
  },
  "capabilities": {
    "custom_feature": true
  }
}
```

### 2. Benchmark Customization
Adjust benchmark parameters:

```json
{
  "performance": {
    "benchmark_target": 0.90,
    "benchmark_frequency": 600
  }
}
```

### 3. Logging Configuration
Configure detailed logging:

```json
{
  "logging": {
    "level": "DEBUG",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_enabled": true,
    "console_enabled": true,
    "max_file_size": "10MB",
    "backup_count": 5
  }
}
```

## Production Deployment

### 1. System Optimization
- Use SSD storage
- Optimize CPU affinity
- Configure memory limits
- Enable performance monitoring

### 2. Monitoring Setup
- Configure log rotation
- Set up performance alerts
- Monitor resource usage
- Track benchmark scores

### 3. Backup and Recovery
- Backup configuration files
- Monitor system health
- Implement recovery procedures
- Test disaster recovery

## Support and Maintenance

### Regular Maintenance
1. Monitor system performance
2. Update configurations as needed
3. Check for memory leaks
4. Validate benchmark scores
5. Review system logs

### Performance Tuning
1. Analyze performance metrics
2. Optimize resource allocation
3. Adjust team priorities
4. Fine-tune benchmark targets

### Updates and Upgrades
1. Backup current configuration
2. Test updates in development
3. Monitor post-upgrade performance
4. Validate system integrity

---

For additional support, refer to the troubleshooting section in the main documentation or enable debug logging for detailed error information.
