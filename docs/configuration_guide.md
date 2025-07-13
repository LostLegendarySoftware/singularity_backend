# Configuration Documentation

## Overview

This document provides comprehensive documentation for all configuration files in the Four-Team AGI Framework, including parameter explanations, valid ranges, and optimization recommendations.

## Configuration File Structure

The framework uses five main configuration files:

1. **system_config.json** - Global system parameters
2. **ariel_config.json** - ARIEL team specific settings
3. **debate_config.json** - DEBATE team specific settings
4. **warp_config.json** - WARP team specific settings
5. **helix_cortex_config.json** - HeliX CorteX team specific settings

## System Configuration (system_config.json)

### Current Configuration
```json
{
  "system": {
    "name": "Four-Team AGI Framework",
    "version": "1.0.0",
    "description": "ARIEL, DEBATE, WARP, HeliX CorteX integrated framework",
    "benchmark_target": 0.95,
    "total_memory": "6GB",
    "total_logic_cores": 384,
    "teams_count": 4
  },
  "memory": {
    "team_memory_size": 1610612736,
    "logic_bases_per_team": 6,
    "logic_cores_per_base": 16,
    "logic_cores_per_team": 96,
    "memory_per_base": 268435456,
    "memory_per_core": 16777216,
    "compression_enabled": true,
    "ariel_compression_ratio": 1000.0
  },
  "performance": {
    "monitoring_interval": 1.0,
    "benchmark_frequency": 300,
    "optimization_threshold": 0.8,
    "anomaly_detection_enabled": true,
    "quantum_processing_enabled": true
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_enabled": true,
    "console_enabled": true
  }
}
```

### Parameter Explanations

#### System Section
- **name**: Framework identifier
- **version**: Current framework version
- **description**: Framework description
- **benchmark_target**: Performance target for neuromorphic evolution (0.95 = 95%)
- **total_memory**: Total system memory allocation (6GB across 4 teams)
- **total_logic_cores**: Total processing cores (384 cores, 96 per team)
- **teams_count**: Number of active teams (4: ARIEL, DEBATE, WARP, HeliX CorteX)

#### Memory Section
- **team_memory_size**: Memory per team in bytes (1,610,612,736 = 1.5GB)
- **logic_bases_per_team**: Logic algorithm bases per team (6)
- **logic_cores_per_base**: Processing cores per base (16)
- **logic_cores_per_team**: Total cores per team (96)
- **memory_per_base**: Memory per logic base (268,435,456 = 256MB)
- **memory_per_core**: Memory per logic core (16,777,216 = 16MB)
- **compression_enabled**: Enable memory compression features
- **ariel_compression_ratio**: ARIEL team compression ratio (1000:1)

#### Performance Section
- **monitoring_interval**: Performance monitoring frequency in seconds (1.0)
- **benchmark_frequency**: Benchmark evaluation frequency in seconds (300)
- **optimization_threshold**: Performance threshold for optimization triggers (0.8)
- **anomaly_detection_enabled**: Enable system anomaly detection
- **quantum_processing_enabled**: Enable quantum processing features

#### Logging Section
- **level**: Logging level (INFO, DEBUG, WARNING, ERROR)
- **format**: Log message format string
- **file_enabled**: Enable file logging
- **console_enabled**: Enable console logging

## Team-Specific Configurations

### ARIEL Team Configuration (ariel_config.json)

```json
{
  "team": {
    "name": "ARIEL",
    "full_name": "Advanced Reinforced Incentives & Emotions Learning",
    "memory_size": 1610612736,
    "logic_cores": 96,
    "logic_bases": 6,
    "priority": "HIGH"
  },
  "capabilities": {
    "parameter_capacity": 500000000000000,
    "compression_ratio": 1000.0,
    "quantum_memory_enabled": true,
    "emotional_modeling": true,
    "self_healing": true,
    "governance_rules": true
  },
  "quantum_memory": {
    "size": 1000,
    "compression_active": true,
    "entanglement_enabled": true,
    "optimization_enabled": true
  },
  "emotional_state": {
    "default_values": {
      "joy": 0.5,
      "sadness": 0.5,
      "anger": 0.5,
      "fear": 0.5,
      "surprise": 0.5,
      "disgust": 0.5,
      "trust": 0.5,
      "anticipation": 0.5,
      "confidence": 0.5,
      "curiosity": 0.5,
      "satisfaction": 0.5,
      "frustration": 0.5,
      "stability": 0.5,
      "adaptability": 0.5,
      "resilience": 0.5
    },
    "update_frequency": 1.0,
    "normalization_enabled": true
  },
  "self_healing": {
    "enabled": true,
    "strategies": [
      "memory_corruption",
      "performance_degradation",
      "resource_exhaustion",
      "communication_failure",
      "quantum_decoherence"
    ],
    "detection_threshold": 0.1,
    "healing_timeout": 30.0
  },
  "governance": {
    "rules_enabled": true,
    "default_rules": {
      "memory_usage_limit": {
        "limit": 0.9,
        "action": "throttle"
      },
      "cpu_usage_limit": {
        "limit": 0.95,
        "action": "balance"
      },
      "error_rate_limit": {
        "limit": 0.1,
        "action": "investigate"
      },
      "performance_threshold": {
        "limit": 0.7,
        "action": "optimize"
      }
    },
    "enforcement_enabled": true
  }
}
```

**Key Parameters:**
- **compression_ratio**: Memory compression ratio (1000.0 for 1000:1 compression)
- **emotional_sensitivity**: Emotional state sensitivity (0.0-1.0)
- **quantum_memory_enabled**: Enable quantum-inspired memory features
- **self_healing_enabled**: Enable autonomous error correction
- **parameter_capacity**: Target parameter operation capacity (500T+)

### DEBATE Team Configuration (debate_config.json)

```json
{
  "team": {
    "name": "DEBATE",
    "full_name": "16-Agent Debate System",
    "memory_size": 1610612736,
    "logic_cores": 96,
    "logic_bases": 6,
    "priority": "HIGH"
  },
  "agents": {
    "total_agents": 16,
    "cores_per_agent": 6,
    "role_distribution": {
      "proposer": 3,
      "opponent": 3,
      "analyst": 3,
      "moderator": 2,
      "validator": 2,
      "synthesizer": 3
    }
  },
  "debate_engine": {
    "priority": "HIGH",
    "consensus_threshold": 0.75,
    "max_debate_duration": 300,
    "argument_timeout": 30.0,
    "voting_enabled": true
  },
  "debate_phases": {
    "initialization": {
      "enabled": true,
      "timeout": 10
    },
    "argument_presentation": {
      "enabled": true,
      "timeout": 60
    },
    "cross_examination": {
      "enabled": true,
      "timeout": 90
    },
    "rebuttal": {
      "enabled": true,
      "timeout": 60
    },
    "consensus_building": {
      "enabled": true,
      "timeout": 60
    },
    "final_decision": {
      "enabled": true,
      "timeout": 30
    },
    "validation": {
      "enabled": true,
      "timeout": 30
    }
  },
  "visualization": {
    "priority": "MEDIUM",
    "enabled": true,
    "real_time_updates": true,
    "history_retention": 1000
  },
  "consensus_algorithm": {
    "method": "hybrid",
    "validation_enabled": true,
    "confidence_threshold": 0.7,
    "dissent_tracking": true
  }
}
```

**Key Parameters:**
- **agent_count**: Number of debate agents (16)
- **consensus_threshold**: Minimum agreement level for consensus (0.0-1.0)
- **debate_timeout**: Maximum debate duration in seconds
- **agent_roles**: Distribution of agent roles (Proposer, Opponent, Analyst, etc.)
- **visualization_enabled**: Enable internal debate visualization

### WARP Team Configuration (warp_config.json)

```json
{
  "team": {
    "name": "WARP",
    "full_name": "Optimization and Acceleration",
    "memory_size": 1610612736,
    "logic_cores": 96,
    "logic_bases": 6,
    "priority": "HIGH"
  },
  "warp_phases": {
    "total_phases": 7,
    "phases": {
      "1": {
        "name": "INITIALIZATION",
        "requirements": {
          "min_efficiency": 0.0,
          "required_teams": 1,
          "stability_threshold": 0.5
        }
      },
      "2": {
        "name": "ACCELERATION",
        "requirements": {
          "min_efficiency": 0.6,
          "required_teams": 2,
          "stability_threshold": 0.7
        }
      },
      "3": {
        "name": "LIGHTSPEED",
        "requirements": {
          "min_efficiency": 0.8,
          "required_teams": 3,
          "stability_threshold": 0.8
        }
      },
      "4": {
        "name": "OPTIMIZATION",
        "requirements": {
          "min_efficiency": 0.85,
          "required_teams": 4,
          "stability_threshold": 0.85
        }
      },
      "5": {
        "name": "QUANTUM_LEAP",
        "requirements": {
          "min_efficiency": 0.9,
          "required_teams": 5,
          "stability_threshold": 0.9
        }
      },
      "6": {
        "name": "HYPERDIMENSIONAL_SHIFT",
        "requirements": {
          "min_efficiency": 0.95,
          "required_teams": 6,
          "stability_threshold": 0.95
        }
      },
      "7": {
        "name": "SINGULARITY",
        "requirements": {
          "min_efficiency": 0.99,
          "required_teams": 6,
          "stability_threshold": 0.99
        }
      }
    }
  },
  "warp_teams": {
    "total_teams": 6,
    "teams": [
      "algorithm",
      "learning",
      "memory",
      "emotion",
      "optimization",
      "dimensional"
    ],
    "default_active": [
      "algorithm"
    ],
    "activation_threshold": 0.8
  },
  "optimization": {
    "enabled": true,
    "targets": [
      "efficiency",
      "warp_factor",
      "stability"
    ],
    "optimization_interval": 5.0,
    "improvement_threshold": 0.01
  },
  "quantum_effects": {
    "quantum_fluctuation_enabled": true,
    "initial_quantum_fluctuation": 0.01,
    "quantum_leap_enabled": true,
    "hyperdimensional_shift_enabled": true,
    "singularity_threshold": 0.99
  },
  "performance_tracking": {
    "enabled": true,
    "history_size": 10000,
    "categories": [
      "computational",
      "memory",
      "network",
      "optimization",
      "system"
    ],
    "tracking_interval": 1.0
  }
}
```

**Key Parameters:**
- **phases**: 7-phase acceleration system configuration
- **optimization_targets**: Performance optimization targets
- **phase_transition_thresholds**: Requirements for phase transitions
- **acceleration_enabled**: Enable performance acceleration features
- **optimization_strategies**: Available optimization strategies

### HeliX CorteX Team Configuration (helix_cortex_config.json)

```json
{
  "team": {
    "name": "HeliX CorteX",
    "full_name": "System Hypervisor",
    "memory_size": 1610612736,
    "logic_cores": 96,
    "logic_bases": 6,
    "priority": "CRITICAL"
  },
  "hypervisor": {
    "sampling_rate": 0.1,
    "history_size": 100000,
    "coordination_enabled": true,
    "resource_management_enabled": true,
    "anomaly_detection_enabled": true
  },
  "quantum_processing": {
    "qubits": 8,
    "circuit_optimization_enabled": true,
    "neural_network_enabled": true,
    "quantum_inspired_processing": true
  },
  "resource_management": {
    "cpu_management": true,
    "memory_management": true,
    "gpu_management": true,
    "network_management": true,
    "disk_management": true,
    "quantum_resource_management": true
  },
  "anomaly_detection": {
    "contamination_threshold": 0.01,
    "detection_methods": [
      "dbscan",
      "deep_svdd",
      "statistical"
    ],
    "history_retention": 10000,
    "auto_resolution_enabled": true
  },
  "inter_team_coordination": {
    "enabled": true,
    "message_queue_size": 1000,
    "heartbeat_timeout": 30.0,
    "priority_scheduling": true
  },
  "system_monitoring": {
    "metrics_collection": true,
    "performance_scoring": true,
    "health_monitoring": true,
    "predictive_analysis": true
  }
}
```

**Key Parameters:**
- **quantum_processing**: Quantum circuit optimization settings
- **resource_management**: CPU/GPU resource allocation settings
- **anomaly_detection**: System anomaly detection configuration
- **hypervisor_control**: System coordination settings
- **quantum_qubits**: Number of qubits for quantum processing

## Configuration Management

### Loading Configuration
```python
from framework_orchestrator import ConfigurationManager

config_manager = ConfigurationManager()

# Load system configuration
config_manager.load_system_config('./config/system_config.json')

# Load team configurations
config_manager.load_team_configs('./config/')

# Validate all configurations
validation_result = config_manager.validate_configurations()
if not validation_result.valid:
    print(f"Configuration errors: {validation_result.errors}")
```

### Runtime Configuration Updates
```python
# Update system parameters
config_manager.update_system_config({
    'performance': {
        'monitoring_interval': 0.5,
        'benchmark_frequency': 180
    }
})

# Update team-specific parameters
config_manager.update_team_config('ariel', {
    'compression_ratio': 1500.0,
    'emotional_sensitivity': 0.9
})

# Save updated configuration
config_manager.save_configurations('./config/')
```

## Configuration Validation

### Parameter Ranges and Constraints

#### System Configuration Constraints
- **benchmark_target**: 0.5 - 1.0 (50% - 100%)
- **monitoring_interval**: 0.1 - 10.0 seconds
- **benchmark_frequency**: 60 - 3600 seconds
- **optimization_threshold**: 0.1 - 0.95

#### Memory Configuration Constraints
- **team_memory_size**: Must be divisible by logic_bases_per_team
- **logic_bases_per_team**: 1 - 12 bases
- **logic_cores_per_base**: 1 - 32 cores
- **compression_ratio**: 1.0 - 10000.0

#### Team-Specific Constraints
- **ARIEL compression_ratio**: 100.0 - 10000.0
- **DEBATE agent_count**: 4 - 32 agents
- **WARP phase count**: Must be 7 phases
- **HeliX CorteX quantum_qubits**: 2 - 16 qubits

### Configuration Validation Example
```python
def validate_configuration(config):
    errors = []

    # Validate benchmark target
    if not 0.5 <= config['system']['benchmark_target'] <= 1.0:
        errors.append("benchmark_target must be between 0.5 and 1.0")

    # Validate memory allocation
    total_memory = config['memory']['team_memory_size'] * 4
    if total_memory > 8 * 1024 * 1024 * 1024:  # 8GB limit
        errors.append("Total memory allocation exceeds system limits")

    # Validate core allocation
    total_cores = config['memory']['logic_cores_per_team'] * 4
    if total_cores != config['system']['total_logic_cores']:
        errors.append("Core allocation mismatch")

    return len(errors) == 0, errors
```

## Environment-Specific Configurations

### Development Environment
```json
{
  "system": {
    "benchmark_target": 0.85
  },
  "performance": {
    "monitoring_interval": 2.0,
    "benchmark_frequency": 600
  },
  "logging": {
    "level": "DEBUG",
    "console_enabled": true
  }
}
```

### Production Environment
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
    "file_enabled": true
  }
}
```

### Testing Environment
```json
{
  "system": {
    "benchmark_target": 0.80
  },
  "memory": {
    "team_memory_size": 536870912
  },
  "performance": {
    "monitoring_interval": 5.0,
    "benchmark_frequency": 900
  }
}
```

## Configuration Best Practices

### 1. Memory Configuration
- Ensure team_memory_size is appropriate for available system RAM
- Enable compression for ARIEL team to maximize efficiency
- Monitor memory usage and adjust allocation as needed
- Validate memory per core is sufficient for workloads

### 2. Performance Configuration
- Set monitoring_interval based on system responsiveness needs
- Adjust benchmark_frequency based on stability requirements
- Set optimization_threshold to trigger improvements proactively
- Enable quantum processing if hardware supports it

### 3. Team Configuration
- Configure ARIEL compression ratio based on data characteristics
- Set DEBATE agent count based on problem complexity
- Configure WARP phases based on optimization requirements
- Set HeliX CorteX quantum parameters based on hardware capabilities

### 4. Logging Configuration
- Use DEBUG level for development and troubleshooting
- Use INFO level for production monitoring
- Enable file logging for persistent log storage
- Configure log rotation to manage disk space

## Troubleshooting Configuration Issues

### Common Configuration Problems

#### Memory Allocation Errors
```
Error: Insufficient memory for team allocation
Solution: Reduce team_memory_size or increase system RAM
```

#### Core Allocation Mismatches
```
Error: Core count mismatch between teams and system
Solution: Ensure logic_cores_per_team * 4 = total_logic_cores
```

#### Invalid Parameter Ranges
```
Error: benchmark_target out of valid range
Solution: Set benchmark_target between 0.5 and 1.0
```

### Configuration Debugging
```python
# Enable configuration debugging
import logging
logging.getLogger('configuration').setLevel(logging.DEBUG)

# Validate configuration with detailed output
config_manager = ConfigurationManager()
validation_result = config_manager.validate_configurations(verbose=True)

for error in validation_result.errors:
    print(f"Configuration Error: {error}")

for warning in validation_result.warnings:
    print(f"Configuration Warning: {warning}")
```

## Configuration Migration

### Version Compatibility
The framework supports configuration migration between versions:

```python
# Migrate configuration from older version
migrator = ConfigurationMigrator()
migrated_config = migrator.migrate_from_version('1.0.0', old_config)

# Validate migrated configuration
validation_result = config_manager.validate_configuration(migrated_config)
```

### Backup and Recovery
```python
# Backup current configuration
config_manager.backup_configurations('./config_backup/')

# Restore from backup
config_manager.restore_configurations('./config_backup/')
```

