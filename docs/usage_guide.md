# Framework Usage Guide

## Quick Start

### 1. Basic Framework Initialization

```python
from framework_orchestrator import FrameworkOrchestrator

# Initialize the framework
orchestrator = FrameworkOrchestrator()
orchestrator.initialize_framework()
orchestrator.start_framework()

# Check framework status
status = orchestrator.get_framework_status()
print(f"Framework Active: {status.active}")
print(f"Teams Active: {status.teams_active}")
print(f"Performance Score: {status.performance_score}")

# Stop framework
orchestrator.stop_framework()
```

### 2. Team-Specific Operations

#### ARIEL Team Usage
```python
# Access ARIEL team
ariel_team = orchestrator.teams['ariel']

# Check emotional state
emotional_state = ariel_team.emotional_state.get_current_state()
print(f"Current emotion: {emotional_state}")

# Use quantum memory
quantum_memory = ariel_team.quantum_memory_bank
quantum_memory.store_data(data, compression_ratio=1000.0)
```

#### DEBATE Team Usage
```python
# Access DEBATE team
debate_team = orchestrator.teams['debate']

# Start a debate
debate_result = debate_team.start_debate(
    topic="System optimization strategy",
    positions=["Increase memory", "Optimize cores"]
)

# Get consensus
consensus = debate_team.get_consensus_result()
print(f"Consensus reached: {consensus.agreement_level}")
```

#### WARP Team Usage
```python
# Access WARP team
warp_team = orchestrator.teams['warp']

# Check current phase
current_phase = warp_team.get_current_phase()
print(f"Current WARP phase: {current_phase}")

# Trigger optimization
warp_team.optimize_performance()
```

#### HeliX CorteX Team Usage
```python
# Access HeliX CorteX team
helix_team = orchestrator.teams['helix_cortex']

# Monitor system resources
resource_status = helix_team.get_resource_status()
print(f"CPU utilization: {resource_status.cpu_usage}")
print(f"Memory utilization: {resource_status.memory_usage}")

# Run quantum optimization
helix_team.optimize_quantum_circuits()
```

## Advanced Usage

### Memory Management
```python
from core.memory_manager import MemoryManager, TeamType

memory_manager = MemoryManager()

# Allocate memory for ARIEL team
success = memory_manager.allocate_memory(
    team=TeamType.ARIEL,
    base_id=0,
    core_id=5,
    size=16777216  # 16MB
)

# Check memory usage
usage = memory_manager.get_memory_usage(TeamType.ARIEL)
print(f"ARIEL memory utilization: {usage['utilization']}")

# Apply compression (ARIEL specific)
memory_manager.apply_compression(TeamType.ARIEL, compression_ratio=1000.0)
```

### Logic Core Management
```python
from core.logic_cores import LogicCoreManager, Task, TaskPriority

core_manager = LogicCoreManager()

# Submit a task
task = Task(
    task_id="optimization_task_001",
    team=TeamType.WARP,
    priority=TaskPriority.HIGH,
    function=lambda: print("Optimizing performance..."),
    estimated_duration=5.0
)

core_manager.submit_task(task)

# Check core utilization
utilization = core_manager.get_core_utilization(TeamType.WARP)
print(f"WARP team core utilization: {utilization}")
```

### Benchmark Monitoring
```python
# Run system benchmark
benchmark_result = orchestrator.run_system_benchmark()
print(f"Benchmark score: {benchmark_result.overall_score}")
print(f"Target met: {benchmark_result.target_met}")

# Check if neuromorphic evolution should trigger
if benchmark_result.overall_score >= 0.95:
    orchestrator.trigger_neuromorphic_evolution()
```

### Configuration Management
```python
# Load custom configuration
config_manager = orchestrator.config_manager
config_manager.load_configuration('./custom_config.json')

# Update team-specific settings
config_manager.update_team_config('ariel', {
    'compression_ratio': 1500.0,
    'emotional_sensitivity': 0.8
})

# Save configuration
config_manager.save_configuration('./updated_config.json')
```

## Performance Optimization Tips

### 1. Memory Optimization
- Enable compression for ARIEL team to achieve 1000:1 ratios
- Monitor memory usage regularly to prevent leaks
- Use appropriate memory allocation patterns

### 2. Core Utilization
- Balance task distribution across all 96 cores per team
- Use priority queuing for critical tasks
- Monitor core utilization to identify bottlenecks

### 3. Inter-Team Communication
- Minimize unnecessary inter-team messages
- Use asynchronous communication patterns
- Implement proper error handling for team coordination

### 4. Benchmark Achievement
- Monitor benchmark scores continuously
- Optimize team-specific performance metrics
- Trigger neuromorphic evolution when 95% target is consistently met

## Error Handling

### Common Error Patterns
```python
try:
    orchestrator.start_framework()
except MemoryError as e:
    print(f"Insufficient memory: {e}")
    # Reduce memory allocation or enable compression

except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Check and fix configuration files

except TeamCoordinationError as e:
    print(f"Team coordination failed: {e}")
    # Check team health and restart if necessary
```

### Recovery Procedures
```python
# Check team health
health_reports = orchestrator.get_team_health_reports()
for report in health_reports:
    if report.status != "healthy":
        print(f"Team {report.team_name} needs attention: {report.issues}")

        # Attempt recovery
        orchestrator.restart_team(report.team_name)
```

## Monitoring and Logging

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Framework operations will now show detailed logs
orchestrator.start_framework()
```

### Performance Monitoring
```python
# Continuous monitoring
while orchestrator.is_active():
    metrics = orchestrator.get_performance_metrics()
    print(f"System performance: {metrics.overall_performance}")

    if metrics.overall_performance < 0.8:
        print("Performance degradation detected")
        orchestrator.optimize_system()

    time.sleep(60)  # Check every minute
```

## Best Practices

1. **Always initialize the framework before use**
2. **Monitor memory usage to prevent exhaustion**
3. **Use appropriate task priorities for core scheduling**
4. **Implement proper error handling and recovery**
5. **Regular benchmark monitoring for performance validation**
6. **Enable logging for debugging and monitoring**
7. **Use configuration files for environment-specific settings**
8. **Implement graceful shutdown procedures**

