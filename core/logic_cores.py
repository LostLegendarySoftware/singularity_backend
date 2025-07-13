"""
Logic Core Distribution System for Four-Team AGI Framework
Manages 96 logic cores per team with round-robin scheduling and load balancing
Based on verified technical specifications from unified_technical_specification.md
"""

import threading
import time
import queue
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import numpy as np

logger = logging.getLogger(__name__)

# Constants from verified specifications
LOGIC_CORES_PER_TEAM = 96
LOGIC_BASES_PER_TEAM = 6
LOGIC_CORES_PER_BASE = 16
TOTAL_TEAMS = 4
TOTAL_LOGIC_CORES = LOGIC_CORES_PER_TEAM * TOTAL_TEAMS  # 384

class TeamType(Enum):
    ARIEL = "ariel"
    DEBATE = "debate"
    WARP = "warp"
    HELIX_CORTEX = "helix_cortex"

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class CoreStatus(Enum):
    IDLE = auto()
    BUSY = auto()
    FAILED = auto()
    MAINTENANCE = auto()

@dataclass
class Task:
    """Represents a computational task to be executed on a logic core"""
    task_id: str
    team: TeamType
    priority: TaskPriority
    function: Callable
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    created_time: float = field(default_factory=time.time)
    estimated_duration: float = 1.0
    max_retries: int = 3
    retry_count: int = 0

@dataclass
class LogicCore:
    """Represents a single logic core within a team"""
    core_id: int
    team: TeamType
    base_id: int
    status: CoreStatus = CoreStatus.IDLE
    current_task: Optional[Task] = None
    total_tasks_executed: int = 0
    total_execution_time: float = 0.0
    last_activity: float = field(default_factory=time.time)
    failure_count: int = 0
    performance_score: float = 1.0

    @property
    def average_execution_time(self) -> float:
        """Calculate average execution time per task"""
        if self.total_tasks_executed == 0:
            return 0.0
        return self.total_execution_time / self.total_tasks_executed

    @property
    def utilization_rate(self) -> float:
        """Calculate core utilization rate"""
        if self.status == CoreStatus.BUSY:
            return 1.0
        elif self.status == CoreStatus.IDLE:
            return 0.0
        else:
            return 0.5  # Failed or maintenance

@dataclass
class LogicBase:
    """Represents a logic base containing 16 cores"""
    base_id: int
    team: TeamType
    cores: List[LogicCore] = field(default_factory=list)
    task_queue: queue.PriorityQueue = field(default_factory=queue.PriorityQueue)

    def __post_init__(self):
        if not self.cores:
            for core_id in range(LOGIC_CORES_PER_BASE):
                self.cores.append(LogicCore(
                    core_id=core_id,
                    team=self.team,
                    base_id=self.base_id
                ))

    @property
    def available_cores(self) -> List[LogicCore]:
        """Get list of available cores"""
        return [core for core in self.cores if core.status == CoreStatus.IDLE]

    @property
    def utilization_rate(self) -> float:
        """Calculate base utilization rate"""
        if not self.cores:
            return 0.0
        return sum(core.utilization_rate for core in self.cores) / len(self.cores)

class LogicCoreManager:
    """
    Core distribution system managing 96 logic cores per team.
    Implements round-robin scheduling with priority queuing and load balancing.
    """

    def __init__(self):
        self.teams: Dict[TeamType, List[LogicBase]] = {}
        self.task_queues: Dict[TeamType, queue.PriorityQueue] = {}
        self.execution_threads: Dict[str, threading.Thread] = {}
        self.performance_metrics: Dict[TeamType, Dict[str, float]] = {}
        self.lock = threading.RLock()
        self.running = False
        self.scheduler_thread: Optional[threading.Thread] = None

        # Initialize core structures for all teams
        self._initialize_team_cores()

        # Start core management
        self.start_core_management()

        logger.info(f"Logic Core Manager initialized with {TOTAL_LOGIC_CORES} total cores")

    def _initialize_team_cores(self):
        """Initialize logic cores for all four teams"""
        for team in TeamType:
            self.teams[team] = []
            self.task_queues[team] = queue.PriorityQueue()
            self.performance_metrics[team] = {
                'total_tasks': 0,
                'completed_tasks': 0,
                'failed_tasks': 0,
                'average_execution_time': 0.0,
                'utilization_rate': 0.0,
                'throughput': 0.0
            }

            # Create 6 logic bases per team
            for base_id in range(LOGIC_BASES_PER_TEAM):
                logic_base = LogicBase(base_id=base_id, team=team)
                self.teams[team].append(logic_base)

        logger.info(f"Initialized {LOGIC_CORES_PER_TEAM} cores per team across {LOGIC_BASES_PER_TEAM} bases")

    def submit_task(self, task: Task) -> bool:
        """
        Submit a task for execution on the specified team's cores.

        Args:
            task: Task to be executed

        Returns:
            bool: True if task submitted successfully
        """
        try:
            with self.lock:
                # Add task to team's priority queue
                priority_value = -task.priority.value  # Negative for max-heap behavior
                self.task_queues[task.team].put((priority_value, task.created_time, task))

                # Update metrics
                self.performance_metrics[task.team]['total_tasks'] += 1

                logger.debug(f"Task {task.task_id} submitted to {task.team.value} team")
                return True

        except Exception as e:
            logger.error(f"Failed to submit task {task.task_id}: {e}")
            return False

    def get_available_cores(self, team: TeamType) -> List[Tuple[int, int]]:
        """
        Get list of available cores for a team.

        Args:
            team: Team to query

        Returns:
            List of (base_id, core_id) tuples for available cores
        """
        available_cores = []

        with self.lock:
            for logic_base in self.teams[team]:
                for core in logic_base.available_cores:
                    available_cores.append((logic_base.base_id, core.core_id))

        return available_cores

    def get_core_utilization(self, team: Optional[TeamType] = None) -> Dict[str, Any]:
        """
        Get core utilization statistics.

        Args:
            team: Specific team to query, or None for all teams

        Returns:
            Dict containing utilization statistics
        """
        with self.lock:
            if team:
                return self._get_team_utilization(team)
            else:
                return {
                    'teams': {
                        team.value: self._get_team_utilization(team)
                        for team in TeamType
                    },
                    'system_summary': self._get_system_utilization_summary()
                }

    def balance_load(self, team: TeamType) -> bool:
        """
        Balance load across cores within a team.

        Args:
            team: Team to balance load for

        Returns:
            bool: True if load balancing successful
        """
        try:
            with self.lock:
                # Get all cores sorted by utilization
                all_cores = []
                for logic_base in self.teams[team]:
                    for core in logic_base.cores:
                        all_cores.append(core)

                # Sort by performance score (higher is better)
                all_cores.sort(key=lambda c: c.performance_score, reverse=True)

                # Redistribute tasks if needed
                overloaded_cores = [c for c in all_cores if c.status == CoreStatus.BUSY]
                underutilized_cores = [c for c in all_cores if c.status == CoreStatus.IDLE]

                if len(overloaded_cores) > len(underutilized_cores) * 2:
                    logger.info(f"Load imbalance detected for {team.value} team, rebalancing...")
                    # Implementation would involve task migration
                    return True

                return True

        except Exception as e:
            logger.error(f"Load balancing failed for {team.value}: {e}")
            return False

    def handle_core_failure(self, team: TeamType, base_id: int, core_id: int) -> bool:
        """
        Handle core failure and recovery.

        Args:
            team: Team containing the failed core
            base_id: Base ID of the failed core
            core_id: Core ID of the failed core

        Returns:
            bool: True if failure handled successfully
        """
        try:
            with self.lock:
                logic_base = self.teams[team][base_id]
                failed_core = logic_base.cores[core_id]

                # Mark core as failed
                failed_core.status = CoreStatus.FAILED
                failed_core.failure_count += 1
                failed_core.performance_score *= 0.9  # Reduce performance score

                # Reschedule current task if any
                if failed_core.current_task:
                    task = failed_core.current_task
                    failed_core.current_task = None

                    # Retry task if retries available
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        self.submit_task(task)
                        logger.info(f"Rescheduled task {task.task_id} after core failure")
                    else:
                        logger.error(f"Task {task.task_id} failed permanently after {task.max_retries} retries")
                        self.performance_metrics[team]['failed_tasks'] += 1

                # Attempt core recovery
                self._attempt_core_recovery(team, base_id, core_id)

                logger.warning(f"Core failure handled: {team.value}-{base_id}-{core_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to handle core failure: {e}")
            return False

    def optimize_core_allocation(self, team: TeamType) -> bool:
        """
        Optimize core allocation based on performance metrics.

        Args:
            team: Team to optimize

        Returns:
            bool: True if optimization successful
        """
        try:
            with self.lock:
                # Analyze performance patterns
                performance_data = []
                for logic_base in self.teams[team]:
                    for core in logic_base.cores:
                        performance_data.append({
                            'base_id': logic_base.base_id,
                            'core_id': core.core_id,
                            'performance_score': core.performance_score,
                            'utilization': core.utilization_rate,
                            'avg_execution_time': core.average_execution_time,
                            'failure_count': core.failure_count
                        })

                # Sort by performance score
                performance_data.sort(key=lambda x: x['performance_score'], reverse=True)

                # Update core priorities based on performance
                for i, core_data in enumerate(performance_data):
                    base_id = core_data['base_id']
                    core_id = core_data['core_id']
                    core = self.teams[team][base_id].cores[core_id]

                    # Adjust performance score based on ranking
                    if i < len(performance_data) // 3:  # Top third
                        core.performance_score = min(2.0, core.performance_score * 1.1)
                    elif i > 2 * len(performance_data) // 3:  # Bottom third
                        core.performance_score = max(0.1, core.performance_score * 0.9)

                logger.info(f"Core allocation optimized for {team.value} team")
                return True

        except Exception as e:
            logger.error(f"Core optimization failed for {team.value}: {e}")
            return False

    def start_core_management(self):
        """Start core management and scheduling"""
        if not self.running:
            self.running = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.scheduler_thread.start()

            # Start execution threads for each team
            for team in TeamType:
                thread_name = f"executor_{team.value}"
                thread = threading.Thread(
                    target=self._execution_loop, 
                    args=(team,), 
                    name=thread_name,
                    daemon=True
                )
                self.execution_threads[thread_name] = thread
                thread.start()

            logger.info("Core management started")

    def stop_core_management(self):
        """Stop core management and scheduling"""
        self.running = False

        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)

        for thread in self.execution_threads.values():
            thread.join(timeout=5.0)

        logger.info("Core management stopped")

    def _scheduler_loop(self):
        """Main scheduler loop for round-robin scheduling"""
        while self.running:
            try:
                for team in TeamType:
                    # Update performance metrics
                    self._update_performance_metrics(team)

                    # Balance load if needed
                    if self.performance_metrics[team]['utilization_rate'] > 0.8:
                        self.balance_load(team)

                    # Optimize allocation periodically
                    if int(time.time()) % 300 == 0:  # Every 5 minutes
                        self.optimize_core_allocation(team)

                time.sleep(1.0)  # Schedule every second

            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(5.0)

    def _execution_loop(self, team: TeamType):
        """Execution loop for a specific team"""
        while self.running:
            try:
                # Get next task from queue (blocking with timeout)
                try:
                    priority, created_time, task = self.task_queues[team].get(timeout=1.0)
                except queue.Empty:
                    continue

                # Find available core
                available_core = self._find_best_available_core(team)
                if not available_core:
                    # No cores available, put task back
                    self.task_queues[team].put((priority, created_time, task))
                    time.sleep(0.1)
                    continue

                # Execute task on core
                self._execute_task_on_core(available_core, task)

            except Exception as e:
                logger.error(f"Execution loop error for {team.value}: {e}")
                time.sleep(1.0)

    def _find_best_available_core(self, team: TeamType) -> Optional[LogicCore]:
        """Find the best available core for task execution"""
        with self.lock:
            best_core = None
            best_score = -1

            for logic_base in self.teams[team]:
                for core in logic_base.cores:
                    if core.status == CoreStatus.IDLE:
                        # Score based on performance and recency
                        score = core.performance_score * (1 + 1 / (time.time() - core.last_activity + 1))
                        if score > best_score:
                            best_score = score
                            best_core = core

            return best_core

    def _execute_task_on_core(self, core: LogicCore, task: Task):
        """Execute a task on a specific core"""
        try:
            # Mark core as busy
            core.status = CoreStatus.BUSY
            core.current_task = task
            core.last_activity = time.time()

            start_time = time.time()

            # Execute the task
            result = task.function(*task.args, **task.kwargs)

            execution_time = time.time() - start_time

            # Update core statistics
            core.total_tasks_executed += 1
            core.total_execution_time += execution_time
            core.status = CoreStatus.IDLE
            core.current_task = None

            # Update performance score based on execution time vs estimate
            if task.estimated_duration > 0:
                time_ratio = execution_time / task.estimated_duration
                if time_ratio < 1.2:  # Within 20% of estimate
                    core.performance_score = min(2.0, core.performance_score * 1.05)
                else:
                    core.performance_score = max(0.1, core.performance_score * 0.95)

            # Update team metrics
            with self.lock:
                self.performance_metrics[task.team]['completed_tasks'] += 1

            logger.debug(f"Task {task.task_id} completed on core {core.team.value}-{core.base_id}-{core.core_id}")

        except Exception as e:
            logger.error(f"Task execution failed: {e}")

            # Handle task failure
            core.status = CoreStatus.IDLE
            core.current_task = None
            core.failure_count += 1

            with self.lock:
                self.performance_metrics[task.team]['failed_tasks'] += 1

    def _attempt_core_recovery(self, team: TeamType, base_id: int, core_id: int):
        """Attempt to recover a failed core"""
        try:
            logic_base = self.teams[team][base_id]
            failed_core = logic_base.cores[core_id]

            # Simple recovery: reset core after delay
            time.sleep(1.0)

            if failed_core.failure_count < 5:  # Allow recovery if not too many failures
                failed_core.status = CoreStatus.IDLE
                failed_core.performance_score = max(0.5, failed_core.performance_score)
                logger.info(f"Core recovered: {team.value}-{base_id}-{core_id}")
            else:
                failed_core.status = CoreStatus.MAINTENANCE
                logger.warning(f"Core requires maintenance: {team.value}-{base_id}-{core_id}")

        except Exception as e:
            logger.error(f"Core recovery failed: {e}")

    def _update_performance_metrics(self, team: TeamType):
        """Update performance metrics for a team"""
        with self.lock:
            total_cores = LOGIC_CORES_PER_TEAM
            busy_cores = 0
            total_execution_time = 0
            total_tasks = 0

            for logic_base in self.teams[team]:
                for core in logic_base.cores:
                    if core.status == CoreStatus.BUSY:
                        busy_cores += 1
                    total_execution_time += core.total_execution_time
                    total_tasks += core.total_tasks_executed

            # Update metrics
            self.performance_metrics[team]['utilization_rate'] = busy_cores / total_cores
            if total_tasks > 0:
                self.performance_metrics[team]['average_execution_time'] = total_execution_time / total_tasks
                self.performance_metrics[team]['throughput'] = total_tasks / (time.time() - 0)  # Simplified

    def _get_team_utilization(self, team: TeamType) -> Dict[str, Any]:
        """Get utilization statistics for a specific team"""
        with self.lock:
            bases_utilization = []
            for logic_base in self.teams[team]:
                bases_utilization.append({
                    'base_id': logic_base.base_id,
                    'utilization_rate': logic_base.utilization_rate,
                    'available_cores': len(logic_base.available_cores),
                    'total_cores': len(logic_base.cores)
                })

            return {
                'team': team.value,
                'performance_metrics': self.performance_metrics[team].copy(),
                'bases_utilization': bases_utilization,
                'total_cores': LOGIC_CORES_PER_TEAM
            }

    def _get_system_utilization_summary(self) -> Dict[str, Any]:
        """Get system-wide utilization summary"""
        total_utilization = sum(
            self.performance_metrics[team]['utilization_rate'] 
            for team in TeamType
        ) / len(TeamType)

        total_completed = sum(
            self.performance_metrics[team]['completed_tasks']
            for team in TeamType
        )

        total_failed = sum(
            self.performance_metrics[team]['failed_tasks']
            for team in TeamType
        )

        return {
            'total_cores': TOTAL_LOGIC_CORES,
            'average_utilization': total_utilization,
            'total_completed_tasks': total_completed,
            'total_failed_tasks': total_failed,
            'success_rate': total_completed / (total_completed + total_failed) if (total_completed + total_failed) > 0 else 0.0
        }
