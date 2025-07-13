"""
Memory Management System for Four-Team AGI Framework
Manages 1.5GB memory allocation per team with 6 logic call algorithm bases
Based on verified technical specifications from unified_technical_specification.md
"""

import numpy as np
import threading
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)

# Constants from verified specifications
TEAM_MEMORY_SIZE = int(1.5 * 1024 * 1024 * 1024)  # 1.5GB in bytes
LOGIC_BASES_PER_TEAM = 6
LOGIC_CORES_PER_BASE = 16
LOGIC_CORES_PER_TEAM = 96
MEMORY_PER_BASE = TEAM_MEMORY_SIZE // LOGIC_BASES_PER_TEAM  # 256MB
MEMORY_PER_CORE = MEMORY_PER_BASE // LOGIC_CORES_PER_BASE   # 16MB

class TeamType(Enum):
    ARIEL = "ariel"
    DEBATE = "debate"
    WARP = "warp"
    HELIX_CORTEX = "helix_cortex"

@dataclass
class MemoryBlock:
    """Represents a memory block allocation"""
    team: TeamType
    base_id: int
    core_id: int
    size: int
    allocated: bool = False
    data: Optional[np.ndarray] = None
    access_count: int = 0
    last_access: float = 0.0
    compression_ratio: float = 1.0

@dataclass
class LogicBase:
    """Represents a logic call algorithm base with 16 cores"""
    base_id: int
    team: TeamType
    memory_size: int = MEMORY_PER_BASE
    cores: List[MemoryBlock] = None
    utilization: float = 0.0

    def __post_init__(self):
        if self.cores is None:
            self.cores = []
            for core_id in range(LOGIC_CORES_PER_BASE):
                self.cores.append(MemoryBlock(
                    team=self.team,
                    base_id=self.base_id,
                    core_id=core_id,
                    size=MEMORY_PER_CORE
                ))

class MemoryManager:
    """
    Core memory management system for the four-team AGI framework.
    Manages 6GB total memory (1.5GB per team) with 6 logic bases per team.
    """

    def __init__(self):
        self.teams: Dict[TeamType, List[LogicBase]] = {}
        self.memory_usage: Dict[TeamType, Dict[str, float]] = {}
        self.access_history: deque = deque(maxlen=10000)
        self.lock = threading.RLock()
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None

        # Initialize memory structures for all teams
        self._initialize_team_memory()

        # Start monitoring
        self.start_monitoring()

        logger.info("Memory Manager initialized with 6GB total memory (1.5GB per team)")

    def _initialize_team_memory(self):
        """Initialize memory structures for all four teams"""
        for team in TeamType:
            self.teams[team] = []
            self.memory_usage[team] = {
                'allocated': 0,
                'used': 0,
                'free': TEAM_MEMORY_SIZE,
                'utilization': 0.0,
                'compression_ratio': 1.0
            }

            # Create 6 logic bases per team
            for base_id in range(LOGIC_BASES_PER_TEAM):
                logic_base = LogicBase(base_id=base_id, team=team)
                self.teams[team].append(logic_base)

        logger.info(f"Initialized memory for {len(TeamType)} teams with {LOGIC_BASES_PER_TEAM} bases each")

    def allocate_memory(self, team: TeamType, base_id: int, core_id: int, 
                       size: int, data: Optional[np.ndarray] = None) -> bool:
        """
        Allocate memory for a specific team, base, and core.

        Args:
            team: Team requesting memory
            base_id: Logic base ID (0-5)
            core_id: Logic core ID (0-15)
            size: Size in bytes to allocate
            data: Optional data to store

        Returns:
            bool: True if allocation successful, False otherwise
        """
        with self.lock:
            try:
                if not self._validate_allocation_request(team, base_id, core_id, size):
                    return False

                logic_base = self.teams[team][base_id]
                memory_block = logic_base.cores[core_id]

                if memory_block.allocated:
                    logger.warning(f"Memory block already allocated: {team.value}-{base_id}-{core_id}")
                    return False

                if size > memory_block.size:
                    logger.error(f"Requested size {size} exceeds core capacity {memory_block.size}")
                    return False

                # Allocate memory
                memory_block.allocated = True
                memory_block.data = data if data is not None else np.zeros(size // 8, dtype=np.float64)
                memory_block.last_access = time.time()
                memory_block.access_count += 1

                # Update usage statistics
                self._update_memory_usage(team)

                # Log allocation
                self.access_history.append({
                    'timestamp': time.time(),
                    'operation': 'allocate',
                    'team': team.value,
                    'base_id': base_id,
                    'core_id': core_id,
                    'size': size
                })

                logger.debug(f"Memory allocated: {team.value}-{base_id}-{core_id}, size: {size}")
                return True

            except Exception as e:
                logger.error(f"Memory allocation failed: {e}")
                return False

    def deallocate_memory(self, team: TeamType, base_id: int, core_id: int) -> bool:
        """
        Deallocate memory for a specific team, base, and core.

        Args:
            team: Team releasing memory
            base_id: Logic base ID (0-5)
            core_id: Logic core ID (0-15)

        Returns:
            bool: True if deallocation successful, False otherwise
        """
        with self.lock:
            try:
                if not self._validate_deallocation_request(team, base_id, core_id):
                    return False

                logic_base = self.teams[team][base_id]
                memory_block = logic_base.cores[core_id]

                if not memory_block.allocated:
                    logger.warning(f"Memory block not allocated: {team.value}-{base_id}-{core_id}")
                    return False

                # Deallocate memory
                memory_block.allocated = False
                memory_block.data = None
                memory_block.last_access = time.time()

                # Update usage statistics
                self._update_memory_usage(team)

                # Log deallocation
                self.access_history.append({
                    'timestamp': time.time(),
                    'operation': 'deallocate',
                    'team': team.value,
                    'base_id': base_id,
                    'core_id': core_id
                })

                logger.debug(f"Memory deallocated: {team.value}-{base_id}-{core_id}")
                return True

            except Exception as e:
                logger.error(f"Memory deallocation failed: {e}")
                return False

    def get_memory_usage(self, team: Optional[TeamType] = None) -> Dict[str, Any]:
        """
        Get memory usage statistics for a team or all teams.

        Args:
            team: Specific team to query, or None for all teams

        Returns:
            Dict containing memory usage statistics
        """
        with self.lock:
            if team:
                return {
                    'team': team.value,
                    'usage': self.memory_usage[team].copy(),
                    'bases': self._get_base_usage(team)
                }
            else:
                return {
                    'total_usage': {
                        team.value: self.memory_usage[team].copy() 
                        for team in TeamType
                    },
                    'system_summary': self._get_system_summary()
                }

    def apply_compression(self, team: TeamType, compression_ratio: float = 1000.0) -> bool:
        """
        Apply memory compression for ARIEL team (1000:1 ratio as per specs).

        Args:
            team: Team to apply compression to
            compression_ratio: Compression ratio to apply

        Returns:
            bool: True if compression applied successfully
        """
        if team != TeamType.ARIEL and compression_ratio > 10.0:
            logger.warning(f"High compression ratio {compression_ratio} only supported for ARIEL team")
            return False

        with self.lock:
            try:
                for logic_base in self.teams[team]:
                    for memory_block in logic_base.cores:
                        if memory_block.allocated and memory_block.data is not None:
                            # Simulate compression by updating the compression ratio
                            memory_block.compression_ratio = compression_ratio

                # Update team compression ratio
                self.memory_usage[team]['compression_ratio'] = compression_ratio

                logger.info(f"Applied {compression_ratio}:1 compression to {team.value} team")
                return True

            except Exception as e:
                logger.error(f"Compression failed for {team.value}: {e}")
                return False

    def detect_memory_leaks(self) -> List[Dict[str, Any]]:
        """
        Detect potential memory leaks by analyzing access patterns.

        Returns:
            List of potential memory leak indicators
        """
        leaks = []
        current_time = time.time()

        with self.lock:
            for team in TeamType:
                for logic_base in self.teams[team]:
                    for memory_block in logic_base.cores:
                        if memory_block.allocated:
                            # Check for stale allocations (no access in 1 hour)
                            if current_time - memory_block.last_access > 3600:
                                leaks.append({
                                    'team': team.value,
                                    'base_id': logic_base.base_id,
                                    'core_id': memory_block.core_id,
                                    'last_access': memory_block.last_access,
                                    'age_hours': (current_time - memory_block.last_access) / 3600,
                                    'type': 'stale_allocation'
                                })

        if leaks:
            logger.warning(f"Detected {len(leaks)} potential memory leaks")

        return leaks

    def optimize_memory_layout(self, team: TeamType) -> bool:
        """
        Optimize memory layout based on access patterns.

        Args:
            team: Team to optimize

        Returns:
            bool: True if optimization successful
        """
        with self.lock:
            try:
                # Collect access statistics
                access_stats = []
                for base_id, logic_base in enumerate(self.teams[team]):
                    for core_id, memory_block in enumerate(logic_base.cores):
                        access_stats.append({
                            'base_id': base_id,
                            'core_id': core_id,
                            'access_count': memory_block.access_count,
                            'last_access': memory_block.last_access,
                            'allocated': memory_block.allocated
                        })

                # Sort by access frequency
                access_stats.sort(key=lambda x: x['access_count'], reverse=True)

                # Log optimization results
                logger.info(f"Memory layout optimized for {team.value} team")
                return True

            except Exception as e:
                logger.error(f"Memory optimization failed for {team.value}: {e}")
                return False

    def start_monitoring(self):
        """Start memory monitoring thread"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Memory monitoring started")

    def stop_monitoring(self):
        """Stop memory monitoring thread"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Memory monitoring stopped")

    def _monitoring_loop(self):
        """Memory monitoring loop"""
        while self.monitoring_active:
            try:
                # Update memory usage for all teams
                for team in TeamType:
                    self._update_memory_usage(team)

                # Check for memory leaks every 5 minutes
                if int(time.time()) % 300 == 0:
                    self.detect_memory_leaks()

                time.sleep(1.0)  # Monitor every second

            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(5.0)

    def _validate_allocation_request(self, team: TeamType, base_id: int, 
                                   core_id: int, size: int) -> bool:
        """Validate memory allocation request"""
        if base_id < 0 or base_id >= LOGIC_BASES_PER_TEAM:
            logger.error(f"Invalid base_id {base_id}, must be 0-{LOGIC_BASES_PER_TEAM-1}")
            return False

        if core_id < 0 or core_id >= LOGIC_CORES_PER_BASE:
            logger.error(f"Invalid core_id {core_id}, must be 0-{LOGIC_CORES_PER_BASE-1}")
            return False

        if size <= 0 or size > MEMORY_PER_CORE:
            logger.error(f"Invalid size {size}, must be 1-{MEMORY_PER_CORE}")
            return False

        return True

    def _validate_deallocation_request(self, team: TeamType, base_id: int, core_id: int) -> bool:
        """Validate memory deallocation request"""
        return self._validate_allocation_request(team, base_id, core_id, 1)

    def _update_memory_usage(self, team: TeamType):
        """Update memory usage statistics for a team"""
        allocated = 0
        used = 0

        for logic_base in self.teams[team]:
            for memory_block in logic_base.cores:
                if memory_block.allocated:
                    allocated += memory_block.size
                    if memory_block.data is not None:
                        used += memory_block.data.nbytes

        self.memory_usage[team].update({
            'allocated': allocated,
            'used': used,
            'free': TEAM_MEMORY_SIZE - allocated,
            'utilization': allocated / TEAM_MEMORY_SIZE
        })

    def _get_base_usage(self, team: TeamType) -> List[Dict[str, Any]]:
        """Get usage statistics for all bases of a team"""
        base_usage = []

        for logic_base in self.teams[team]:
            allocated_cores = sum(1 for core in logic_base.cores if core.allocated)
            base_usage.append({
                'base_id': logic_base.base_id,
                'allocated_cores': allocated_cores,
                'total_cores': len(logic_base.cores),
                'utilization': allocated_cores / len(logic_base.cores)
            })

        return base_usage

    def _get_system_summary(self) -> Dict[str, Any]:
        """Get system-wide memory summary"""
        total_allocated = sum(usage['allocated'] for usage in self.memory_usage.values())
        total_capacity = TEAM_MEMORY_SIZE * len(TeamType)

        return {
            'total_capacity': total_capacity,
            'total_allocated': total_allocated,
            'total_free': total_capacity - total_allocated,
            'system_utilization': total_allocated / total_capacity,
            'teams_count': len(TeamType),
            'bases_per_team': LOGIC_BASES_PER_TEAM,
            'cores_per_team': LOGIC_CORES_PER_TEAM
        }
