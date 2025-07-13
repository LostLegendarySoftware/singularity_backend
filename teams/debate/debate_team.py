"""
DEBATE Team Implementation - 16-Agent Debate System
Based on verified specifications for consensus-based decision making
Implements multi-agent internal reasoning with distributed processing
"""

import threading
import time
import logging
import queue
import random
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import json
import hashlib

logger = logging.getLogger(__name__)

# Constants from verified specifications
DEBATE_MEMORY_SIZE = int(1.5 * 1024 * 1024 * 1024)  # 1.5GB
DEBATE_LOGIC_BASES = 6
DEBATE_LOGIC_CORES = 96
DEBATE_AGENTS = 16
CORES_PER_AGENT = DEBATE_LOGIC_CORES // DEBATE_AGENTS  # 6 cores per agent

class DebatePhase(Enum):
    INITIALIZATION = auto()
    ARGUMENT_PRESENTATION = auto()
    CROSS_EXAMINATION = auto()
    REBUTTAL = auto()
    CONSENSUS_BUILDING = auto()
    FINAL_DECISION = auto()
    VALIDATION = auto()

class AgentRole(Enum):
    PROPOSER = "proposer"
    OPPONENT = "opponent"
    MODERATOR = "moderator"
    ANALYST = "analyst"
    VALIDATOR = "validator"
    SYNTHESIZER = "synthesizer"

class ArgumentType(Enum):
    LOGICAL = "logical"
    EMPIRICAL = "empirical"
    ETHICAL = "ethical"
    PRACTICAL = "practical"
    THEORETICAL = "theoretical"
    EXPERIENTIAL = "experiential"

@dataclass
class Argument:
    """Represents a single argument in the debate"""
    id: str
    agent_id: int
    content: str
    argument_type: ArgumentType
    strength: float  # 0.0 to 1.0
    evidence: List[str] = field(default_factory=list)
    counterarguments: List[str] = field(default_factory=list)
    support_votes: int = 0
    opposition_votes: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def credibility_score(self) -> float:
        """Calculate argument credibility based on votes and evidence"""
        if self.support_votes + self.opposition_votes == 0:
            return self.strength

        vote_ratio = self.support_votes / (self.support_votes + self.opposition_votes)
        evidence_bonus = min(len(self.evidence) * 0.1, 0.3)

        return min(1.0, (vote_ratio * 0.7 + self.strength * 0.3) + evidence_bonus)

@dataclass
class DebatePosition:
    """Represents a position in the debate"""
    id: str
    title: str
    description: str
    arguments: List[Argument] = field(default_factory=list)
    supporting_agents: List[int] = field(default_factory=list)
    opposing_agents: List[int] = field(default_factory=list)
    confidence_score: float = 0.5

    def add_argument(self, argument: Argument):
        """Add an argument to this position"""
        self.arguments.append(argument)
        self._update_confidence_score()

    def _update_confidence_score(self):
        """Update confidence score based on arguments"""
        if not self.arguments:
            self.confidence_score = 0.5
            return

        total_credibility = sum(arg.credibility_score for arg in self.arguments)
        self.confidence_score = min(1.0, total_credibility / len(self.arguments))

@dataclass
class ConsensusResult:
    """Result of consensus building process"""
    decision: str
    confidence: float
    supporting_arguments: List[Argument]
    dissenting_opinions: List[str]
    consensus_level: float  # 0.0 to 1.0
    validation_score: float
    timestamp: float = field(default_factory=time.time)

class DebateAgent:
    """Individual debate agent with specific role and reasoning capabilities"""

    def __init__(self, agent_id: int, role: AgentRole, personality_traits: Dict[str, float] = None):
        self.agent_id = agent_id
        self.role = role
        self.personality_traits = personality_traits or self._generate_personality()

        # Agent state
        self.active = False
        self.current_position: Optional[DebatePosition] = None
        self.argument_history: List[Argument] = []
        self.vote_history: List[Dict[str, Any]] = []

        # Performance metrics
        self.arguments_made = 0
        self.successful_arguments = 0
        self.consensus_contributions = 0
        self.credibility_score = 0.5

        # Reasoning capabilities
        self.reasoning_strategies = self._initialize_reasoning_strategies()

        logger.debug(f"DebateAgent {agent_id} initialized with role {role.value}")

    def _generate_personality(self) -> Dict[str, float]:
        """Generate personality traits for the agent"""
        return {
            'aggressiveness': random.uniform(0.2, 0.8),
            'analytical_depth': random.uniform(0.3, 0.9),
            'creativity': random.uniform(0.2, 0.8),
            'skepticism': random.uniform(0.3, 0.7),
            'collaboration': random.uniform(0.4, 0.9),
            'risk_tolerance': random.uniform(0.2, 0.8)
        }

    def _initialize_reasoning_strategies(self) -> Dict[str, Callable]:
        """Initialize reasoning strategies based on role"""
        base_strategies = {
            'logical_analysis': self._logical_reasoning,
            'evidence_evaluation': self._evidence_based_reasoning,
            'counterargument_generation': self._generate_counterarguments,
            'synthesis': self._synthesize_positions
        }

        # Role-specific strategies
        if self.role == AgentRole.ANALYST:
            base_strategies['deep_analysis'] = self._deep_analytical_reasoning
        elif self.role == AgentRole.MODERATOR:
            base_strategies['conflict_resolution'] = self._conflict_resolution_reasoning
        elif self.role == AgentRole.VALIDATOR:
            base_strategies['validation'] = self._validation_reasoning

        return base_strategies

    def generate_argument(self, topic: str, context: Dict[str, Any]) -> Optional[Argument]:
        """Generate an argument on the given topic"""
        try:
            # Select reasoning strategy based on personality and role
            strategy = self._select_reasoning_strategy(context)

            # Generate argument content
            content = strategy(topic, context)

            if not content:
                return None

            # Determine argument type
            arg_type = self._determine_argument_type(content, context)

            # Calculate argument strength
            strength = self._calculate_argument_strength(content, context)

            # Create argument
            argument = Argument(
                id=f"arg_{self.agent_id}_{int(time.time())}_{random.randint(1000, 9999)}",
                agent_id=self.agent_id,
                content=content,
                argument_type=arg_type,
                strength=strength,
                evidence=self._gather_evidence(content, context)
            )

            # Update agent state
            self.argument_history.append(argument)
            self.arguments_made += 1

            logger.debug(f"Agent {self.agent_id} generated argument: {argument.id}")
            return argument

        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to generate argument: {e}")
            return None

    def evaluate_argument(self, argument: Argument, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate another agent's argument"""
        try:
            evaluation = {
                'agent_id': self.agent_id,
                'argument_id': argument.id,
                'credibility_assessment': self._assess_credibility(argument),
                'logical_consistency': self._check_logical_consistency(argument),
                'evidence_quality': self._evaluate_evidence(argument),
                'relevance_score': self._assess_relevance(argument, context),
                'overall_score': 0.0,
                'comments': []
            }

            # Calculate overall score
            scores = [
                evaluation['credibility_assessment'],
                evaluation['logical_consistency'],
                evaluation['evidence_quality'],
                evaluation['relevance_score']
            ]
            evaluation['overall_score'] = sum(scores) / len(scores)

            # Generate comments
            evaluation['comments'] = self._generate_evaluation_comments(argument, evaluation)

            return evaluation

        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to evaluate argument {argument.id}: {e}")
            return {'error': str(e)}

    def vote_on_argument(self, argument: Argument, context: Dict[str, Any]) -> bool:
        """Vote on an argument (True for support, False for opposition)"""
        try:
            evaluation = self.evaluate_argument(argument, context)

            # Decision based on evaluation and personality
            support_threshold = 0.5 + (self.personality_traits['skepticism'] - 0.5) * 0.3

            vote = evaluation.get('overall_score', 0.5) > support_threshold

            # Record vote
            self.vote_history.append({
                'argument_id': argument.id,
                'vote': vote,
                'timestamp': time.time(),
                'evaluation_score': evaluation.get('overall_score', 0.5)
            })

            return vote

        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to vote on argument {argument.id}: {e}")
            return False

    def _select_reasoning_strategy(self, context: Dict[str, Any]) -> Callable:
        """Select appropriate reasoning strategy"""
        # Simple strategy selection based on role and context
        if self.role == AgentRole.ANALYST:
            return self.reasoning_strategies.get('deep_analysis', self.reasoning_strategies['logical_analysis'])
        elif self.role == AgentRole.VALIDATOR:
            return self.reasoning_strategies.get('validation', self.reasoning_strategies['evidence_evaluation'])
        else:
            return self.reasoning_strategies['logical_analysis']

    def _logical_reasoning(self, topic: str, context: Dict[str, Any]) -> str:
        """Generate logical reasoning-based argument"""
        premises = context.get('premises', [])

        if not premises:
            return f"Based on logical analysis of {topic}, we must consider the fundamental principles involved."

        return f"Given the premises {premises}, it logically follows that {topic} should be approached with careful consideration of cause-and-effect relationships."

    def _evidence_based_reasoning(self, topic: str, context: Dict[str, Any]) -> str:
        """Generate evidence-based argument"""
        evidence = context.get('evidence', [])

        if evidence:
            return f"The available evidence regarding {topic} suggests that empirical data supports a measured approach based on observed patterns."
        else:
            return f"While direct evidence for {topic} may be limited, we can draw inferences from related empirical observations."

    def _generate_counterarguments(self, topic: str, context: Dict[str, Any]) -> str:
        """Generate counterarguments"""
        existing_positions = context.get('positions', [])

        if existing_positions:
            return f"While the current positions on {topic} have merit, we must also consider alternative perspectives that challenge these assumptions."
        else:
            return f"In examining {topic}, it's crucial to anticipate and address potential objections to ensure robust reasoning."

    def _synthesize_positions(self, topic: str, context: Dict[str, Any]) -> str:
        """Synthesize multiple positions"""
        positions = context.get('positions', [])

        if len(positions) >= 2:
            return f"Considering the various perspectives on {topic}, we can identify common ground and build a more comprehensive understanding."
        else:
            return f"To fully understand {topic}, we need to integrate multiple viewpoints and find synthesis opportunities."

    def _deep_analytical_reasoning(self, topic: str, context: Dict[str, Any]) -> str:
        """Deep analytical reasoning (for analyst role)"""
        return f"Through systematic analysis of {topic}, examining underlying assumptions, causal relationships, and potential implications across multiple dimensions."

    def _conflict_resolution_reasoning(self, topic: str, context: Dict[str, Any]) -> str:
        """Conflict resolution reasoning (for moderator role)"""
        return f"To address the conflicting views on {topic}, we should focus on identifying shared values and finding mutually acceptable solutions."

    def _validation_reasoning(self, topic: str, context: Dict[str, Any]) -> str:
        """Validation reasoning (for validator role)"""
        return f"Validating the claims about {topic} requires rigorous examination of methodology, consistency, and reproducibility of results."

    def _determine_argument_type(self, content: str, context: Dict[str, Any]) -> ArgumentType:
        """Determine the type of argument based on content"""
        content_lower = content.lower()

        if any(word in content_lower for word in ['evidence', 'data', 'empirical', 'observed']):
            return ArgumentType.EMPIRICAL
        elif any(word in content_lower for word in ['logical', 'follows', 'premises', 'therefore']):
            return ArgumentType.LOGICAL
        elif any(word in content_lower for word in ['should', 'ought', 'moral', 'ethical']):
            return ArgumentType.ETHICAL
        elif any(word in content_lower for word in ['practical', 'implementation', 'feasible']):
            return ArgumentType.PRACTICAL
        elif any(word in content_lower for word in ['theory', 'theoretical', 'conceptual']):
            return ArgumentType.THEORETICAL
        else:
            return ArgumentType.EXPERIENTIAL

    def _calculate_argument_strength(self, content: str, context: Dict[str, Any]) -> float:
        """Calculate argument strength based on content and context"""
        base_strength = 0.5

        # Adjust based on personality traits
        analytical_bonus = self.personality_traits['analytical_depth'] * 0.2
        creativity_bonus = self.personality_traits['creativity'] * 0.1

        # Adjust based on content quality (simplified)
        content_quality = min(len(content) / 200.0, 1.0) * 0.2

        return min(1.0, base_strength + analytical_bonus + creativity_bonus + content_quality)

    def _gather_evidence(self, content: str, context: Dict[str, Any]) -> List[str]:
        """Gather supporting evidence for the argument"""
        evidence = []

        # Extract evidence from context
        available_evidence = context.get('evidence', [])

        # Simple evidence matching (in real implementation, this would be more sophisticated)
        for item in available_evidence:
            if any(word in content.lower() for word in item.lower().split()):
                evidence.append(item)

        # Generate synthetic evidence references if none found
        if not evidence:
            evidence = [f"Supporting reference for argument about {content[:50]}..."]

        return evidence[:3]  # Limit to 3 pieces of evidence

    def _assess_credibility(self, argument: Argument) -> float:
        """Assess the credibility of an argument"""
        # Base credibility on evidence quality and logical structure
        evidence_score = min(len(argument.evidence) * 0.2, 0.6)
        content_score = min(len(argument.content) / 300.0, 0.4)

        return evidence_score + content_score

    def _check_logical_consistency(self, argument: Argument) -> float:
        """Check logical consistency of an argument"""
        # Simplified logical consistency check
        content = argument.content.lower()

        # Look for logical indicators
        logical_indicators = ['because', 'therefore', 'thus', 'hence', 'consequently', 'since']
        consistency_score = 0.5

        for indicator in logical_indicators:
            if indicator in content:
                consistency_score += 0.1

        return min(1.0, consistency_score)

    def _evaluate_evidence(self, argument: Argument) -> float:
        """Evaluate the quality of evidence provided"""
        if not argument.evidence:
            return 0.2

        # Score based on quantity and assumed quality
        quantity_score = min(len(argument.evidence) * 0.25, 0.75)
        quality_score = 0.25  # Assumed base quality

        return quantity_score + quality_score

    def _assess_relevance(self, argument: Argument, context: Dict[str, Any]) -> float:
        """Assess relevance of argument to the topic"""
        topic = context.get('topic', '')

        if not topic:
            return 0.5

        # Simple relevance check based on keyword overlap
        topic_words = set(topic.lower().split())
        content_words = set(argument.content.lower().split())

        overlap = len(topic_words.intersection(content_words))
        relevance = min(overlap / max(len(topic_words), 1), 1.0)

        return max(0.3, relevance)  # Minimum relevance threshold

    def _generate_evaluation_comments(self, argument: Argument, evaluation: Dict[str, Any]) -> List[str]:
        """Generate evaluation comments"""
        comments = []

        if evaluation['overall_score'] > 0.8:
            comments.append("Strong argument with solid reasoning and evidence.")
        elif evaluation['overall_score'] > 0.6:
            comments.append("Good argument with room for improvement in evidence or logic.")
        elif evaluation['overall_score'] > 0.4:
            comments.append("Moderate argument that needs strengthening.")
        else:
            comments.append("Weak argument requiring significant improvement.")

        if evaluation['evidence_quality'] < 0.3:
            comments.append("Insufficient evidence provided to support claims.")

        if evaluation['logical_consistency'] < 0.4:
            comments.append("Logical structure could be improved for clarity.")

        return comments

class DebateEngine:
    """Multi-agent internal reasoning system - HIGH priority component"""

    def __init__(self):
        self.agents: List[DebateAgent] = []
        self.current_debate: Optional[Dict[str, Any]] = None
        self.debate_history: List[Dict[str, Any]] = []
        self.positions: List[DebatePosition] = []

        # Debate state
        self.phase = DebatePhase.INITIALIZATION
        self.active = False
        self.consensus_threshold = 0.75  # 75% agreement for consensus

        # Performance metrics
        self.debates_conducted = 0
        self.consensus_achieved = 0
        self.average_debate_duration = 0.0

        # Threading
        self.debate_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()

        # Initialize agents
        self._initialize_agents()

        logger.info("DebateEngine initialized with 16 agents")

    def _initialize_agents(self):
        """Initialize 16 debate agents with different roles"""
        roles = [
            AgentRole.PROPOSER, AgentRole.PROPOSER, AgentRole.PROPOSER,  # 3 proposers
            AgentRole.OPPONENT, AgentRole.OPPONENT, AgentRole.OPPONENT,  # 3 opponents
            AgentRole.ANALYST, AgentRole.ANALYST, AgentRole.ANALYST,     # 3 analysts
            AgentRole.MODERATOR, AgentRole.MODERATOR,                    # 2 moderators
            AgentRole.VALIDATOR, AgentRole.VALIDATOR,                    # 2 validators
            AgentRole.SYNTHESIZER, AgentRole.SYNTHESIZER, AgentRole.SYNTHESIZER  # 3 synthesizers
        ]

        for i in range(DEBATE_AGENTS):
            role = roles[i] if i < len(roles) else AgentRole.ANALYST
            agent = DebateAgent(agent_id=i, role=role)
            self.agents.append(agent)

    def start_debate(self, topic: str, context: Dict[str, Any] = None) -> str:
        """Start a new debate on the given topic"""
        try:
            with self.lock:
                if self.active:
                    return "Debate already in progress"

                # Initialize debate
                debate_id = f"debate_{int(time.time())}_{random.randint(1000, 9999)}"

                self.current_debate = {
                    'id': debate_id,
                    'topic': topic,
                    'context': context or {},
                    'start_time': time.time(),
                    'arguments': [],
                    'votes': {},
                    'phase_history': []
                }

                self.positions = []
                self.phase = DebatePhase.INITIALIZATION
                self.active = True

                # Start debate thread
                self.debate_thread = threading.Thread(target=self._debate_loop, daemon=True)
                self.debate_thread.start()

                logger.info(f"Started debate {debate_id} on topic: {topic}")
                return debate_id

        except Exception as e:
            logger.error(f"Failed to start debate: {e}")
            return f"Error: {e}"

    def get_debate_status(self) -> Dict[str, Any]:
        """Get current debate status"""
        with self.lock:
            if not self.current_debate:
                return {'status': 'no_active_debate'}

            return {
                'debate_id': self.current_debate['id'],
                'topic': self.current_debate['topic'],
                'phase': self.phase.name,
                'active': self.active,
                'arguments_count': len(self.current_debate['arguments']),
                'positions_count': len(self.positions),
                'duration': time.time() - self.current_debate['start_time'],
                'agent_participation': {
                    agent.agent_id: {
                        'arguments_made': agent.arguments_made,
                        'role': agent.role.value,
                        'credibility': agent.credibility_score
                    }
                    for agent in self.agents
                }
            }

    def force_consensus(self) -> Optional[ConsensusResult]:
        """Force consensus building (for testing or emergency situations)"""
        try:
            with self.lock:
                if not self.current_debate:
                    return None

                self.phase = DebatePhase.CONSENSUS_BUILDING
                return self._build_consensus()

        except Exception as e:
            logger.error(f"Failed to force consensus: {e}")
            return None

    def _debate_loop(self):
        """Main debate processing loop"""
        try:
            while self.active and self.current_debate:
                if self.phase == DebatePhase.INITIALIZATION:
                    self._initialization_phase()
                elif self.phase == DebatePhase.ARGUMENT_PRESENTATION:
                    self._argument_presentation_phase()
                elif self.phase == DebatePhase.CROSS_EXAMINATION:
                    self._cross_examination_phase()
                elif self.phase == DebatePhase.REBUTTAL:
                    self._rebuttal_phase()
                elif self.phase == DebatePhase.CONSENSUS_BUILDING:
                    consensus = self._consensus_building_phase()
                    if consensus:
                        self._finalize_debate(consensus)
                        break
                elif self.phase == DebatePhase.FINAL_DECISION:
                    self._final_decision_phase()
                    break

                time.sleep(1.0)  # Phase processing interval

        except Exception as e:
            logger.error(f"Debate loop error: {e}")
        finally:
            self.active = False

    def _initialization_phase(self):
        """Initialize debate positions and agent assignments"""
        try:
            topic = self.current_debate['topic']
            context = self.current_debate['context']

            # Create initial positions
            pro_position = DebatePosition(
                id=f"pro_{int(time.time())}",
                title=f"Support for {topic}",
                description=f"Arguments supporting the proposition: {topic}"
            )

            con_position = DebatePosition(
                id=f"con_{int(time.time())}",
                title=f"Opposition to {topic}",
                description=f"Arguments opposing the proposition: {topic}"
            )

            self.positions = [pro_position, con_position]

            # Assign agents to positions
            proposers = [agent for agent in self.agents if agent.role == AgentRole.PROPOSER]
            opponents = [agent for agent in self.agents if agent.role == AgentRole.OPPONENT]

            for agent in proposers:
                pro_position.supporting_agents.append(agent.agent_id)
                agent.current_position = pro_position

            for agent in opponents:
                con_position.supporting_agents.append(agent.agent_id)
                agent.current_position = con_position

            # Record phase completion
            self.current_debate['phase_history'].append({
                'phase': self.phase.name,
                'timestamp': time.time(),
                'duration': 1.0
            })

            self.phase = DebatePhase.ARGUMENT_PRESENTATION
            logger.info("Debate initialization completed")

        except Exception as e:
            logger.error(f"Initialization phase failed: {e}")
            self.phase = DebatePhase.FINAL_DECISION

    def _argument_presentation_phase(self):
        """Agents present their initial arguments"""
        try:
            topic = self.current_debate['topic']
            context = self.current_debate['context']

            # Each agent with a position presents arguments
            for agent in self.agents:
                if agent.current_position:
                    argument = agent.generate_argument(topic, context)
                    if argument:
                        self.current_debate['arguments'].append(argument)
                        agent.current_position.add_argument(argument)

            # Record phase completion
            self.current_debate['phase_history'].append({
                'phase': self.phase.name,
                'timestamp': time.time(),
                'arguments_generated': len([a for a in self.current_debate['arguments'] 
                                          if a.timestamp > time.time() - 60])
            })

            self.phase = DebatePhase.CROSS_EXAMINATION
            logger.info(f"Argument presentation completed with {len(self.current_debate['arguments'])} arguments")

        except Exception as e:
            logger.error(f"Argument presentation phase failed: {e}")
            self.phase = DebatePhase.CONSENSUS_BUILDING

    def _cross_examination_phase(self):
        """Agents examine and vote on each other's arguments"""
        try:
            context = self.current_debate['context']
            context['topic'] = self.current_debate['topic']

            # Each agent evaluates arguments from other agents
            for argument in self.current_debate['arguments']:
                for agent in self.agents:
                    if agent.agent_id != argument.agent_id:
                        vote = agent.vote_on_argument(argument, context)

                        if vote:
                            argument.support_votes += 1
                        else:
                            argument.opposition_votes += 1

                        # Store vote details
                        vote_key = f"{agent.agent_id}_{argument.id}"
                        self.current_debate['votes'][vote_key] = {
                            'voter_id': agent.agent_id,
                            'argument_id': argument.id,
                            'vote': vote,
                            'timestamp': time.time()
                        }

            # Record phase completion
            self.current_debate['phase_history'].append({
                'phase': self.phase.name,
                'timestamp': time.time(),
                'votes_cast': len(self.current_debate['votes'])
            })

            self.phase = DebatePhase.REBUTTAL
            logger.info(f"Cross-examination completed with {len(self.current_debate['votes'])} votes")

        except Exception as e:
            logger.error(f"Cross-examination phase failed: {e}")
            self.phase = DebatePhase.CONSENSUS_BUILDING

    def _rebuttal_phase(self):
        """Agents provide rebuttals to opposing arguments"""
        try:
            topic = self.current_debate['topic']
            context = self.current_debate['context']

            # Generate rebuttals for low-scoring arguments
            for argument in self.current_debate['arguments']:
                if argument.credibility_score < 0.5:
                    # Find agents from opposing position to provide rebuttals
                    opposing_agents = [
                        agent for agent in self.agents 
                        if agent.current_position and 
                        agent.current_position.id != argument.agent_id
                    ]

                    if opposing_agents:
                        rebuttal_agent = random.choice(opposing_agents)
                        rebuttal_context = context.copy()
                        rebuttal_context['target_argument'] = argument

                        rebuttal = rebuttal_agent.generate_argument(
                            f"Rebuttal to: {argument.content[:100]}...", 
                            rebuttal_context
                        )

                        if rebuttal:
                            argument.counterarguments.append(rebuttal.id)
                            self.current_debate['arguments'].append(rebuttal)

            # Record phase completion
            self.current_debate['phase_history'].append({
                'phase': self.phase.name,
                'timestamp': time.time(),
                'rebuttals_generated': len([a for a in self.current_debate['arguments'] 
                                          if 'Rebuttal' in a.content])
            })

            self.phase = DebatePhase.CONSENSUS_BUILDING
            logger.info("Rebuttal phase completed")

        except Exception as e:
            logger.error(f"Rebuttal phase failed: {e}")
            self.phase = DebatePhase.CONSENSUS_BUILDING

    def _consensus_building_phase(self) -> Optional[ConsensusResult]:
        """Build consensus from debate results"""
        try:
            consensus = self._build_consensus()

            if consensus and consensus.consensus_level >= self.consensus_threshold:
                self.phase = DebatePhase.VALIDATION
                return consensus
            else:
                # If no consensus, continue debate or force decision
                debate_duration = time.time() - self.current_debate['start_time']
                if debate_duration > 300:  # 5 minutes maximum
                    self.phase = DebatePhase.FINAL_DECISION
                else:
                    # Return to argument presentation for another round
                    self.phase = DebatePhase.ARGUMENT_PRESENTATION

                return None

        except Exception as e:
            logger.error(f"Consensus building failed: {e}")
            self.phase = DebatePhase.FINAL_DECISION
            return None

    def _build_consensus(self) -> ConsensusResult:
        """Build consensus from current debate state"""
        try:
            # Analyze argument strengths and positions
            position_scores = {}
            all_arguments = []

            for position in self.positions:
                total_credibility = sum(arg.credibility_score for arg in position.arguments)
                avg_credibility = total_credibility / max(len(position.arguments), 1)
                position_scores[position.id] = avg_credibility
                all_arguments.extend(position.arguments)

            # Find winning position
            if not position_scores:
                winning_position = "No clear position"
                confidence = 0.0
            else:
                winning_position_id = max(position_scores.keys(), key=lambda k: position_scores[k])
                winning_position = next(p.title for p in self.positions if p.id == winning_position_id)
                confidence = position_scores[winning_position_id]

            # Calculate consensus level
            if len(position_scores) < 2:
                consensus_level = 1.0
            else:
                scores = list(position_scores.values())
                scores.sort(reverse=True)
                consensus_level = (scores[0] - scores[1]) if len(scores) > 1 else 1.0

            # Get supporting arguments
            supporting_args = [arg for arg in all_arguments if arg.credibility_score > 0.6]
            supporting_args.sort(key=lambda x: x.credibility_score, reverse=True)

            # Get dissenting opinions
            dissenting_opinions = [
                f"Agent {arg.agent_id}: {arg.content[:100]}..." 
                for arg in all_arguments 
                if arg.credibility_score < 0.4
            ]

            # Validation score (simplified)
            validation_score = min(confidence * consensus_level, 1.0)

            consensus = ConsensusResult(
                decision=winning_position,
                confidence=confidence,
                supporting_arguments=supporting_args[:5],  # Top 5 arguments
                dissenting_opinions=dissenting_opinions[:3],  # Top 3 dissenting opinions
                consensus_level=consensus_level,
                validation_score=validation_score
            )

            logger.info(f"Consensus built: {consensus.decision} (confidence: {consensus.confidence:.2f})")
            return consensus

        except Exception as e:
            logger.error(f"Consensus building failed: {e}")
            return ConsensusResult(
                decision="Error in consensus building",
                confidence=0.0,
                supporting_arguments=[],
                dissenting_opinions=[str(e)],
                consensus_level=0.0,
                validation_score=0.0
            )

    def _final_decision_phase(self):
        """Make final decision when consensus cannot be reached"""
        try:
            # Force consensus with lower threshold
            original_threshold = self.consensus_threshold
            self.consensus_threshold = 0.5

            consensus = self._build_consensus()
            self.consensus_threshold = original_threshold

            if consensus:
                self._finalize_debate(consensus)
            else:
                # Create default decision
                default_consensus = ConsensusResult(
                    decision="No consensus reached - requires further deliberation",
                    confidence=0.3,
                    supporting_arguments=[],
                    dissenting_opinions=["Insufficient agreement among agents"],
                    consensus_level=0.3,
                    validation_score=0.2
                )
                self._finalize_debate(default_consensus)

        except Exception as e:
            logger.error(f"Final decision phase failed: {e}")
            self.active = False

    def _finalize_debate(self, consensus: ConsensusResult):
        """Finalize the debate with the consensus result"""
        try:
            with self.lock:
                if self.current_debate:
                    # Add consensus to debate record
                    self.current_debate['consensus'] = consensus
                    self.current_debate['end_time'] = time.time()
                    self.current_debate['duration'] = (
                        self.current_debate['end_time'] - self.current_debate['start_time']
                    )

                    # Update performance metrics
                    self.debates_conducted += 1
                    if consensus.consensus_level >= self.consensus_threshold:
                        self.consensus_achieved += 1

                    # Update average duration
                    self.average_debate_duration = (
                        (self.average_debate_duration * (self.debates_conducted - 1) + 
                         self.current_debate['duration']) / self.debates_conducted
                    )

                    # Update agent credibility scores
                    self._update_agent_credibility()

                    # Store in history
                    self.debate_history.append(self.current_debate.copy())

                    # Keep only recent history
                    if len(self.debate_history) > 100:
                        self.debate_history.pop(0)

                    logger.info(f"Debate finalized: {consensus.decision}")

                # Reset state
                self.current_debate = None
                self.positions = []
                self.phase = DebatePhase.INITIALIZATION
                self.active = False

        except Exception as e:
            logger.error(f"Debate finalization failed: {e}")
            self.active = False

    def _update_agent_credibility(self):
        """Update agent credibility scores based on debate performance"""
        try:
            if not self.current_debate:
                return

            for agent in self.agents:
                # Calculate performance in this debate
                agent_arguments = [
                    arg for arg in self.current_debate['arguments'] 
                    if arg.agent_id == agent.agent_id
                ]

                if agent_arguments:
                    avg_credibility = sum(arg.credibility_score for arg in agent_arguments) / len(agent_arguments)

                    # Update agent's overall credibility (exponential moving average)
                    alpha = 0.2
                    agent.credibility_score = (1 - alpha) * agent.credibility_score + alpha * avg_credibility

                    # Update success count
                    successful_args = sum(1 for arg in agent_arguments if arg.credibility_score > 0.6)
                    agent.successful_arguments += successful_args

        except Exception as e:
            logger.error(f"Agent credibility update failed: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get debate engine performance metrics"""
        with self.lock:
            consensus_rate = self.consensus_achieved / max(self.debates_conducted, 1)

            agent_metrics = {}
            for agent in self.agents:
                success_rate = agent.successful_arguments / max(agent.arguments_made, 1)
                agent_metrics[agent.agent_id] = {
                    'role': agent.role.value,
                    'arguments_made': agent.arguments_made,
                    'success_rate': success_rate,
                    'credibility_score': agent.credibility_score,
                    'consensus_contributions': agent.consensus_contributions
                }

            return {
                'debates_conducted': self.debates_conducted,
                'consensus_achieved': self.consensus_achieved,
                'consensus_rate': consensus_rate,
                'average_debate_duration': self.average_debate_duration,
                'agent_metrics': agent_metrics,
                'current_debate_active': self.active
            }

class DebateVisualizer:
    """Internal debate display system - MEDIUM priority component"""

    def __init__(self):
        self.visualization_data = {}
        self.display_active = False
        self.lock = threading.RLock()

        logger.info("DebateVisualizer initialized")

    def visualize_debate(self, debate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create visualization data for a debate"""
        try:
            with self.lock:
                visualization = {
                    'debate_id': debate_data.get('id', 'unknown'),
                    'topic': debate_data.get('topic', 'Unknown Topic'),
                    'timeline': self._create_timeline(debate_data),
                    'argument_network': self._create_argument_network(debate_data),
                    'agent_participation': self._create_participation_chart(debate_data),
                    'consensus_evolution': self._create_consensus_evolution(debate_data),
                    'position_strength': self._create_position_strength_chart(debate_data)
                }

                self.visualization_data[debate_data.get('id', 'current')] = visualization
                return visualization

        except Exception as e:
            logger.error(f"Debate visualization failed: {e}")
            return {'error': str(e)}

    def _create_timeline(self, debate_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create timeline of debate events"""
        timeline = []

        # Add phase transitions
        for phase_record in debate_data.get('phase_history', []):
            timeline.append({
                'timestamp': phase_record['timestamp'],
                'event_type': 'phase_transition',
                'description': f"Entered {phase_record['phase']} phase",
                'phase': phase_record['phase']
            })

        # Add argument presentations
        for argument in debate_data.get('arguments', []):
            timeline.append({
                'timestamp': argument.timestamp,
                'event_type': 'argument',
                'description': f"Agent {argument.agent_id} presented argument",
                'agent_id': argument.agent_id,
                'argument_id': argument.id
            })

        # Sort by timestamp
        timeline.sort(key=lambda x: x['timestamp'])
        return timeline

    def _create_argument_network(self, debate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create argument relationship network"""
        network = {
            'nodes': [],
            'edges': []
        }

        # Add argument nodes
        for argument in debate_data.get('arguments', []):
            network['nodes'].append({
                'id': argument.id,
                'label': f"Arg {argument.agent_id}",
                'type': argument.argument_type.value,
                'strength': argument.strength,
                'credibility': argument.credibility_score,
                'agent_id': argument.agent_id
            })

        # Add counterargument edges
        for argument in debate_data.get('arguments', []):
            for counter_id in argument.counterarguments:
                network['edges'].append({
                    'source': counter_id,
                    'target': argument.id,
                    'type': 'counterargument'
                })

        return network

    def _create_participation_chart(self, debate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create agent participation chart"""
        participation = {}

        for argument in debate_data.get('arguments', []):
            agent_id = argument.agent_id
            if agent_id not in participation:
                participation[agent_id] = {
                    'arguments_count': 0,
                    'total_strength': 0.0,
                    'total_credibility': 0.0
                }

            participation[agent_id]['arguments_count'] += 1
            participation[agent_id]['total_strength'] += argument.strength
            participation[agent_id]['total_credibility'] += argument.credibility_score

        # Calculate averages
        for agent_id, data in participation.items():
            count = data['arguments_count']
            data['avg_strength'] = data['total_strength'] / count if count > 0 else 0
            data['avg_credibility'] = data['total_credibility'] / count if count > 0 else 0

        return participation

    def _create_consensus_evolution(self, debate_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create consensus evolution over time"""
        evolution = []

        # Simplified consensus tracking
        phase_history = debate_data.get('phase_history', [])
        for i, phase_record in enumerate(phase_history):
            consensus_level = min(0.1 * i, 1.0)  # Simplified progression
            evolution.append({
                'timestamp': phase_record['timestamp'],
                'consensus_level': consensus_level,
                'phase': phase_record['phase']
            })

        return evolution

    def _create_position_strength_chart(self, debate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create position strength comparison"""
        positions = {}

        for argument in debate_data.get('arguments', []):
            # Simplified position assignment based on agent role
            position = 'pro' if argument.agent_id < 8 else 'con'  # First 8 agents pro, rest con

            if position not in positions:
                positions[position] = {
                    'argument_count': 0,
                    'total_strength': 0.0,
                    'total_credibility': 0.0,
                    'support_votes': 0,
                    'opposition_votes': 0
                }

            positions[position]['argument_count'] += 1
            positions[position]['total_strength'] += argument.strength
            positions[position]['total_credibility'] += argument.credibility_score
            positions[position]['support_votes'] += argument.support_votes
            positions[position]['opposition_votes'] += argument.opposition_votes

        return positions

    def get_visualization_summary(self) -> Dict[str, Any]:
        """Get summary of available visualizations"""
        with self.lock:
            return {
                'available_debates': list(self.visualization_data.keys()),
                'total_visualizations': len(self.visualization_data),
                'display_active': self.display_active
            }

class AgentCoordinator:
    """Distributed agent management system"""

    def __init__(self, agents: List[DebateAgent]):
        self.agents = agents
        self.coordination_active = False
        self.task_queue = queue.Queue()
        self.coordination_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()

        logger.info(f"AgentCoordinator initialized with {len(agents)} agents")

    def start_coordination(self):
        """Start agent coordination"""
        if not self.coordination_active:
            self.coordination_active = True
            self.coordination_thread = threading.Thread(target=self._coordination_loop, daemon=True)
            self.coordination_thread.start()
            logger.info("Agent coordination started")

    def stop_coordination(self):
        """Stop agent coordination"""
        self.coordination_active = False
        if self.coordination_thread:
            self.coordination_thread.join(timeout=5.0)
        logger.info("Agent coordination stopped")

    def assign_task(self, task: Dict[str, Any]) -> bool:
        """Assign a task to appropriate agents"""
        try:
            self.task_queue.put(task)
            return True
        except Exception as e:
            logger.error(f"Task assignment failed: {e}")
            return False

    def _coordination_loop(self):
        """Agent coordination loop"""
        while self.coordination_active:
            try:
                # Process task queue
                try:
                    task = self.task_queue.get(timeout=1.0)
                    self._process_coordination_task(task)
                except queue.Empty:
                    continue

                # Monitor agent health
                self._monitor_agent_health()

                # Balance workload
                self._balance_agent_workload()

            except Exception as e:
                logger.error(f"Coordination loop error: {e}")
                time.sleep(5.0)

    def _process_coordination_task(self, task: Dict[str, Any]):
        """Process a coordination task"""
        task_type = task.get('type', 'unknown')

        if task_type == 'role_assignment':
            self._assign_roles(task)
        elif task_type == 'workload_balance':
            self._balance_agent_workload()
        elif task_type == 'performance_review':
            self._review_agent_performance()
        else:
            logger.warning(f"Unknown coordination task type: {task_type}")

    def _assign_roles(self, task: Dict[str, Any]):
        """Assign roles to agents for specific debate"""
        try:
            required_roles = task.get('required_roles', {})

            # Simple role assignment based on agent capabilities
            for role, count in required_roles.items():
                suitable_agents = [
                    agent for agent in self.agents 
                    if agent.role.value == role and not agent.active
                ]

                for i, agent in enumerate(suitable_agents[:count]):
                    agent.active = True
                    logger.debug(f"Assigned {role} role to agent {agent.agent_id}")

        except Exception as e:
            logger.error(f"Role assignment failed: {e}")

    def _monitor_agent_health(self):
        """Monitor agent health and performance"""
        try:
            for agent in self.agents:
                # Simple health check based on recent activity
                if agent.active and len(agent.argument_history) == 0:
                    logger.warning(f"Agent {agent.agent_id} appears inactive")

                # Check credibility score
                if agent.credibility_score < 0.2:
                    logger.warning(f"Agent {agent.agent_id} has low credibility: {agent.credibility_score}")

        except Exception as e:
            logger.error(f"Agent health monitoring failed: {e}")

    def _balance_agent_workload(self):
        """Balance workload across agents"""
        try:
            # Calculate workload distribution
            workloads = [agent.arguments_made for agent in self.agents]

            if not workloads:
                return

            avg_workload = sum(workloads) / len(workloads)
            max_workload = max(workloads)

            # If imbalance is significant, log it
            if max_workload > avg_workload * 2:
                logger.info("Significant workload imbalance detected among agents")

        except Exception as e:
            logger.error(f"Workload balancing failed: {e}")

    def _review_agent_performance(self):
        """Review and update agent performance metrics"""
        try:
            for agent in self.agents:
                # Update performance metrics
                if agent.arguments_made > 0:
                    success_rate = agent.successful_arguments / agent.arguments_made

                    # Adjust credibility based on success rate
                    if success_rate > 0.8:
                        agent.credibility_score = min(1.0, agent.credibility_score * 1.05)
                    elif success_rate < 0.3:
                        agent.credibility_score = max(0.1, agent.credibility_score * 0.95)

        except Exception as e:
            logger.error(f"Performance review failed: {e}")

    def get_coordination_status(self) -> Dict[str, Any]:
        """Get coordination system status"""
        with self.lock:
            return {
                'coordination_active': self.coordination_active,
                'agents_count': len(self.agents),
                'active_agents': sum(1 for agent in self.agents if agent.active),
                'task_queue_size': self.task_queue.qsize(),
                'agent_summary': {
                    agent.agent_id: {
                        'role': agent.role.value,
                        'active': agent.active,
                        'credibility': agent.credibility_score,
                        'arguments_made': agent.arguments_made
                    }
                    for agent in self.agents
                }
            }

class ConsensusAlgorithm:
    """Decision-making through debate consensus algorithm"""

    def __init__(self, consensus_threshold: float = 0.75):
        self.consensus_threshold = consensus_threshold
        self.consensus_history: List[ConsensusResult] = []
        self.algorithm_metrics = {
            'total_decisions': 0,
            'successful_consensus': 0,
            'average_confidence': 0.0,
            'average_consensus_time': 0.0
        }

        logger.info(f"ConsensusAlgorithm initialized with threshold {consensus_threshold}")

    def calculate_consensus(self, debate_data: Dict[str, Any]) -> ConsensusResult:
        """Calculate consensus from debate data"""
        try:
            start_time = time.time()

            # Extract arguments and votes
            arguments = debate_data.get('arguments', [])
            votes = debate_data.get('votes', {})

            if not arguments:
                return self._create_empty_consensus("No arguments provided")

            # Analyze argument strengths
            argument_scores = self._analyze_argument_strengths(arguments, votes)

            # Identify positions
            positions = self._identify_positions(arguments, argument_scores)

            # Calculate position strengths
            position_strengths = self._calculate_position_strengths(positions)

            # Determine winning position
            winning_position = self._determine_winning_position(position_strengths)

            # Calculate consensus metrics
            consensus_level = self._calculate_consensus_level(position_strengths)
            confidence = self._calculate_confidence(winning_position, position_strengths)

            # Validate consensus
            validation_score = self._validate_consensus(
                winning_position, consensus_level, confidence, arguments
            )

            # Create consensus result
            consensus = ConsensusResult(
                decision=winning_position['description'],
                confidence=confidence,
                supporting_arguments=winning_position['arguments'][:5],
                dissenting_opinions=self._extract_dissenting_opinions(arguments, winning_position),
                consensus_level=consensus_level,
                validation_score=validation_score
            )

            # Update metrics
            consensus_time = time.time() - start_time
            self._update_metrics(consensus, consensus_time)

            # Store in history
            self.consensus_history.append(consensus)
            if len(self.consensus_history) > 1000:
                self.consensus_history.pop(0)

            logger.info(f"Consensus calculated: {consensus.decision[:50]}... (level: {consensus_level:.2f})")
            return consensus

        except Exception as e:
            logger.error(f"Consensus calculation failed: {e}")
            return self._create_empty_consensus(f"Error: {e}")

    def _analyze_argument_strengths(self, arguments: List[Argument], votes: Dict[str, Any]) -> Dict[str, float]:
        """Analyze the strength of each argument"""
        argument_scores = {}

        for argument in arguments:
            # Base score from argument properties
            base_score = (argument.strength + argument.credibility_score) / 2

            # Vote-based adjustment
            total_votes = argument.support_votes + argument.opposition_votes
            if total_votes > 0:
                vote_ratio = argument.support_votes / total_votes
                vote_adjustment = (vote_ratio - 0.5) * 0.4  # Max ±0.2 adjustment
                base_score += vote_adjustment

            # Evidence bonus
            evidence_bonus = min(len(argument.evidence) * 0.05, 0.2)

            # Final score
            final_score = min(1.0, max(0.0, base_score + evidence_bonus))
            argument_scores[argument.id] = final_score

        return argument_scores

    def _identify_positions(self, arguments: List[Argument], scores: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Identify distinct positions from arguments"""
        positions = {}

        # Group arguments by agent roles/positions (simplified)
        for argument in arguments:
            # Simple position identification based on agent ID
            position_key = 'pro' if argument.agent_id < 8 else 'con'

            if position_key not in positions:
                positions[position_key] = {
                    'arguments': [],
                    'total_score': 0.0,
                    'agent_ids': set()
                }

            positions[position_key]['arguments'].append(argument)
            positions[position_key]['total_score'] += scores.get(argument.id, 0.0)
            positions[position_key]['agent_ids'].add(argument.agent_id)

        return positions

    def _calculate_position_strengths(self, positions: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate the strength of each position"""
        position_strengths = {}

        for position_key, position_data in positions.items():
            arguments = position_data['arguments']
            total_score = position_data['total_score']

            if not arguments:
                position_strengths[position_key] = 0.0
                continue

            # Average argument strength
            avg_strength = total_score / len(arguments)

            # Agent diversity bonus
            agent_diversity = len(position_data['agent_ids']) / max(len(arguments), 1)
            diversity_bonus = min(agent_diversity * 0.1, 0.2)

            # Final position strength
            position_strengths[position_key] = min(1.0, avg_strength + diversity_bonus)

        return position_strengths

    def _determine_winning_position(self, position_strengths: Dict[str, float]) -> Dict[str, Any]:
        """Determine the winning position"""
        if not position_strengths:
            return {
                'key': 'none',
                'description': 'No clear position',
                'strength': 0.0,
                'arguments': []
            }

        winning_key = max(position_strengths.keys(), key=lambda k: position_strengths[k])
        winning_strength = position_strengths[winning_key]

        return {
            'key': winning_key,
            'description': f"Position {winning_key} with strength {winning_strength:.2f}",
            'strength': winning_strength,
            'arguments': []  # Would be populated with actual position arguments
        }

    def _calculate_consensus_level(self, position_strengths: Dict[str, float]) -> float:
        """Calculate the level of consensus"""
        if len(position_strengths) < 2:
            return 1.0

        strengths = list(position_strengths.values())
        strengths.sort(reverse=True)

        # Consensus is higher when there's a clear winner
        if len(strengths) >= 2:
            consensus_level = (strengths[0] - strengths[1]) / max(strengths[0], 0.1)
        else:
            consensus_level = strengths[0] if strengths else 0.0

        return min(1.0, max(0.0, consensus_level))

    def _calculate_confidence(self, winning_position: Dict[str, Any], position_strengths: Dict[str, float]) -> float:
        """Calculate confidence in the consensus"""
        winning_strength = winning_position['strength']

        # Base confidence from winning position strength
        base_confidence = winning_strength

        # Adjustment based on competition
        if len(position_strengths) > 1:
            strengths = list(position_strengths.values())
            strengths.sort(reverse=True)
            competition_factor = 1.0 - (strengths[1] / max(strengths[0], 0.1))
            base_confidence *= (0.5 + 0.5 * competition_factor)

        return min(1.0, max(0.0, base_confidence))

    def _validate_consensus(self, winning_position: Dict[str, Any], consensus_level: float, 
                          confidence: float, arguments: List[Argument]) -> float:
        """Validate the consensus result"""
        validation_score = 0.0

        # Validation based on consensus level
        validation_score += consensus_level * 0.4

        # Validation based on confidence
        validation_score += confidence * 0.3

        # Validation based on argument quality
        if arguments:
            avg_credibility = sum(arg.credibility_score for arg in arguments) / len(arguments)
            validation_score += avg_credibility * 0.3

        return min(1.0, max(0.0, validation_score))

    def _extract_dissenting_opinions(self, arguments: List[Argument], winning_position: Dict[str, Any]) -> List[str]:
        """Extract dissenting opinions from arguments"""
        dissenting = []

        # Find arguments that oppose the winning position (simplified)
        for argument in arguments:
            if argument.credibility_score < 0.4:  # Low credibility indicates dissent
                dissenting.append(f"Agent {argument.agent_id}: {argument.content[:100]}...")

        return dissenting[:5]  # Limit to 5 dissenting opinions

    def _create_empty_consensus(self, reason: str) -> ConsensusResult:
        """Create an empty consensus result"""
        return ConsensusResult(
            decision=f"No consensus reached: {reason}",
            confidence=0.0,
            supporting_arguments=[],
            dissenting_opinions=[reason],
            consensus_level=0.0,
            validation_score=0.0
        )

    def _update_metrics(self, consensus: ConsensusResult, consensus_time: float):
        """Update algorithm performance metrics"""
        self.algorithm_metrics['total_decisions'] += 1

        if consensus.consensus_level >= self.consensus_threshold:
            self.algorithm_metrics['successful_consensus'] += 1

        # Update average confidence
        total_decisions = self.algorithm_metrics['total_decisions']
        current_avg_confidence = self.algorithm_metrics['average_confidence']
        self.algorithm_metrics['average_confidence'] = (
            (current_avg_confidence * (total_decisions - 1) + consensus.confidence) / total_decisions
        )

        # Update average consensus time
        current_avg_time = self.algorithm_metrics['average_consensus_time']
        self.algorithm_metrics['average_consensus_time'] = (
            (current_avg_time * (total_decisions - 1) + consensus_time) / total_decisions
        )

    def get_algorithm_performance(self) -> Dict[str, Any]:
        """Get consensus algorithm performance metrics"""
        success_rate = (
            self.algorithm_metrics['successful_consensus'] / 
            max(self.algorithm_metrics['total_decisions'], 1)
        )

        return {
            'consensus_threshold': self.consensus_threshold,
            'total_decisions': self.algorithm_metrics['total_decisions'],
            'successful_consensus': self.algorithm_metrics['successful_consensus'],
            'success_rate': success_rate,
            'average_confidence': self.algorithm_metrics['average_confidence'],
            'average_consensus_time': self.algorithm_metrics['average_consensus_time'],
            'recent_consensus_count': len(self.consensus_history)
        }

class DEBATETeam:
    """
    DEBATE Team - 16-Agent Debate System
    Main coordination class for the DEBATE team implementation.
    """

    def __init__(self):
        # Core components
        self.debate_engine = DebateEngine()
        self.debate_visualizer = DebateVisualizer()
        self.agent_coordinator = AgentCoordinator(self.debate_engine.agents)
        self.consensus_algorithm = ConsensusAlgorithm()

        # Team state
        self.active = False
        self.current_debates: Dict[str, Dict[str, Any]] = {}

        # Performance metrics
        self.performance_metrics = {
            'debates_completed': 0,
            'consensus_rate': 0.0,
            'average_agent_participation': 0.0,
            'decision_quality_score': 0.0,
            'multi_perspective_coverage': 0.0
        }

        # Threading
        self.main_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()

        logger.info("DEBATE Team initialized with 16-agent debate system")

    def start(self):
        """Start DEBATE team operations"""
        if not self.active:
            self.active = True

            # Start coordination systems
            self.agent_coordinator.start_coordination()

            # Start main processing thread
            self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
            self.main_thread.start()

            logger.info("DEBATE Team started")

    def stop(self):
        """Stop DEBATE team operations"""
        self.active = False

        # Stop coordination
        self.agent_coordinator.stop_coordination()

        # Wait for main thread
        if self.main_thread:
            self.main_thread.join(timeout=5.0)

        logger.info("DEBATE Team stopped")

    def initiate_debate(self, topic: str, context: Dict[str, Any] = None) -> str:
        """Initiate a new debate on the given topic"""
        try:
            debate_id = self.debate_engine.start_debate(topic, context)

            with self.lock:
                self.current_debates[debate_id] = {
                    'topic': topic,
                    'context': context or {},
                    'start_time': time.time(),
                    'status': 'active'
                }

            logger.info(f"Initiated debate: {debate_id} on topic: {topic}")
            return debate_id

        except Exception as e:
            logger.error(f"Failed to initiate debate: {e}")
            return f"Error: {e}"

    def get_debate_consensus(self, debate_id: str) -> Optional[ConsensusResult]:
        """Get consensus result for a specific debate"""
        try:
            # Force consensus if debate is still active
            if self.debate_engine.current_debate and self.debate_engine.current_debate['id'] == debate_id:
                return self.debate_engine.force_consensus()

            # Look for completed debate in history
            for debate_record in self.debate_engine.debate_history:
                if debate_record['id'] == debate_id:
                    return debate_record.get('consensus')

            return None

        except Exception as e:
            logger.error(f"Failed to get consensus for debate {debate_id}: {e}")
            return None

    def validate_decision(self, decision: str, supporting_evidence: List[str]) -> Dict[str, Any]:
        """Validate a decision through multi-agent analysis"""
        try:
            # Create validation context
            validation_context = {
                'decision': decision,
                'evidence': supporting_evidence,
                'validation_request': True
            }

            # Start validation debate
            validation_topic = f"Validation of decision: {decision}"
            debate_id = self.initiate_debate(validation_topic, validation_context)

            # Wait for validation to complete (simplified)
            time.sleep(5.0)  # In real implementation, this would be event-driven

            # Get validation result
            consensus = self.get_debate_consensus(debate_id)

            if consensus:
                return {
                    'validated': consensus.consensus_level >= 0.7,
                    'validation_score': consensus.validation_score,
                    'confidence': consensus.confidence,
                    'supporting_arguments': [arg.content for arg in consensus.supporting_arguments],
                    'concerns': consensus.dissenting_opinions
                }
            else:
                return {
                    'validated': False,
                    'validation_score': 0.0,
                    'error': 'Validation consensus not reached'
                }

        except Exception as e:
            logger.error(f"Decision validation failed: {e}")
            return {
                'validated': False,
                'validation_score': 0.0,
                'error': str(e)
            }

    def analyze_multiple_perspectives(self, topic: str, perspectives: List[str]) -> Dict[str, Any]:
        """Analyze a topic from multiple perspectives"""
        try:
            analysis_results = {}

            for i, perspective in enumerate(perspectives):
                # Create perspective-specific context
                context = {
                    'perspective': perspective,
                    'other_perspectives': [p for j, p in enumerate(perspectives) if j != i],
                    'analysis_mode': True
                }

                # Start perspective analysis
                analysis_topic = f"Analysis from {perspective} perspective: {topic}"
                debate_id = self.initiate_debate(analysis_topic, context)

                # Store analysis reference
                analysis_results[perspective] = {
                    'debate_id': debate_id,
                    'status': 'analyzing'
                }

            # Wait for analyses to complete
            time.sleep(10.0)  # Simplified timing

            # Collect results
            for perspective, analysis_data in analysis_results.items():
                consensus = self.get_debate_consensus(analysis_data['debate_id'])
                if consensus:
                    analysis_data.update({
                        'status': 'completed',
                        'conclusion': consensus.decision,
                        'confidence': consensus.confidence,
                        'key_arguments': [arg.content for arg in consensus.supporting_arguments[:3]]
                    })
                else:
                    analysis_data['status'] = 'failed'

            return {
                'topic': topic,
                'perspectives_analyzed': len(perspectives),
                'analysis_results': analysis_results,
                'synthesis_available': all(
                    result['status'] == 'completed' 
                    for result in analysis_results.values()
                )
            }

        except Exception as e:
            logger.error(f"Multi-perspective analysis failed: {e}")
            return {'error': str(e)}

    def get_team_status(self) -> Dict[str, Any]:
        """Get comprehensive DEBATE team status"""
        with self.lock:
            # Get component statuses
            debate_status = self.debate_engine.get_debate_status()
            coordination_status = self.agent_coordinator.get_coordination_status()
            consensus_performance = self.consensus_algorithm.get_algorithm_performance()
            visualization_summary = self.debate_visualizer.get_visualization_summary()

            return {
                'active': self.active,
                'performance_metrics': self.performance_metrics.copy(),
                'current_debates': {
                    debate_id: {
                        'topic': debate_data['topic'],
                        'duration': time.time() - debate_data['start_time'],
                        'status': debate_data['status']
                    }
                    for debate_id, debate_data in self.current_debates.items()
                },
                'debate_engine': debate_status,
                'agent_coordination': coordination_status,
                'consensus_algorithm': consensus_performance,
                'visualization': visualization_summary,
                'agents_summary': {
                    'total_agents': len(self.debate_engine.agents),
                    'active_agents': sum(1 for agent in self.debate_engine.agents if agent.active),
                    'role_distribution': {
                        role.value: sum(1 for agent in self.debate_engine.agents if agent.role == role)
                        for role in AgentRole
                    }
                }
            }

    def _main_loop(self):
        """Main processing loop for DEBATE team"""
        while self.active:
            try:
                # Update performance metrics
                self._update_performance_metrics()

                # Clean up completed debates
                self._cleanup_completed_debates()

                # Monitor agent performance
                self._monitor_agent_performance()

                # Update team coordination
                self._update_team_coordination()

                time.sleep(2.0)  # Main loop interval

            except Exception as e:
                logger.error(f"DEBATE main loop error: {e}")
                time.sleep(5.0)

    def _update_performance_metrics(self):
        """Update team performance metrics"""
        try:
            with self.lock:
                # Get debate engine metrics
                engine_metrics = self.debate_engine.get_performance_metrics()

                # Update team metrics
                self.performance_metrics['debates_completed'] = engine_metrics['debates_conducted']
                self.performance_metrics['consensus_rate'] = engine_metrics['consensus_rate']

                # Calculate agent participation
                total_agents = len(self.debate_engine.agents)
                active_agents = sum(1 for agent in self.debate_engine.agents if agent.arguments_made > 0)
                self.performance_metrics['average_agent_participation'] = active_agents / max(total_agents, 1)

                # Calculate decision quality (simplified)
                consensus_performance = self.consensus_algorithm.get_algorithm_performance()
                self.performance_metrics['decision_quality_score'] = consensus_performance['average_confidence']

                # Multi-perspective coverage (simplified)
                role_coverage = len(set(agent.role for agent in self.debate_engine.agents if agent.active))
                max_roles = len(AgentRole)
                self.performance_metrics['multi_perspective_coverage'] = role_coverage / max_roles

        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")

    def _cleanup_completed_debates(self):
        """Clean up completed debates"""
        try:
            with self.lock:
                completed_debates = []

                for debate_id, debate_data in self.current_debates.items():
                    # Check if debate is completed
                    if not self.debate_engine.active or debate_data.get('status') == 'completed':
                        completed_debates.append(debate_id)

                # Remove completed debates
                for debate_id in completed_debates:
                    del self.current_debates[debate_id]
                    logger.debug(f"Cleaned up completed debate: {debate_id}")

        except Exception as e:
            logger.error(f"Debate cleanup failed: {e}")

    def _monitor_agent_performance(self):
        """Monitor individual agent performance"""
        try:
            for agent in self.debate_engine.agents:
                # Check for underperforming agents
                if agent.arguments_made > 10 and agent.credibility_score < 0.3:
                    logger.warning(f"Agent {agent.agent_id} has low credibility: {agent.credibility_score}")

                # Check for inactive agents
                if agent.active and agent.arguments_made == 0:
                    logger.warning(f"Agent {agent.agent_id} is active but not participating")

        except Exception as e:
            logger.error(f"Agent performance monitoring failed: {e}")

    def _update_team_coordination(self):
        """Update team coordination tasks"""
        try:
            # Submit coordination tasks
            coordination_tasks = [
                {'type': 'workload_balance'},
                {'type': 'performance_review'}
            ]

            for task in coordination_tasks:
                self.agent_coordinator.assign_task(task)

        except Exception as e:
            logger.error(f"Team coordination update failed: {e}")
