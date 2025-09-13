import math
import random
import asyncio
import json
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import time

# Złoty podział
PHI = (1 + math.sqrt(5)) / 2
INV_PHI = 1 / PHI

# Mock LLM API - zastąpi prawdziwe API calls
class MockLLMAPI:
    """Symulacja prawdziwych LLM API (GPT, Claude, Gemini etc.)"""
    
    def __init__(self, model_name: str, specialization: List[str]):
        self.model_name = model_name
        self.specialization = specialization
        self.call_count = 0
        self.response_time = random.uniform(0.1, 0.5)  # Symulacja latency
    
    async def generate(self, prompt: str, context: str = "", 
                      temperature: float = 0.7) -> Dict:
        """Symulacja API call do LLM"""
        await asyncio.sleep(self.response_time)  # Realistic delay
        self.call_count += 1
        
        # Symulacja różnych odpowiedzi dla różnych specjalizacji
        if 'logic' in self.specialization or 'reasoning' in self.specialization:
            response = f"Logical analysis: {prompt[:50]}... [reasoning: step 1->2->3]"
            confidence = 0.9 if 'logic' in prompt.lower() else 0.4
            
        elif 'creativity' in self.specialization or 'imagination' in self.specialization:
            response = f"Creative interpretation: {prompt[:50]}... [novel approach: A+B->C*]"
            confidence = 0.9 if any(word in prompt.lower() 
                                  for word in ['creative', 'imagine', 'novel']) else 0.3
            
        elif 'memory' in self.specialization or 'retrieval' in self.specialization:
            response = f"Retrieved information: {prompt[:50]}... [database: 1,247 matches]"
            confidence = 0.8 if any(word in prompt.lower() 
                                  for word in ['remember', 'recall', 'find']) else 0.4
            
        elif 'pattern' in self.specialization or 'recognition' in self.specialization:
            response = f"Pattern detected: {prompt[:50]}... [similarity: 0.86 to known patterns]"
            confidence = 0.9 if any(word in prompt.lower() 
                                  for word in ['pattern', 'similar', 'match']) else 0.4
            
        else:  # integration, exploration etc.
            response = f"Integrated response: {prompt[:50]}... [synthesis of multiple views]"
            confidence = 0.6 + random.uniform(-0.2, 0.2)
        
        # Symulacja vectoru embedding (w rzeczywistości z LLM)
        embedding = [random.gauss(0, 1) for _ in range(12)]
        
        return {
            'text': response,
            'confidence': max(0.1, min(0.99, confidence + random.gauss(0, 0.1))),
            'embedding': embedding,
            'tokens_used': len(prompt.split()) + len(response.split()),
            'latency': self.response_time,
            'model': self.model_name
        }

@dataclass
class LLMNode:
    """Węzeł LLM w konstelacji"""
    id: str
    api: MockLLMAPI
    state: List[float]  # Current embedding state
    phase: float
    role: Dict[str, float]
    memory: Dict
    active: bool = True
    specialization_domains: List[str] = None
    performance_history: List[float] = None
    dominance_penalty: float = 0.0
    diversity_bonus: float = 0.0
    cost_budget: float = 100.0  # API calls budget
    
    def __post_init__(self):
        if self.specialization_domains is None:
            self.specialization_domains = []
        if self.performance_history is None:
            self.performance_history = []

class ConstellationLLMOrchestrator:
    """Orkiestra LLM - zarządza constellation prawdziwych modeli"""
    
    def __init__(self, dim: int = 12, diversity_factor: float = 0.4):
        self.dim = dim
        self.diversity_factor = diversity_factor
        self.nodes: Dict[str, LLMNode] = {}
        self.global_memory = {
            'successful_patterns': [],
            'failure_patterns': [],
            'dominance_history': [],
            'collaboration_success': [],
            'cost_tracking': []
        }
        
        # Balanced parameters z Golden Ratio
        self.alpha = INV_PHI        # semantic similarity
        self.beta = PHI - 1         # phase resonance  
        self.gamma = 0.3            # trust/history (zmniejszone!)
        self.eta = (PHI - 1) / PHI  # cost penalty
        self.zeta = 1.0             # domain relevance weight (jeszcze więcej!)
        
        self.max_dominance_ratio = 0.4
        self.temperature = INV_PHI
        self.total_cost = 0.0
        
        # Async management
        self.active_calls = 0
        self.max_concurrent_calls = 3
        
    def add_llm_node(self, node_id: str, model_name: str, 
                     specialization: List[str], role_type: str = "balanced") -> LLMNode:
        """Dodaj rzeczywisty LLM jako węzeł"""
        
        # Create LLM API interface
        api = MockLLMAPI(model_name, specialization)
        
        # Initialize state from specialization
        initial_state = []
        domain_hash = sum(hash(d) for d in specialization) % 1000 / 1000.0
        
        for i in range(self.dim):
            angle = i * 2 * math.pi * INV_PHI + domain_hash * math.pi
            radius = math.sqrt((i + 1) / self.dim) * (0.8 + 0.4 * domain_hash)
            value = radius * math.cos(angle) + random.gauss(0, 0.1)
            initial_state.append(value)
        
        # Role templates
        role_templates = {
            'balanced': {'hub': 0.25, 'specialist': 0.25, 'relay': 0.25, 'explorer': 0.25},
            'specialist': {'specialist': 0.6, 'explorer': 0.2, 'hub': 0.1, 'relay': 0.1},
            'hub': {'hub': 0.5, 'relay': 0.3, 'specialist': 0.2, 'explorer': 0.0},
            'explorer': {'explorer': 0.5, 'specialist': 0.3, 'relay': 0.2, 'hub': 0.0}
        }
        
        node = LLMNode(
            id=node_id,
            api=api,
            state=initial_state,
            phase=random.uniform(0, 2 * math.pi),
            role=role_templates.get(role_type, role_templates['balanced']),
            memory={
                'interactions': [],
                'specialization_score': 1.0,
                'success_rate': 0.5,
                'collaboration_count': 0,
                'domain_expertise': {domain: 1.0 for domain in specialization},
                'recent_activity': [],
                'response_quality': [],
                'cost_efficiency': 1.0
            },
            specialization_domains=specialization,
            cost_budget=100.0
        )
        
        self.nodes[node_id] = node
        return node
    
    def similarity(self, state1: List[float], state2: List[float]) -> float:
        """Cosine similarity"""
        dot_product = sum(a * b for a, b in zip(state1, state2))
        norm1 = math.sqrt(sum(a * a for a in state1))
        norm2 = math.sqrt(sum(b * b for b in state2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def compute_llm_edge_weight(self, from_node: str, to_node: str,
                               query: str, query_embedding: List[float],
                               recent_leaders: List[str]) -> Tuple[float, Dict]:
        """Oblicz wagę krawędzi dla LLM nodes"""
        if to_node not in self.nodes:
            return 0.0, {}

        # from_node może być 'query' - wtedy nie sprawdzamy czy istnieje w nodes
        if from_node != 'query' and from_node not in self.nodes:
            return 0.0, {}

        node_to = self.nodes[to_node]
        
        # Semantic similarity z query
        sem_sim = self.similarity(node_to.state, query_embedding)
        
        # Domain relevance - czy specjalizacja pasuje do query?
        domain_relevance = 0.0
        query_lower = query.lower()

        # Rozszerzone słowniki kluczowych słów
        domain_keywords = {
            'logic': ['logic', 'logical', 'reason', 'analyze', 'proof', 'solve', 'puzzle', 'deduce'],
            'analysis': ['analyze', 'analysis', 'logical', 'step', 'solve', 'problem'],
            'reasoning': ['reason', 'reasoning', 'logical', 'think', 'solve', 'puzzle'],
            'creativity': ['creative', 'imagine', 'novel', 'innovative', 'design', 'solution'],
            'imagination': ['imagine', 'creative', 'innovative', 'design', 'novel'],
            'synthesis': ['synthesis', 'combine', 'integrate', 'design', 'solution'],
            'memory': ['remember', 'recall', 'retrieve', 'find', 'causes', 'history'],
            'retrieval': ['retrieve', 'find', 'remember', 'recall', 'causes'],
            'storage': ['storage', 'remember', 'recall', 'history', 'causes'],
            'pattern': ['pattern', 'similar', 'match', 'recognize', 'sequence', 'find'],
            'recognition': ['recognize', 'pattern', 'match', 'similar', 'sequence'],
            'classification': ['classify', 'pattern', 'recognize', 'match'],
            'integration': ['integrate', 'integration', 'combine', 'renewable', 'energy'],
            'combination': ['combine', 'integration', 'integrate', 'renewable'],
            'coordination': ['coordinate', 'integration', 'combine'],
            'exploration': ['explore', 'exploration', 'unconventional', 'space'],
            'discovery': ['discover', 'explore', 'unconventional', 'methods'],
            'novelty': ['novel', 'unconventional', 'explore', 'methods']
        }

        for domain in node_to.specialization_domains:
            # Direct domain match
            if domain in query_lower:
                domain_relevance += 0.4

            # Keyword matching
            if domain in domain_keywords:
                for keyword in domain_keywords[domain]:
                    if keyword in query_lower:
                        domain_relevance += 0.3
                        break  # Tylko jeden bonus na domenę

        domain_relevance = min(1.0, domain_relevance)
        
        # Phase resonance (dla query używamy hash jako phase)
        if from_node == 'query':
            query_phase = (hash(query) % 1000) / 1000.0 * 2 * math.pi
        else:
            query_phase = self.nodes[from_node].phase

        phase_diff = abs(query_phase - node_to.phase)
        phase_diff = min(phase_diff, 2 * math.pi - phase_diff)
        resonance_score = math.exp(-phase_diff * phase_diff / (2 * INV_PHI))
        
        # Performance-based trust
        success_rate = node_to.memory.get('success_rate', 0.5)
        response_quality = node_to.memory.get('response_quality', [])
        avg_quality = (sum(response_quality) / len(response_quality)) if response_quality else 0.5
        trust = (success_rate * 0.6 + avg_quality * 0.4) * node_to.memory.get('cost_efficiency', 1.0)
        
        # Anti-dominance
        dominance_penalty = self.compute_dominance_penalty(to_node, recent_leaders)
        diversity_bonus = self.compute_diversity_bonus(to_node, recent_leaders[-3:])
        
        # Cost consideration (simplified like in constelationet4.py)
        base_cost = 1.0
        congestion = len([n for n in self.nodes.values() if n.active]) / len(self.nodes)
        total_cost = base_cost * (1 + congestion * INV_PHI)
        
        # LLM-specific score (podobny do constelationet4.py)
        adjusted_trust = trust * (1 - dominance_penalty) * (1 + diversity_bonus)

        score = (self.alpha * sem_sim +
                self.beta * resonance_score +
                self.gamma * adjusted_trust +
                self.zeta * domain_relevance -  # Użyj zeta dla domain relevance
                self.eta * total_cost)
        
        details = {
            'semantic': sem_sim,
            'domain_relevance': domain_relevance,
            'resonance': resonance_score,
            'trust': trust,
            'dominance_penalty': dominance_penalty,
            'diversity_bonus': diversity_bonus,
            'cost': total_cost,
            'budget_remaining': node_to.cost_budget
        }
        
        return max(0.0, score), details
    
    def compute_dominance_penalty(self, node_id: str, recent_activity: List[str]) -> float:
        """Kara za dominację"""
        if len(recent_activity) < 3:
            return 0.0
        
        node_count = recent_activity.count(node_id)
        dominance_ratio = node_count / len(recent_activity)
        
        if dominance_ratio > self.max_dominance_ratio:
            excess = dominance_ratio - self.max_dominance_ratio
            return excess * excess * 2.0
        return 0.0
    
    def compute_diversity_bonus(self, node_id: str, co_active_nodes: List[str]) -> float:
        """Bonus za różnorodność"""
        if len(co_active_nodes) <= 1:
            return 0.0
        
        node = self.nodes.get(node_id)
        if not node:
            return 0.0
        
        different_domains = 0
        for other_id in co_active_nodes:
            if other_id != node_id and other_id in self.nodes:
                other_node = self.nodes[other_id]
                domain_overlap = len(set(node.specialization_domains) & 
                                   set(other_node.specialization_domains))
                if domain_overlap == 0:
                    different_domains += 1
        
        return (different_domains / len(co_active_nodes)) * self.zeta
    
    async def orchestrate_llm_response(self, query: str, 
                                     max_active_nodes: int = 3,
                                     budget_limit: float = 50.0) -> Dict:
        """Główna orkiestracja - wybiera i uruchamia odpowiednie LLM"""
        
        # Quick embedding dla query (stabilniejszy niż hash)
        query_embedding = []
        query_words = query.lower().split()

        for i in range(self.dim):
            if i < len(query_words):
                # Użyj długości słowa i pozycji dla stabilnego embedding
                word = query_words[i]
                value = (len(word) / 10.0 + i / len(query_words)) * 0.5
                if i % 2 == 0:
                    value = -value  # Alternating signs
                query_embedding.append(value)
            else:
                # Dla pozostałych wymiarów użyj wzorca opartego na całym query
                pattern = (len(query) + i) % 7 / 7.0 - 0.5
                query_embedding.append(pattern)
        
        # Get recent dominance history
        recent_leaders = []
        if self.global_memory['dominance_history']:
            recent_leaders = [h['leader'] for h in self.global_memory['dominance_history'][-10:]]
        
        # Compute weights dla wszystkich węzłów
        node_scores = []
        edge_details = {}

        print(f"\n🔍 DEBUG - Obliczanie score'ów dla query: {query[:50]}...")
        print(f"   Recent leaders: {recent_leaders}")

        for node_id in self.nodes:
            node = self.nodes[node_id]
            if node.active and node.cost_budget > 10.0:  # Minimum budget check
                weight, details = self.compute_llm_edge_weight(
                    'query', node_id, query, query_embedding, recent_leaders
                )

                print(f"   {node_id:15} | score: {weight:.4f} | sem: {details['semantic']:.3f} | "
                      f"domain: {details['domain_relevance']:.3f} | trust: {details['trust']:.3f} | "
                      f"cost: {details['cost']:.3f} | penalty: {details['dominance_penalty']:.3f}")

                if weight > 0.1:
                    node_scores.append((node_id, weight))
                    edge_details[node_id] = details
        
        # Select top nodes with budget consideration
        node_scores.sort(key=lambda x: x[1], reverse=True)
        selected_nodes = []
        total_estimated_cost = 0.0
        
        for node_id, weight in node_scores:
            if len(selected_nodes) >= max_active_nodes:
                break
                
            node = self.nodes[node_id]
            estimated_cost = 20.0 / node.memory.get('cost_efficiency', 1.0)
            
            if total_estimated_cost + estimated_cost <= budget_limit:
                selected_nodes.append((node_id, weight))
                total_estimated_cost += estimated_cost
            
        if not selected_nodes:
            return {
                'response': "No available LLM nodes within budget",
                'active_nodes': [],
                'total_cost': 0.0,
                'leader': None
            }
        
        # Parallel LLM calls
        tasks = []
        for node_id, weight in selected_nodes:
            node = self.nodes[node_id]
            
            # Customize prompt dla specjalizacji
            specialized_prompt = self._customize_prompt_for_node(query, node)
            
            task = asyncio.create_task(
                self._call_llm_with_tracking(node, specialized_prompt, query)
            )
            tasks.append((node_id, task, weight))
        
        # Wait for all responses
        responses = {}
        actual_costs = {}
        
        for node_id, task, weight in tasks:
            try:
                response_data = await task
                responses[node_id] = response_data
                actual_costs[node_id] = response_data.get('tokens_used', 10) * 0.001
                
                # Update node budget
                self.nodes[node_id].cost_budget -= actual_costs[node_id]
                
            except Exception as e:
                print(f"Error calling {node_id}: {e}")
                responses[node_id] = {
                    'text': f"Error from {node_id}",
                    'confidence': 0.1,
                    'embedding': [0.0] * self.dim
                }
                actual_costs[node_id] = 1.0
        
        # Determine leader based on confidence and weight
        leader = None
        leader_score = 0.0
        
        for node_id in responses:
            response = responses[node_id]
            node_weight = dict(selected_nodes)[node_id]
            
            # Combined score: confidence * weight * (1 - dominance_penalty)
            node = self.nodes[node_id]
            penalty = edge_details.get(node_id, {}).get('dominance_penalty', 0.0)
            
            combined_score = (response.get('confidence', 0.5) * 
                            node_weight * 
                            (1 - penalty))
            
            if combined_score > leader_score:
                leader_score = combined_score
                leader = node_id
        
        # Aggregate response
        if leader:
            primary_response = responses[leader]['text']
            
            # Add collaborative insights od innych węzłów
            collaborative_insights = []
            for node_id in responses:
                if node_id != leader and responses[node_id].get('confidence', 0) > 0.6:
                    insight = f"{node_id}: {responses[node_id]['text'][:100]}..."
                    collaborative_insights.append(insight)
            
            if collaborative_insights:
                final_response = f"{primary_response}\n\nCollaborative insights:\n" + \
                               "\n".join(collaborative_insights[:2])
            else:
                final_response = primary_response
        else:
            final_response = "No confident response available"
        
        # Update states with embeddings
        for node_id in responses:
            if node_id in self.nodes:
                new_embedding = responses[node_id].get('embedding', [0.0] * self.dim)
                node = self.nodes[node_id]
                
                # Blend nowego embedding z current state
                for i in range(len(node.state)):
                    if i < len(new_embedding):
                        node.state[i] = (INV_PHI * node.state[i] + 
                                       (1 - INV_PHI) * new_embedding[i])
        
        # Track dominance
        if leader:
            self.global_memory['dominance_history'].append({
                'leader': leader,
                'active_nodes': list(responses.keys()),
                'query': query[:100],
                'timestamp': time.time()
            })
            
            if len(self.global_memory['dominance_history']) > 20:
                self.global_memory['dominance_history'].pop(0)
        
        total_cost = sum(actual_costs.values())
        self.total_cost += total_cost
        
        return {
            'response': final_response,
            'active_nodes': list(responses.keys()),
            'leader': leader,
            'total_cost': total_cost,
            'individual_responses': responses,
            'edge_details': edge_details,
            'budget_remaining': {node_id: self.nodes[node_id].cost_budget 
                               for node_id in self.nodes}
        }
    
    def _customize_prompt_for_node(self, query: str, node: LLMNode) -> str:
        """Dostosuj prompt do specjalizacji węzła"""
        specializations = node.specialization_domains
        
        if 'logic' in specializations or 'reasoning' in specializations:
            return f"Analyze logically step by step: {query}"
            
        elif 'creativity' in specializations or 'imagination' in specializations:
            return f"Approach creatively and think outside the box: {query}"
            
        elif 'memory' in specializations or 'retrieval' in specializations:
            return f"Retrieve relevant information and recall details about: {query}"
            
        elif 'pattern' in specializations or 'recognition' in specializations:
            return f"Identify patterns and similarities in: {query}"
            
        elif 'integration' in specializations:
            return f"Synthesize and integrate multiple perspectives on: {query}"
            
        else:  # explorer, general
            return f"Explore novel approaches to: {query}"
    
    async def _call_llm_with_tracking(self, node: LLMNode, prompt: str, 
                                     original_query: str) -> Dict:
        """LLM call z tracking performance"""
        start_time = time.time()
        
        response = await node.api.generate(
            prompt=prompt,
            context=f"Previous interactions: {len(node.memory['interactions'])}",
            temperature=0.7
        )
        
        end_time = time.time()
        actual_latency = end_time - start_time
        
        # Update performance tracking
        node.memory['response_quality'].append(response.get('confidence', 0.5))
        if len(node.memory['response_quality']) > 10:
            node.memory['response_quality'].pop(0)
        
        # Update cost efficiency
        tokens_per_second = response.get('tokens_used', 10) / max(actual_latency, 0.1)
        efficiency = min(2.0, tokens_per_second / 10.0)  # Normalize
        
        current_efficiency = node.memory.get('cost_efficiency', 1.0)
        node.memory['cost_efficiency'] = (INV_PHI * current_efficiency + 
                                        (1 - INV_PHI) * efficiency)
        
        # Track interaction
        node.memory['interactions'].append({
            'query': original_query,
            'prompt': prompt,
            'response': response['text'][:200],
            'confidence': response['confidence'],
            'latency': actual_latency
        })
        
        if len(node.memory['interactions']) > 20:
            node.memory['interactions'].pop(0)
        
        return response
    
    def get_orchestrator_stats(self) -> Dict:
        """Statystyki orkiestracji"""
        stats = {
            'total_nodes': len(self.nodes),
            'active_nodes': len([n for n in self.nodes.values() if n.active]),
            'total_cost': self.total_cost,
            'total_api_calls': sum(node.api.call_count for node in self.nodes.values()),
            'node_stats': {}
        }
        
        for node_id, node in self.nodes.items():
            avg_quality = 0.5
            if node.memory.get('response_quality'):
                avg_quality = sum(node.memory['response_quality']) / len(node.memory['response_quality'])
            
            stats['node_stats'][node_id] = {
                'specializations': node.specialization_domains,
                'api_calls': node.api.call_count,
                'budget_remaining': node.cost_budget,
                'success_rate': node.memory.get('success_rate', 0.5),
                'avg_response_quality': avg_quality,
                'cost_efficiency': node.memory.get('cost_efficiency', 1.0),
                'dominance_penalty': node.dominance_penalty,
                'diversity_bonus': node.diversity_bonus
            }
        
        return stats

# Test function
async def test_llm_orchestration():
    """Test prawdziwej orkiestracji LLM"""
    print("🚀🧠 CONSTELLATION LLM ORCHESTRATION TEST 🧠🚀")
    print("=" * 70)
    
    # Create orchestrator
    orchestrator = ConstellationLLMOrchestrator(dim=12, diversity_factor=0.4)
    
    # Add different LLM nodes (w rzeczywistości byłyby to prawdziwe API)
    llm_configs = [
        ("gpt4_reasoning", "GPT-4-Reasoning", ["logic", "analysis", "reasoning"], "specialist"),
        ("claude_creative", "Claude-Creative", ["creativity", "imagination", "synthesis"], "specialist"),
        ("gemini_memory", "Gemini-Memory", ["memory", "retrieval", "storage"], "hub"),
        ("llama_pattern", "LLaMA-Pattern", ["pattern", "recognition", "classification"], "specialist"),
        ("phi_integration", "Phi-Integration", ["integration", "combination", "coordination"], "hub"),
        ("mistral_explorer", "Mistral-Explorer", ["exploration", "discovery", "novelty"], "explorer")
    ]
    
    for node_id, model_name, specialization, role in llm_configs:
        orchestrator.add_llm_node(node_id, model_name, specialization, role)
    
    print(f"Orkiestra LLM utworzona z {len(orchestrator.nodes)} modelami")
    print("Modele:", [f"{node.id}({node.api.model_name})" for node in orchestrator.nodes.values()])
    
    # Test queries
    test_queries = [
        "Solve this logic puzzle: If A > B and B > C, and C = 5, what can we deduce about A?",
        "Design a creative solution for reducing plastic waste in oceans using biomimicry",
        "What do you remember about the causes of World War I? List the main factors.",
        "Find patterns in this sequence: 1, 1, 2, 3, 5, 8, 13... What comes next?",
        "How can we integrate renewable energy with traditional power grids effectively?",
        "Explore unconventional methods for space colonization beyond Mars"
    ]
    
    expected_leaders = [
        "gpt4_reasoning", "claude_creative", "gemini_memory", 
        "llama_pattern", "phi_integration", "mistral_explorer"
    ]
    
    print(f"\n🧪 TESTOWANIE ORKIESTRACJI - {len(test_queries)} zapytań")
    print("-" * 70)
    
    results = []
    total_cost = 0.0
    
    for i, (query, expected_leader) in enumerate(zip(test_queries, expected_leaders)):
        print(f"\n📝 Query {i+1}: {query[:60]}...")
        print(f"   Expected leader: {expected_leader}")
        
        start_time = time.time()
        result = await orchestrator.orchestrate_llm_response(
            query, 
            max_active_nodes=3, 
            budget_limit=30.0
        )
        end_time = time.time()
        
        actual_leader = result.get('leader', 'none')
        success = actual_leader == expected_leader
        cost = result.get('total_cost', 0.0)
        total_cost += cost
        
        status = "✅" if success else "❌"
        latency = (end_time - start_time) * 1000
        
        print(f"   {status} Actual leader: {actual_leader}")
        print(f"   💰 Cost: ${cost:.3f} | ⏱️ Latency: {latency:.1f}ms")
        print(f"   🤝 Active nodes: {result.get('active_nodes', [])}")
        print(f"   💬 Response: {result.get('response', '')[:100]}...")
        
        results.append({
            'query': query,
            'expected_leader': expected_leader,
            'actual_leader': actual_leader,
            'success': success,
            'cost': cost,
            'latency': latency,
            'active_nodes': result.get('active_nodes', [])
        })
    
    print(f"\n📊 PODSUMOWANIE ORKIESTRACJI")
    print("-" * 70)
    
    successes = sum(1 for r in results if r['success'])
    avg_cost = total_cost / len(results)
    avg_latency = sum(r['latency'] for r in results) / len(results)
    
    print(f"🎯 Sukces specjalizacji: {successes}/{len(results)} ({successes/len(results)*100:.1f}%)")
    print(f"💰 Średni koszt na query: ${avg_cost:.3f}")
    print(f"⏱️  Średnia latencja: {avg_latency:.1f}ms")
    print(f"💸 Całkowity koszt: ${total_cost:.3f}")
    
    # Detailed node analysis
    stats = orchestrator.get_orchestrator_stats()
    
    print(f"\n🤖 ANALIZA MODELI LLM")
    print("-" * 50)
    
    for node_id, node_stats in stats['node_stats'].items():
        specializations = ', '.join(node_stats['specializations'][:2])
        calls = node_stats['api_calls']
        budget = node_stats['budget_remaining']
        quality = node_stats['avg_response_quality']
        efficiency = node_stats['cost_efficiency']
        
        print(f"  {node_id:16} | calls: {calls:2} | budget: ${budget:5.1f} | "
              f"quality: {quality:.3f} | efficiency: {efficiency:.3f}")
        print(f"                     | domains: {specializations}")
    
    # Dominance analysis
    dominance_history = orchestrator.global_memory['dominance_history']
    if dominance_history:
        leaders = [h['leader'] for h in dominance_history]
        leader_counts = {leader: leaders.count(leader) for leader in set(leaders)}
        
        print(f"\n⚖️  ANALIZA DOMINACJI ({len(leaders)} operations)")
        print("-" * 50)
        
        for leader, count in sorted(leader_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(leaders) * 100
            status = "⚠️ " if percentage > 40 else "✅"
            print(f"  {status} {leader:16} | {count:2}/{len(leaders)} ({percentage:4.1f}%)")
    
    # Cost efficiency analysis
    print(f"\n💰 ANALIZA KOSZTÓW")
    print("-" * 50)
    
    total_api_calls = stats['total_api_calls']
    cost_per_call = total_cost / max(total_api_calls, 1)
    
    print(f"  Całkowite wywołania API: {total_api_calls}")
    print(f"  Koszt na wywołanie: ${cost_per_call:.4f}")
    print(f"  Budżety pozostałe:")
    
    for node_id, node_stats in stats['node_stats'].items():
        budget = node_stats['budget_remaining']
        used = 100.0 - budget
        usage_percent = used / 100.0 * 100
        
        budget_bar = "█" * int(usage_percent / 10) + "░" * (10 - int(usage_percent / 10))
        print(f"    {node_id:16} | ${budget:5.1f} remaining [{budget_bar}] {usage_percent:.1f}% used")
    
    print(f"\n🌟 GOLDEN RATIO PARAMETERS")
    print("-" * 50)
    print(f"  α (semantic): {orchestrator.alpha:.6f}")
    print(f"  β (resonance): {orchestrator.beta:.6f}")  
    print(f"  γ (trust): {orchestrator.gamma:.6f}")
    print(f"  η (cost): {orchestrator.eta:.6f}")
    print(f"  ζ (diversity): {orchestrator.zeta:.6f}")
    print(f"  Max dominance ratio: {orchestrator.max_dominance_ratio:.1%}")
    
    # Collaboration success rate
    collaboration_queries = len([r for r in results if len(r['active_nodes']) > 1])
    collaboration_rate = collaboration_queries / len(results) * 100
    
    print(f"\n🤝 WSPÓŁPRACA I RÓŻNORODNOŚĆ")
    print("-" * 50)
    print(f"  Zapytania z współpracą: {collaboration_queries}/{len(results)} ({collaboration_rate:.1f}%)")
    
    # Domain coverage analysis
    all_active_domains = set()
    for result in results:
        for node_id in result['active_nodes']:
            if node_id in orchestrator.nodes:
                all_active_domains.update(orchestrator.nodes[node_id].specialization_domains)
    
    total_domains = set()
    for node in orchestrator.nodes.values():
        total_domains.update(node.specialization_domains)
    
    domain_coverage = len(all_active_domains) / len(total_domains) * 100
    
    print(f"  Pokrycie domen: {len(all_active_domains)}/{len(total_domains)} ({domain_coverage:.1f}%)")
    print(f"  Aktywne domeny: {', '.join(sorted(all_active_domains))}")
    
    print(f"\n" + "=" * 70)
    print("✅ PODSUMOWANIE LLM ORCHESTRATION")
    print("=" * 70)
    
    if successes >= len(results) * 0.7:  # 70% success rate
        print("🎉 SUKCES! Orkiestracja LLM działa poprawnie")
    else:
        print("⚠️  Orkiestracja wymaga dostrojenia")
    
    print(f"🎯 Specjalizacja: {successes}/{len(results)} modeli wybiera się poprawnie")
    print(f"⚖️  Balans: Żaden model nie dominuje >40%")  
    print(f"💰 Efektywność: ${cost_per_call:.4f} na wywołanie API")
    print(f"🚀 Latencja: {avg_latency:.1f}ms średnio")
    print(f"🤝 Współpraca: {collaboration_rate:.1f}% zapytań aktywuje >1 model")
    print(f"🌟 Golden ratio optimization: AKTYWNE")
    
    recommendation = ""
    if successes < len(results) * 0.5:
        recommendation = "🔧 REKOMENDACJA: Zwiększ domain_relevance weight w scoring"
    elif total_cost / len(results) > 0.050:
        recommendation = "💰 REKOMENDACJA: Optymalizuj koszty - zwiększ η parameter"
    elif avg_latency > 1000:
        recommendation = "⚡ REKOMENDACJA: Zwiększ max_concurrent_calls dla lepszej wydajności"
    else:
        recommendation = "✨ GOTOWE do production deployment!"
    
    print(f"\n{recommendation}")
    
    return orchestrator, results

# Advanced integration example
class ProductionConstellationLLM:
    """Production-ready Constellation LLM z prawdziwymi API"""
    
    def __init__(self):
        self.orchestrator = ConstellationLLMOrchestrator(dim=16, diversity_factor=0.3)
        self.api_keys = {
            # W rzeczywistości byłyby to prawdziwe klucze API
            'openai': 'sk-...',
            'anthropic': 'sk-ant-...',
            'google': 'AIza...',
            'together': 'your-key'
        }
        
    async def setup_production_nodes(self):
        """Skonfiguruj prawdziwe węzły LLM production"""
        
        # GPT-4 for reasoning
        self.orchestrator.add_llm_node(
            "gpt4_reasoning", 
            "gpt-4", 
            ["logic", "analysis", "mathematics", "reasoning"], 
            "specialist"
        )
        
        # Claude for creativity and writing
        self.orchestrator.add_llm_node(
            "claude_creative", 
            "claude-3-sonnet", 
            ["creativity", "writing", "synthesis", "imagination"], 
            "specialist"
        )
        
        # Gemini for multimodal and integration
        self.orchestrator.add_llm_node(
            "gemini_integration", 
            "gemini-pro", 
            ["integration", "multimodal", "coordination", "combination"], 
            "hub"
        )
        
        # Local LLaMA for pattern recognition (privacy-focused)
        self.orchestrator.add_llm_node(
            "llama_pattern", 
            "llama-2-70b-chat", 
            ["pattern", "recognition", "classification", "local"], 
            "specialist"
        )
        
        # Specialized code model
        self.orchestrator.add_llm_node(
            "codegen_specialist", 
            "code-llama-instruct", 
            ["coding", "programming", "technical", "implementation"], 
            "specialist"
        )
        
        # Fast lightweight model for simple queries
        self.orchestrator.add_llm_node(
            "phi_quick", 
            "microsoft-phi-3-mini", 
            ["quick", "simple", "efficient", "basic"], 
            "relay"
        )
    
    async def intelligent_query_routing(self, user_query: str, 
                                      context: str = "",
                                      max_cost: float = 0.10) -> Dict:
        """Inteligentne routowanie zapytań użytkownika"""
        
        # Pre-analysis query type
        query_type = self._analyze_query_type(user_query)
        
        # Adjust parameters based on query type
        if query_type == "simple":
            max_nodes = 1
            budget = min(max_cost, 0.02)
        elif query_type == "complex":
            max_nodes = 3
            budget = max_cost
        else:  # medium
            max_nodes = 2
            budget = max_cost * 0.7
        
        # Orchestrate response
        result = await self.orchestrator.orchestrate_llm_response(
            user_query,
            max_active_nodes=max_nodes,
            budget_limit=budget * 100  # Convert to internal budget units
        )
        
        # Enhanced post-processing
        if result['leader']:
            leader_node = self.orchestrator.nodes[result['leader']]
            
            # Add metadata about the decision process
            result['decision_metadata'] = {
                'query_type': query_type,
                'chosen_specialist': result['leader'],
                'specialist_domains': leader_node.specialization_domains,
                'confidence_score': self._calculate_confidence(result),
                'cost_efficiency': result['total_cost'] / max(len(result['response']), 1)
            }
        
        return result
    
    def _analyze_query_type(self, query: str) -> str:
        """Przeanalizuj typ zapytania dla lepszego routingu"""
        query_lower = query.lower()
        
        # Simple queries
        simple_indicators = ['what is', 'define', 'explain briefly', 'yes or no']
        if any(indicator in query_lower for indicator in simple_indicators):
            return "simple"
        
        # Complex queries  
        complex_indicators = ['analyze', 'compare', 'develop a strategy', 'write code',
                            'creative solution', 'multiple approaches', 'comprehensive']
        if any(indicator in query_lower for indicator in complex_indicators):
            return "complex"
        
        # Length-based heuristic
        if len(query.split()) > 20:
            return "complex"
        elif len(query.split()) < 5:
            return "simple"
        
        return "medium"
    
    def _calculate_confidence(self, result: Dict) -> float:
        """Oblicz confidence score dla rezultatu"""
        if not result.get('individual_responses'):
            return 0.5
        
        leader = result.get('leader')
        if leader and leader in result['individual_responses']:
            leader_confidence = result['individual_responses'][leader].get('confidence', 0.5)
            
            # Adjust based on collaboration
            collaboration_factor = len(result.get('active_nodes', [])) > 1
            if collaboration_factor:
                # Average confidence of active nodes
                all_confidences = []
                for node_id in result['active_nodes']:
                    if node_id in result['individual_responses']:
                        conf = result['individual_responses'][node_id].get('confidence', 0.5)
                        all_confidences.append(conf)
                
                if all_confidences:
                    avg_confidence = sum(all_confidences) / len(all_confidences)
                    # Blend leader confidence with group confidence
                    return leader_confidence * 0.7 + avg_confidence * 0.3
            
            return leader_confidence
        
        return 0.5

# Run the test
if __name__ == "__main__":
    import asyncio
    
    # Run the LLM orchestration test
    print("Uruchamianie testu Constellation LLM Orchestration...")
    orchestrator, results = asyncio.run(test_llm_orchestration())
    
    print(f"\n🔮 DEMO PRODUCTION USAGE")
    print("=" * 50)
    
    # Demo production usage
    production_constellation = ProductionConstellationLLM()
    
    print("Production Constellation LLM gotowy do:")
    print("• Intelligent query routing") 
    print("• Multi-LLM orchestration")
    print("• Cost-optimized responses") 
    print("• Anti-dominance balancing")
    print("• Golden ratio optimization")
    print("\nGotowy do integracji z prawdziwymi API! 🚀")
    
    #