# 🌟 Constellation LLM Orchestrator

**An innovative multi-model LLM orchestration system using constellation theory and the golden ratio**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

## 🚀 Overview

Constellation LLM Orchestrator is an advanced multi-model LLM orchestration system that automatically selects the best model for each query based on specialization, performance, and cost. The system uses the golden ratio (φ ≈ 1.618) to optimally balance parameters. ### ✨ Key Features

- **🎯 Intelligent Routing** - Automatically selects the best LLM model for each query
- **⚖️ Anti-dominance Balancing** - Prevents monopolization by a single model
- **💰 Cost Optimization** - Real-time budget management for API calls
- **🤝 Model Collaboration** - Ability to activate multiple models for complex tasks
- **📊 Performance Monitoring** - Detailed metrics and dominance analysis
- **🌟 Golden Ratio Optimization** - Mathematically optimal parameter weights

## 🏗️ Architecture

The system is based on constellation theory, where each LLM model is a "node" with its own:
- **Specializations** (logic, creativity, memory, pattern, integration, exploration)
- **Embedding State** (semantic representation)
- **Resonance Phase** (synchronization with the query)
- **Performance history** (trust score)
- **Cost budget** (API limitations)

### Scoring algorithm

Each node receives a score based on the formula:

score = α×semantic_similarity + β×phase_resonance + γ×trust×(1-dominance_penalty)×(1+diversity_bonus) + ζ×domain_relevance - η×cost

Where the parameters are optimized using the golden ratio:
- α = 0.618 (semantic similarity)
- β = 0.618 (phase resonance)
- γ = 0.300 (trust/history)
- η = 0.382 (cost penalty)
- ζ = 1.000 (domain relevance)

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/constellation-llm
cd constellation-llm
pip install -r requirements.txt
```

### Requirements
```
asyncio
dataclasses
typing
math
random
time
json
```

## 🎮 Quick start

```python
import asyncio
from constellationllm import ConstellationLLMOrchestrator

async def main(): 
# Create an orchestra 
orchestrator = ConstellationLLMOrchestrator(dim=12, diversity_factor=0.4) 

# Add LLM models 
orchestrator.add_llm_node( 
"gpt4_reasoning", 
"gpt-4", 
["logic", "analysis", "reasoning"], 
"specialist" 
) 

orchestrator.add_llm_node( 
"claude_creative", 
"claude-3-sonnet", 
["creativity", "imagination", "synthesis"], 
"specialist" 
) 

# Execute the query 
result = await orchestrator.orchestrate_llm_response( 
"Solve this logic puzzle: If A > B and B > C, what can we deduce?", 
max_active_nodes=2, 
budget_limit=50.0 
) 

print(f"Leader: {result['leader']}") 
print(f"Response: {result['response']}") 
print(f"Cost: ${result['total_cost']:.3f}")

# Run
asyncio.run(main())
```

## 📋 Usage Examples

### Basic Orchestration
```python
# Automatic model selection based on specialization
result = await orchestrator.orchestrate_llm_response(
"Design a creative solution for reducing plastic waste",
max_active_nodes=3,
budget_limit=30.0
)
# Selects a model with the "creativity" specialization
```

### Production deployment
```python
from constellationllm import ProductionConstellationLLM

# Production-ready system with real APIs
production = ProductionConstellationLLM()
await production.setup_production_nodes()

result = await production.intelligent_query_routing(
"How can we integrate renewable energy with traditional power grids?",
max_cost=0.10
)
```

## 🔧 Configuration

### Node Types
- **specialist** - Highly specialized in a specific domain
- **hub** - Central hub for coordination
- **explorer** - Exploring new solutions
- **relay** - Fast, efficient responses

### Specialization Domains
- `logic`, `analysis`, `reasoning` - Logical reasoning
- `creativity`, `imagination`, `synthesis` - Creative thinking
- `memory`, `retrieval`, `storage` - Information retrieval
- `pattern`, `recognition`, `classification` - Pattern recognition
- `integration`, `combination`, `coordination` - System integration
- `exploration`, `discovery`, `novelty` - Exploring new solutions

## 📊 Monitoring and Analysis

The system provides detailed metrics:

```python
stats = orchestrator.get_orchestrator_stats()
print(f"Total API calls: {stats['total_api_calls']}")
print(f"Cost efficiency: ${stats['total_cost']:.3f}")

# Dominance analysis
for node_id, node_stats in stats['node_stats'].items(): 
print(f"{node_id}: {node_stats['api_calls']} calls, " 
f"quality: {node_stats['avg_response_quality']:.3f}