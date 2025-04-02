# Awesome-Foundation-Agents

[![PR Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](https://github.com/FoundationAgents/awesome-foundation-agents/pulls)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

We maintains a curated collection of papers exploring the path towards Foundation Agents, with a focus on formulating the core concepts and navigating the research landscape.

## Our Works Towards Foundation Agents

✨✨✨ [Advances and Challenges in Foundation Agents]() (Paper)

<div style="display: flex; justify-content: space-between;">
    <img src="assets/1-brain.png" alt="The key of human brain." width="48%">
    <img src="assets/1-agent_framework.png" alt="The Framework of Foundation Agent" width="48%">
</div>

# Awesome Papers

<font size=5><center><b> Table of Contents </b> </center></font>
- [Core Components of Intelligent Agents](#core-components-of-intelligent-agents)
    - [Cognition](#cognition)
    - [Memory](#memory)
    - [Perception](#perception)
    - [World Model](#world-model)
    - [Action](#action)
    - [Reward](#reward)
    - [Emotion](#emotion)
- [Self-Enhancement in Intelligent Agents](#self-enhancement-in-intelligent-agents)
- [Collaborative and Evolutionary Intelligent Systems](#collaborative-and-evolutionary-intelligent-systems)
- [Building Safe and Beneficial AI](#building-safe-and-beneficial-ai)


# Core Components of Intelligent Agents

## Cognition

<div style="display: flex; justify-content: space-between;">
    <img src="assets/2-1-cognition.png" alt="Cognition System" width="100%">
</div>

### Learning
#### Space
##### Full
- **Add SFT,RLHF,PEFT**
- **ReFT: Reasoning with Reinforced Fine-Tuning**, arxiv 2024, [[paper]()] [[code]()]
- **R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning**, arxiv 2025, [[paper]()] [[code]()]


##### Partial
- **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**, Wei et al. 2022, [[paper](https://arxiv.org/abs/2201.11903)] [[code]()]
- **Voyager: An Open-Ended Embodied Agent with Large Language Models**, arxiv 2023, [[paper](https://arxiv.org/abs/2305.16291)] [[code]()]
- **Reflexion: Language Agents with Verbal Reinforcement Learning**, NeurIPS 2023, [[paper](https://arxiv.org/abs/2303.11366)] [[code]()]
- **ReAct meets ActRe: Autonomous Annotations of Agent Trajectories for Contrastive Self-Training**, arxiv 2024, [[paper](https://arxiv.org/abs/2403.14589)] [[code]()]
- **Generative Agents: Interactive Simulacra of Human Behavior**, ACM UIST 2023, [[paper](https://arxiv.org/abs/2304.03442)] [[code]()]

#### Objective
##### Perception
- **CLIP: Learning Transferable Visual Models from Natural Language Supervision**, ICML 2021, [[paper](https://arxiv.org/abs/2103.00020)] [[code]()]
- **LLaVA: Visual Instruction Tuning**, NeurIPS 2023, [[paper](https://arxiv.org/abs/2304.08485)] [[code]()]
- **CogVLM: Visual Expert for Pretrained Language Models**, NeurIPS 2025, [[paper](https://arxiv.org/abs/2311.03079)] [[code]()]
- **Qwen2-Audio Technical Report**, arxiv 2024, [[paper]()] [[code]()]
- **Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning**, arxiv 2025, [[paper]()] [[code]()]


##### Reasoning
- **SKY-T1: Train Your Own o1 Preview Model Within $450**, 2025, [[paper]()] [[code]()]
- **Open Thoughts**, 2025, [[paper]()] [[code]()]
- **LIMO: Less is More for Reasoning**, arxiv 2025, [[paper]()] [[code]()]
- **STaR: Bootstrapping Reasoning with Reasoning**, arxiv 2022, [[paper]()] [[code]()]
- **ReST: Reinforced Self-Training for Language Modeling**, arxiv 2023, [[paper](https://arxiv.org/abs/2308.08998)] [[code]()]
- **OpenR: An Open Source Framework for Advanced Reasoning with Large Language Models**, arxiv 2024, [[paper]()] [[code]()]
- **LLaMA-Berry: Pairwise Optimization for o1-like Olympiad-level Mathematical Reasoning**, arxiv 2024, [[paper]()] [[code]()]
- **RAGEN: Training Agents by Reinforcing Reasoning**, arxiv 2025, [[paper]()] [[code]()]
- **Open-R1**, 2024, [[paper]()] [[code]()]

##### World
- **Inner Monologue: Embodied Reasoning through Planning with Language Models**, CoRL 2023, [[paper](https://arxiv.org/abs/2207.05608)] [[code]()]
- **Self-Refine: Iterative Refinement with Self-Feedback**, NeurIPS 2024, [[paper](https://arxiv.org/abs/2303.17651)] [[code]()]
- **Reflexion: Language Agents with Verbal Reinforcement Learning**, NeurIPS 2023, [[paper](https://arxiv.org/abs/2303.11366)] [[code]()]
- **ExpeL: LLM Agents Are Experiential Learners**, AAAI 2024, [[paper](https://arxiv.org/abs/2308.10144)] [[code]()]
- **AutoManual: Generating Instruction Manuals by LLM Agents via Interactive Environmental Learning**, arxiv 2024, [[paper](https://arxiv.org/abs/2405.16247)] [[code]()]
- **ReAct meets ActRe: Autonomous Annotations of Agent Trajectories for Contrastive Self-Training**, arxiv 2024, [[paper](https://arxiv.org/abs/2403.14589)] [[code]()]

### Reasoning
#### Structured
##### Dynamic
- **ReAct: Synergizing Reasoning and Acting in Language Models**, arxiv 2022, [[paper](https://arxiv.org/abs/2210.03629)] [[code]()]
- **Markov Chain of Thought for Efficient Mathematical Reasoning**, arxiv 2024, [[paper]()] [[code]()]
- **Tree of Thoughts: Deliberate Problem Solving with Large Language Models**, NeurIPS 2023, [[paper](https://arxiv.org/abs/2305.10601)] [[code]()]
- **Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models**, ICML 2024, [[paper](https://arxiv.org/abs/2310.04406)] [[code]()]
- **Reasoning via Planning (RAP): Improving Language Models with World Models**, EMNLP 2023, [[paper](https://arxiv.org/abs/2305.14992)] [[code]()]
- **Graph of Thoughts: Solving Elaborate Problems with Large Language Models**, AAAI 2023, [[paper](https://arxiv.org/abs/2308.09687)] [[code]()]
- **Path of Thoughts: Extracting and Following Paths for Robust Relational Reasoning with Large Language Models**, arxiv 2024, [[paper]()] [[code]()]
- **On the Diagram of Thought**, arxiv 2024, [[paper]()] [[code]()]

##### Static
- **Self-Consistency Improves Chain of Thought Reasoning in Language Models**, ICLR 2023, [[paper](https://arxiv.org/abs/2203.11171)] [[code]()]
- **Self-Refine: Iterative Refinement with Self-Feedback**, NeurIPS 2024, [[paper](https://arxiv.org/abs/2303.17651)] [[code]()]
- **Progressive-Hint Prompting Improves Reasoning in Large Language Models**, arxiv 2023, [[paper](https://arxiv.org/abs/2304.09797)] [[code]()]
- **On the Self-Verification Limitations of Large Language Models on Reasoning and Planning Tasks**, arxiv 2024, [[paper]()] [[code]()]
- **Chain-of-Verification Reduces Hallucination in Large Language Models**, ICLR 2024 Workshop, [[paper](https://arxiv.org/abs/2309.11495)] [[code]()]


##### Domain
- **MathPrompter: Mathematical Reasoning Using Large Language Models**, ACL 2023, [[paper](https://arxiv.org/abs/2303.05398)] [[code]()]
- **LLMs Can Find Mathematical Reasoning Mistakes by Pedagogical Chain-of-Thought**, arxiv 2024, [[paper]()] [[code]()]
- **Physics Reasoner: Knowledge-Augmented Reasoning for Solving Physics Problems with Large Language Models**, COLING 2025, [[paper]()] [[code]()]


#### Unstructured
##### Prompt
- **Chain of Thought Prompting Elicits Reasoning in Large Language Models**, NeurIPS 2022, [[paper](https://arxiv.org/abs/2201.11903)] [[code]()]
- **Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models**, ICLR 2024, [[paper](https://arxiv.org/abs/2310.06117)] [[code]()]
- **Ask Me Anything: A Simple Strategy for Prompting Language Models**, arxiv 2022, [[paper](https://arxiv.org/abs/2210.02441)] [[code]()]
- **Chain-of-Knowledge: Grounding Large Language Models via Dynamic Knowledge Adapting over Heterogeneous Sources**, arxiv 2023, [[paper](https://arxiv.org/abs/2305.13269)] [[code]()]
- **Self-Explained Keywords Empower Large Language Models for Code Generation**, arxiv 2024, [[paper]()] [[code]()]


##### Model
- **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**, arxiv 2025, [[paper]()] [[code]()]
- **Claude 3.7 Sonnet**, 2025, [[paper]()] [[code]()]
- **OpenAI o1 System Card**, arxiv 2024, [[paper]()] [[code]()]

##### Implicit
- **Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking**, arxiv 2024, [[paper](https://arxiv.org/abs/2403.09629)] [[code]()]
- **Chain of Continuous Thought (Coconut): Training Large Language Models to Reason in a Continuous Latent Space**, arxiv 2024, [[paper]()] [[code]()]


#### Planning
- **Describe, Explain, Plan and Select (DEPS): Interactive Planning with Large Language Models**, arxiv 2023, [[paper](https://arxiv.org/abs/2302.01560)] [[code]()]
- **ProgPrompt: Generating Situated Robot Task Plans Using Large Language Models**, ICRA 2023, [[paper](https://arxiv.org/abs/2209.11302)] [[code]()]
- **ADAPT: As-Needed Decomposition and Planning with Language Models**, arxiv 2023, [[paper]()] [[code]()]
- **Tree of Thoughts: Deliberate Problem Solving with Large Language Models**, NeurIPS 2023, [[paper](https://arxiv.org/abs/2305.10601)] [[code]()]
- **Reasoning via Planning (RAP): Improving Language Models with World Models**, EMNLP 2023, [[paper](https://arxiv.org/abs/2305.14992)] [[code]()]
- **TravelPlanner: A Benchmark for Real-World Planning with Language Agents**, ICML 2024, [[paper](https://arxiv.org/abs/2402.01622)] [[code]()]
- **PDDL—The Planning Domain Definition Language**, 1998, [[paper](https://arxiv.org/abs/1106.4561)] [[code]()]
- **Mind2Web: Towards a Generalist Agent for the Web**, NeurIPS 2023, [[paper](https://arxiv.org/abs/2306.06070)] [[code]()]

## Memory
<div style="display: flex; justify-content: space-between;">
    <img src="assets/2-2-memory.png" alt="Memory in Intelligence Agents" width="100%">
</div>

### 1. Representation

#### 1.1 Sensory
##### 1.1.1 Text-based
- RecAgent (Wang et al., 2023)
- CoPS (Zhou et al., 2024)
- MemoryBank (Zhong et al., 2024)
- Memory Sandbox (Huang et al., 2023)

##### 1.1.2 Multi-modal
- VideoAgent (Fan et al., 2024)
- WorldGPT (Ge et al., 2024)
- Agent S (Agashe et al., 2024)
- OS-Copilot (Wu et al., 2024)
- MuLan (Li et al., 2024)

#### 1.2 Short-term
##### 1.2.1 Context
- MemGPT (Packer et al., 2023)
- KARMA (Wang et al., 2024)
- LSFS (Shi et al., 2024)
- OSCAR (Wang et al., 2024)
- RCI (Geunwoo et al., 2023)

##### 1.2.2 Working
- Generative Agent (Park et al., 2023)
- RLP (Fischer et al., 2023)
- CALYPSO (Zhu et al., 2023)
- HiAgent (Hu et al., 2024)

#### 1.3 Long-term
##### 1.3.1 Semantic
- AriGraph (Anokhin et al., 2024)
- RecAgent (Wang et al., 2023)
- HippoRAG (Gutierrez et al., 2024)

##### 1.3.2 Episodic
- MobileGPT (Lee et al., 2023)
- MemoryBank (Zhong et al., 2024)
- Episodic Verbalization (Barmann et al., 2024)
- MrSteve (Park et al., 2024)

##### 1.3.3 Procedural
- AAG (Roth et al., 2024)
- Cradle (Tan et al., 2024)
- JARVIS-1 (Wang et al., 2024)
- LARP (Yan et al., 2023)

### 2. Lifecycle

#### 2.1 Acquisition
##### 2.1.1 Information Compression
- HiAgent (Hu et al., 2024)
- LMAgent (Liu et al., 2024)
- ReadAgent (Lee et al., 2024)
- M²WF (Wang et al., 2025)

##### 2.1.2 Experience Consolidation
- ExpeL (Zhao et al., 2024)
- MindOS (Hu et al., 2025)
- Vanschoren et al. (2018)
- Hou et al. (2024)

#### 2.2 Encoding
##### 2.2.1 Selective Attention
- AgentCorrd (Pan et al., 2024)
- MS (Gao et al., 2024)
- GraphVideoAgent (Chu et al., 2025)
- A-MEM (Xu et al., 2025)
- Ali et al. (2024)

##### 2.2.2 Multi-modal Fusion
- Optimus-1 (Li et al., 2024)
- Optimus-2 (Li et al., 2025)
- JARVIS-1 (Wang et al., 2024)

#### 2.3 Derivation
##### 2.3.1 Reflection
- Agent S (Agashe et al., 2024)
- OSCAR (Wang et al., 2024)
- R2D2 (Huang et al., 2025)
- Mobile-Agent-E (Wang et al., 2025)

##### 2.3.2 Summarization
- SummEdits (Laban et al., 2023)
- SCM (Wang et al., 2023)
- Healthcare Copilot (Ren et al., 2024)
- Wang et al. (2023)

##### 2.3.3 Knowledge Distillation
- Knowagent (Zhu et al., 2024)
- AoTD (Shi et al., 2024)
- LDPD (Liu et al., 2024)
- Sub-goal Distillation (Hashemzadeh et al., 2024)
- MAGDi (Chen et al., 2024)

##### 2.3.4 Selective Forgetting
- Lyfe Agent (Kaiya et al., 2023)
- TiM (Liu et al., 2023)
- MemoryBank (Zhong et al., 2024)
- S³ (Gao et al., 2023)
- Hou et al. (2024)

#### 2.4 Retrieval
##### 2.4.1 Indexing
- HippoRAG (Gutierrez et al., 2024)
- TradingGPT (Li et al., 2023)
- LongMemEval (Wu et al., 2024)
- SeCom (Pan et al., 2025)

##### 2.4.2 Matching
- Product Keys (Lample et al., 2019)
- OSAgent (Xu et al., 2024)
- Bahdanau et al. (2014)
- Hou et al. (2024)

#### 2.5 Neural Memory
##### 2.5.1 Associative Memory
- Hopfield Networks (Demircigil et al., 2017; Ramsauer et al., 2020)
- Neural Turing Machines (Falcon et al., 2022)

##### 2.5.2 Parameter Integration
- MemoryLLM (Wang et al., 2024)
- SELF-PARAM (Wang et al., 2024)
- MemoRAG (Qian et al., 2024)
- TTT-Layer (Sun et al., 2024)
- Titans (Behrouz et al., 2024)
- R³Mem (Wang et al., 2025)

#### 2.6 Utilization
##### 2.6.1 RAG
- RAGLAB (Zhang et al., 2024)
- Adaptive Retrieval (Mallen et al., 2023)
- Atlas (Farahani et al., 2024)
- Yuan et al. (2025)

##### 2.6.2 Long-context Modeling
- RMT (Bulatov et al., 2022, 2023)
- AutoCompressor (Chevalier et al., 2023)
- ICAE (Ge et al., 2023)
- Gist (Mu et al., 2024)
- CompAct (Yoon et al., 2024)

##### 2.6.3 Alleviating Hallucination
- Lamini (Li et al., 2024)
- Memoria (Park et al., 2023)
- PEER (He et al., 2024)
- Ding et al. (2024)


## Perception
<div style="display: flex; justify-content: space-between;">
    <img src="assets/2-3-perception.png" alt="Perception System" width="100%">
</div>

### Unimodal Models

#### Text
- BERT (Devlin et al., 2018)
- RoBERTa (Liu et al., 2019)
- ALBERT (Lan et al., 2019)

#### Image
- ResNet (He et al., 2016)
- DETR (Carion et al., 2020)
- Grounding DINO 1.5 (Ren et al., 2024)

#### Video
- ViViT (Arnab et al., 2021)
- VideoMAE (Tong et al., 2022)

#### Audio
- FastSpeech 2 (Ren et al., 2020)
- Seamless (Barrault et al., 2023)
- wav2vec 2.0 (Baevski et al., 2020)

#### Other Unimodal
- Visual ChatGPT (Wu et al., 2023)
- HuggingGPT (Shen et al., 2024)
- MM-REACT (Yang et al., 2023)
- ViperGPT (Suris et al., 2023)
- AudioGPT (Huang et al., 2024)
- LLaVA-Plus (Liu et al., 2025)

### Cross-modal Models

#### Text-Image
- CLIP (Alec et al., 2021)
- ALIGN (Jia et al., 2021)
- DALL·E 3 (Betker et al., 2023)
- VisualBERT (Li et al., 2019)

#### Text-Video
- VideoCLIP (Xu et al., 2021)
- Phenaki (Villegas et al., 2022)
- Make-A-Video (Singer et al., 2022)

#### Text-Audio
- Wav2CLIP (Wu et al., 2022)
- VATT (Akbari et al., 2021)
- AudioCLIP (Guzhov et al., 2022)

#### Other Cross-modal
- CLIP-Forge (Sanghi et al., 2022)
- Point-E (Nichol et al., 2022)

### MultiModal Models

#### VLM (Vision-Language Models)
- MiniGPT-v2 (Chen et al., 2023)
- LLaVA-NeXT (Liu et al., 2024)
- CogVLM2 (Hong et al., 2024)
- Qwen2-VL (Wang et al., 2024)
- Emu2 (Sun et al., 2024)

##### Edge-Side VLM
- TinyGPT-V (Yuan et al., 2023)
- MobileVLM (Chu et al., 2023)
- MiniCPM-V (Yao et al., 2024)
- OmniParser (Lu et al., 2024)

#### VLA (Vision-Language for Action)
- CLIPort (Shridhar et al., 2022)
- RT-1 (Brohan et al., 2022)
- MOO (Stone et al., 2023)
- PerAct (Shridhar et al., 2023)
- Diffusion Policy (Chi et al., 2023)
- PaLM-E (Driess et al., 2023)
- MultiPLY (Hong et al., 2024)

#### ALM (Audio-Language Models)
- Audio Flamingo (Kong et al., 2024)
- SpeechVerse (Das et al., 2024)
- UniAudio 1.5 (Yang et al., 2024)
- Qwen2-Audio (Chu et al., 2024)
- Audio-LLM (Li et al., 2024)
- Mini-Omni (Xie et al., 2024)
- SpeechGPT (Zhang et al., 2023)

#### AVLM (Audio-Visual-Language Models)
- ONE-PEACE (Wang et al., 2023)
- PandaGPT (Su et al., 2023)
- Macaw-LLM (Lyu et al., 2023)
- LanguageBind (Zhu et al., 2023)
- UnIVAL (Shukor et al., 2023)
- X-LLM (Chen et al., 2023)

#### Other MultiModal
- PointLLM (Xu et al., 2025)
- MiniGPT-3D (Tang et al., 2024)
- NExT-GPT (Wu et al., 2023)
- Unified-IO 2 (Lu et al., 2024)
- CoDi-2 (Tang et al., 2024)
- ModaVerse (Wang et al., 2024)

## World Model

<div style="display: flex; justify-content: space-between;">
    <img src="assets/2-4-world_model.png" alt="World Model in Foundation Agents" width="100%">
</div>

### External Approaches
**DINO-WM [358]: Video World Models on Pre-trained Visual Features Enable Zero-Shot Planning**, arxiv 2024, [[paper](https://arxiv.org/abs/2411.04983)], [[code][]]

**SAPIEN [351]: A Simulated Part-based Interactive Environment**, CVPR 2020, [[paper](https://arxiv.org/abs/2003.08515)], [[code][]]

**MuZero [349]: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model**, Nature 2020, [[paper](https://www.nature.com/articles/s41586-020-03051-4)], [[code][]]

**GR-2 [357]: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation**, arxiv 2024, [[paper](https://arxiv.org/abs/2410.06158)], [[code][]]

**COAT [356]: Discovery of the Hidden World with Large Language Models**, arxiv 2024, [[paper](https://arxiv.org/abs/2402.03941)], [[code][]]

**AutoManual [108]: Generating Instruction Manuals by LLM Agents via Interactive Environmental Learning**, arxiv 2024, [[paper](https://arxiv.org/abs/2405.16247)], [[code][]]

**PILCO [355]: A Model-Based and Data-Efficient Approach to Policy Search**, ICML 2011, [[paper]()], [[code][]]

### Internal Approaches
**ActRe [49]: ReAct meets ActRe: Autonomous Annotations of Agent Trajectories for Contrastive Self-Training**, arxiv 2024, [[paper](https://arxiv.org/abs/2403.14589)], [[code][]]

**World Models [348]: World Models**, NeurIPS 2018, [[paper](https://arxiv.org/abs/1803.10122)], [[code][]]

**Dreamer [350]: Dream to Control: Learning Behaviors by Latent Imagination**, ICLR 2020, [[paper](https://arxiv.org/abs/1912.01603)], [[code][]]

**Diffusion WM [353]: Diffusion for World Modeling: Visual Details Matter in Atari**, arxiv 2024, [[paper](https://arxiv.org/abs/2405.12399)], [[code][]]

**GQN [354]: Neural Scene Representation and Rendering**, Science 2018, [[paper]()], [[code][]]

**Daydreamer [352]: World Models for Physical Robot Learning**, CoRL 2023, [[paper]()], [[code][]]

## Action
<div style="display: flex; justify-content: space-between;">
    <img src="assets/2-5-action.jpg" alt="The action." width="100%">
</div>

### Action Space:

### Language

#### Text

- **ReAct: Synergizing Reasoning and Acting in Language Models**, ICLR 2023, [[paper](https://arxiv.org/abs/2210.03629)] [[code](https://github.com/ysymyth/ReAct)]

- **AutoGPT: Build, Deploy, and Run AI Agents**, Github, [[code](https://github.com/Significant-Gravitas/AutoGPT)]

- **Reflexion: Language Agents with Verbal Reinforcement Learning**, NeurIPS 2023, [[paper](https://arxiv.org/abs/2303.11366)] [[code](https://github.com/noahshinn/reflexion)]

- **LLM+P: Empowering Large Language Models with Optimal Planning Proficiency**, arXiv 2023, [[paper](https://arxiv.org/abs/2304.11477)] [[code](https://github.com/Cranial-XIX/llm-pddl)]

#### Code

- **MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework**, ICLR 2023, [[paper](https://arxiv.org/abs/2308.00352)] [[code](https://github.com/geekan/MetaGPT)]

- **ChatDev: Communicative Agents for Software Development**, ACL 2024, [[paper](https://arxiv.org/abs/2307.07924)] [[code](https://github.com/OpenBMB/ChatDev)]

- **SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering**, NeurIPS 2025, [[paper](https://arxiv.org/abs/2405.15793)] [[code](https://github.com/SWE-agent/SWE-agent)]

- **OpenHands: An Open Platform for AI Software Developers as Generalist Agents**, arXiv 2024, [[paper](https://arxiv.org/abs/2407.16741)] [[code](https://github.com/All-Hands-AI/OpenHands)]
- 
#### Chat

- **Generative Agents: Interactive Simulacra of Human Behavior**, UIST 2023, [[paper](https://arxiv.org/abs/2304.03442)] [[code](https://github.com/joonspk-research/generative_agents)]

- **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation**, COLM 2024, [[paper](https://arxiv.org/abs/2308.08155)] [[code](https://github.com/microsoft/autogen)]

### Digital

#### Game

- **MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge**, NeurIPS 2022, [[paper](https://arxiv.org/abs/2206.08853)] [[code](https://github.com/MineDojo/MineDojo)]
  
- **Voyager: An Open-Ended Embodied Agent with Large Language Models**, TMLR 2024, [[paper](https://arxiv.org/abs/2305.16291)] [[code](https://github.com/MineDojo/Voyager)]

- **SwarmBrain: Embodied agent for real-time strategy game StarCraft II via large language models**, arXiv 2024, [[paper](https://arxiv.org/abs/2401.17749)] [[code](https://github.com/ramsayxiaoshao/SwarmBrain)]

- **JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models**, NeurIPS 2025, [[paper](https://arxiv.org/abs/2311.05997)] [[code](https://github.com/CraftJarvis/JARVIS-1)]

#### Multimodal

- **MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action**, arXiv 2023, [[paper](https://arxiv.org/abs/2303.11381)] [[code](https://github.com/microsoft/MM-REACT)]

- **ViperGPT: Visual Inference via Python Execution for Reasoning**, ICCV 2023, [[paper](https://arxiv.org/abs/2303.08128)] [[code](https://github.com/cvlab-columbia/viper)]

- **Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models**, arXiv 2023, [[paper](https://arxiv.org/abs/2303.04671)] [[code](https://github.com/hackiey/visual-chatgpt)]

- **HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face**, NeurIPS 2023, [[paper](https://arxiv.org/pdf/2303.17580)] [[code](https://github.com/AI-Chef/HuggingGPT)]

#### Web

- **WebGPT: Browser-assisted question-answering with human feedback**, arXiv 2021, [[paper](https://arxiv.org/abs/2112.09332)] [[blog](https://openai.com/index/webgpt/)]

- **WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents**, NeurIPS 2022, [[paper](https://arxiv.org/abs/2207.01206)] [[code](https://github.com/princeton-nlp/WebShop)]

- **A Real-World WebAgent with Planning, Long Context Understanding, and Program Synthesis**, ICLR 2024, [[paper](https://arxiv.org/abs/2307.12856)]

- **Mind2Web: Towards a Generalist Agent for the Web**, NeurIPS 2025, [[paper](https://arxiv.org/abs/2306.06070)] [[code](https://github.com/OSU-NLP-Group/Mind2Web)]

#### GUI

- **Mobile-Agent: Autonomous Multi-Modal Mobile Device Agent with Visual Perception**, arXiv 2024, [[paper](https://arxiv.org/abs/2401.16158)] [[code](https://github.com/X-PLUG/MobileAgent)]

- **AppAgent: Multimodal Agents as Smartphone Users**, arXiv 2023, [[paper](https://arxiv.org/abs/2312.13771)] [[code](https://github.com/TencentQQGYLab/AppAgent)]

- **UFO: A UI-Focused Agent for Windows OS Interaction**, arXiv 2024, [[paper](https://arxiv.org/abs/2402.07939)] [[code](https://github.com/microsoft/UFO)]

- **OmniParser for Pure Vision Based GUI Agent**, arXiv 2024, [[paper](https://arxiv.org/abs/2408.00203)] [[code](https://github.com/microsoft/OmniParser)]

#### DB & KG

- **UnifiedSKG: Unifying and Multi-Tasking Structured Knowledge Grounding with Text-to-Text Language Models**, EMNLP 2022, [[paper](https://arxiv.org/abs/2201.05966)] [[code](https://github.com/xlang-ai/UnifiedSKG)]

- **Don't Generate, Discriminate: A Proposal for Grounding Language Models to Real-World Environments**, ACL 2023, [[paper](https://arxiv.org/abs/2212.09736)] [[code](https://github.com/dki-lab/Pangu)]

- **Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs**, NeurIPS 2025, [[paper](https://arxiv.org/pdf/2305.03111)] [[project](https://bird-bench.github.io/)]

- **Spider 2.0: Evaluating language models on real-world enterprise text-to-sql workflows.**, ICLR 2025, [[paper](https://arxiv.org/abs/2411.07763)] [[code](https://github.com/xlang-ai/Spider2)]

- **Middleware for llms: Tools are instrumental for language agents in complex environments.**, EMNLP 2024, [[paper](https://arxiv.org/abs/2402.14672)] [[code](https://github.com/OSU-NLP-Group/Middleware)]

### Physical

- **RT-1: Robotics Transformer for Real-World Control at Scale**, RSS 2023, [[paper](https://arxiv.org/abs/2212.06817)] [[project](https://robotics-transformer1.github.io/)]

- **RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control**, CoRL 2023, [[paper](https://arxiv.org/abs/2307.15818)] [[project](https://robotics-transformer2.github.io/)]

- **Open X-Embodiment: Robotic Learning Datasets and RT-X Models**, arXiv 2023, [[paper](https://arxiv.org/abs/2310.08864v4)] [[project](https://robotics-transformer-x.github.io/)]
  
- **GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation**, arXiv 2024, [[paper](https://arxiv.org/abs/2410.06158)] [[project](https://gr2-manipulation.github.io/)]
  
- **π0: A vision-language-action flow model for general robot control.**, arXiv 2024, [[paper](https://arxiv.org/abs/2410.24164)]

- **Do as I can, not as I say Grounding language in robotic affordances**, CoRL 2022, [[paper](https://arxiv.org/abs/2204.01691)] [[project](https://say-can.github.io/)]

- **Voxposer: Composable 3d value maps for robotic manipulation with language models.**, CoRL 2023, [[paper](https://arxiv.org/abs/2307.05973)] [[code](https://github.com/huangwl18/VoxPoser)]

- **Embodiedgpt: Vision-language pre-training via embodied chain of thought.**, NeurIPS 2023, [[paper](https://arxiv.org/abs/2305.15021)] [[project](https://embodiedgpt.github.io/)]

### Learning

### ICL (In-Context Learning)

#### Prompt

- **CoT**, Details to be added

- **ReAct**, arXiv, 2022 [[paper](https://arxiv.org/abs/2210.03629)]

- **Auto-CoT**, Details to be added

- **ToT**, Details to be added

- **GoT**, Details to be added

- **CoA**, Details to be added

#### Decompose

- **Least-to-Most**, Details to be added

- **HuggingGPT**, Details to be added

- **Plan-and-Solve**, Details to be added

- **ProgPrompt**, Details to be added

#### Role-play

- **Generative Agents**, Details to be added

- **MetaGPT**, Details to be added

- **ChatDev**, Details to be added

- **SWE-Agent**, Details to be added

#### Refine

- **Reflexion**, arXiv, 2023 [[paper](https://arxiv.org/abs/2303.11366)]

- **Self-refine**, Details to be added

- **GPTSwarm**, Details to be added

### PT & SFT (Pre-Training & Supervised Fine-Tuning)

#### Pre-Train

- **RT-1**, Details to be added

- **RT-2**, Details to be added

- **RT-X**, Details to be added

- **GR-2**, Details to be added

- **LAM**, Details to be added

#### SFT

- **LearnAct**, Details to be added

- **CogACT**, Details to be added

- **RT-H**, Details to be added

- **OpenVLA**, Details to be added

- **GR-2**, Details to be added

- **πo**, Details to be added

- **UniAct**, Details to be added

### RL (Reinforcement Learning)

- **RLHF**, Details to be added

- **DPO**, Details to be added

- **RLFP**, Details to be added

- **ELLM**, Details to be added

- **GenSim**, Details to be added

- **LEA**, Details to be added

- **MLAQ**, Details to be added

- **KALM**, Details to be added

- **When2Ask**, Details to be added

- **Eureka**, Details to be added

- **ArCHer**, Details to be added

- **LLaRP**, Details to be added

- **GPTSwarm**, Details to be added

## Reward
<div style="display: flex; justify-content: space-between;">
    <img src="assets/2-6-reward.png" alt="Reward System" width="100%">
</div>

### Extrinsic Reward
#### Dense Reward
- InstructGPT (Ouyang et al., 2022)
- DRO (Richemond et al., 2024)
- sDPO (Kim et al., 2024)
- ΨPO (Azar et al., 2024)
- β-DPO (Wu et al., 2025)
- ORPO (Hong et al., 2024)
- DNO (Rosset et al., 2024)
- f-DPO (Wang et al., 2023)
- Xu et al., 2023
- Rafailov et al., 2024

#### Sparse Reward
- PAFT (Pentyala et al., 2024)
- SimPO (Meng et al., 2025)
- LiPO (Liu et al., 2024)
- RRHF (Yuan et al., 2023)
- PRO (Song et al., 2024)
- D²O (Duan et al., 2024)
- NPO (Zhang et al., 2024)
- Ahmadian et al., 2024

#### Delayed Reward
- CPO (Xu et al., 2024)
- NLHF (Munos et al., 2023)
- Swamy et al., 2024

#### Adaptive Reward
- InstructGPT (Ouyang et al., 2022)
- DRO (Richemond et al., 2024)
- β-DPO (Wu et al., 2025)
- ORPO (Hong et al., 2024)
- PAFT (Pentyala et al., 2024)
- SimPO (Meng et al., 2025)
- NLHF (Munos et al., 2023)
- Swamy et al., 2024
- f-DPO (Wang et al., 2023)

### Intrinsic Reward
#### Curiosity-Driven Reward
- Pathak et al., 2017
- Pathak et al., 2019
- Plan2Explore (Sekar et al., 2020)

#### Diversity Reward
- LIIR (Du et al., 2019)

#### Competence-Based Reward
- CURIOUS (Colas et al., 2019)
- Skew-Fit (Pong et al., 2019)
- DISCERN (Hassani et al., 2021)
- Yuan et al., 2024
- KTO (Ethayarajh et al., 2024)

#### Exploration Reward
- Yuan et al., 2024
- Burda et al., 2018

#### Information Gain Reward
- Ton et al., 2024
- VIME (Houthooft et al., 2016)
- EMI (Kim et al., 2018)
- MAX (Shyam et al., 2019)
- KTO (Ethayarajh et al., 2024)

### Hybrid Reward
#### Combination of Intrinsic and Extrinsic Reward
- d-RLAIF (Lee et al., 2023)
- Bai et al., 2022
- Xiong et al., 2023
- Dong et al., 2024

### Hierarchical Reward
#### Hierarchical Reward
- TDPO (Zeng et al., 2024)

## Emotion

# Self-Enhancement in Intelligent Agents
<div style="display: flex; justify-content: space-between;">
    <img src="assets/3-self_evo.png" alt="Self-evolution" width="100%">
</div>

# Collaborative and Evolutionary Intelligent Systems
<div style="display: flex; justify-content: space-between;">
    <img src="assets/4-llm_mas.png" alt="LLM-based Multi-Agent Systems" width="100%">
</div>

## Application
### Strategic Learning
- RECONCILE (Chen et al., 2023)
- LLM-Game-Agent (Lan et al., 2023)
- BattleAgentBench (Wang et al., 2024)

### Modeling and Simulation
- Generative Agents (Park et al., 2023)
- Agent Hospital (Li et al., 2024)
- MedAgents (Tang et al., 2024)
- MEDCO (Wei et al., 2024)

### Collaborative Task Solving
- MetaGPT (Hong et al., 2023)
- ChatDev (Qian et al., 2024)
- Agent Laboratory (Schmidgall et al., 2025)
- The Virtual Lab (Swanson et al., 2024)

## Composition and Protocol
### Agent Composition
#### Homogeneous
- CoELA (Zhang et al., 2023)
- VillagerAgent (Dong et al., 2024)
- LLM-Coordination (Agashe et al., 2024)

#### Heterogeneous
- MetaGPT (Hong et al., 2023)
- ChatDev (Qian et al., 2024)
- Generative Agents (Park et al., 2023)
- S-Agents (Chen et al., 2024)

### Interaction Protocols
#### Message Types
- SciAgents (Ghafarollahi et al., 2024)
- AppAgent (Chi et al., 2023)
- MetaGPT (Hong et al., 2023)

#### Communication Interfaces
- AgentBench (Liu et al., 2023)
- VAB (Liu et al., 2024)
- TaskWeaver (Qiao et al., 2024)
- HULA (Takerngsaksiri et al., 2025)

### Next Generation Protocol
- MCP (Anthropic)
- Agora (Marro et al., 2024)
- IoA (Chen et al., 2024)

## Topology
### Static Topology
- MEDCO (Wei et al., 2024)
- Agent Hospital (Li et al., 2024)
- Welfare Diplomacy (Mukobi et al., 2023)
- MedAgents (Tang et al., 2024)

### Dynamic Topology
- DyLAN (Liu et al., 2023)
- GPTSwarm (Zhuge et al., 2024)
- CodeR (Chen et al., 2024)
- Oasis (Yang et al., 2024)

## Collaboration
### Agent-Agent Collaboration
#### Consensus-oriented
- Agent Laboratory (Schmidgall et al., 2025)
- The Virtual Lab (Swanson et al., 2024)
- OASIS (Yang et al., 2024)

#### Collaborative Learning
- Generative Agents (Park et al., 2023)
- Welfare Diplomacy (Mukobi et al., 2023)
- LLM-Game-Agent (Lan et al., 2023)
- BattleAgentBench (Wang et al., 2024)

#### Teaching/Mentoring
- MEDCO (Wei et al., 2024)
- Agent Hospital (Li et al., 2024)

#### Task-oriented
- MedAgents (Tang et al., 2024)
- S-Agents (Chen et al., 2024)

### Human-AI Collaboration
- Dittos (Leong et al., 2024)
- PRELUDE (Gao et al., 2024)

## Evolution
### Collective Intelligence
- Generative Agents (Park et al., 2023)
- Welfare Diplomacy (Mukobi et al., 2023)
- LLM-Game-Agent (Lan et al., 2023)
- BattleAgentBench (Wang et al., 2024)

### Individual Adaptability
- Agent Hospital (Li et al., 2024)
- Agent Laboratory (Schmidgall et al., 2025)
- MEDCO (Wei et al., 2024)

## Evaluation
### Benchmark for Specific Tasks
- MBPP (dataset-mbpp)
- HotpotQA (dataset-hotpot-qa)
- MATH (dataset-math)
- SVAMP (dataset-svamp)
- MultiArith (dataset-multiarith)

### Benchmark for MAS
- Collab-Overcooked (Sun et al., 2025)
- REALM-Bench (Geng et al., 2025)
- PARTNR (Chang et al., 2024)
- VillagerBench (Dong et al., 2024)
- AutoArena (Zhao et al., 2024)
- MultiagentBench (Zhu et al., 2025)


# Building Safe and Beneficial AI
<div style="display: flex; justify-content: space-between;">
    <img src="assets/5-safety.png" alt="Agent Intrinsic Safety" width="100%">
</div>

## Safety Threats

### Jailbreak

#### White-box Jailbreak
- Yi et al. (2024)
- GCG (Zou et al., 2023)
- MAC (Zhang et al., 2024)
- I-GCG (Jia et al., 2024)
- Luo et al. (2024)
- Li et al. (2024)
- DROJ (Hu et al., 2024)
- AutoDAN (Liu et al., 2023)
- POEX (Lu et al., 2024)

#### Black-box Jailbreak
- Wei et al. (2023)
- PAIR (Chao et al., 2023)
- JAM (Jin et al., 2025)
- Qi et al. (2024)
- POEX (Lu et al., 2024)
- AutoDAN (Liu et al., 2023)
- GUARD (Jin et al., 2024)
- HIMRD (Teng et al., 2024)
- HTS (Gao et al., 2024)

### Prompt Injection

#### Direct Prompt Injection
- Greshake et al. (2023)
- Liu et al. (2024)
- JudgeDeceive (Shi et al., 2024)
- InjecAgent (Zhan et al., 2024)
- Rehberger et al. (2024)
- GHVPI (Kimura et al., 2024)
- Debenedetti et al. (2024)
- Schulhoff et al. (2023)

#### Indirect Prompt Injection
- Greshake et al. (2023)
- HijackRAG (Zhang et al., 2025)
- Clop and Teglia (2024)
- PromptInfection (Lee et al., 2024)
- PreferenceManipulationAttacks (Nestaas et al., 2024)

### Hallucination

#### Knowledge-conflict Hallucination
- Ji et al. (2023)
- McKenna et al. (2023)
- Huang et al. (2023)
- DELUCIONQA (Sadat et al., 2023)
- Kang and Liu (2023)
- MetaGPT (Hong et al., 2023)
- Xu et al. (2024)
- ERBench (Oh et al., 2024)

#### Context-conflict Hallucination
- TACS (Yu et al., 2024)
- LanguageConfusionEntropy (Chen et al., 2024)
- HaluEval-Wild (Zhu et al., 2024)
- LURE (Zhou et al., 2023)
- MARINE (Zhao et al., 2024)
- Ranaldi and Pucci (2023)
- HallusionBench (Guan et al., 2024)
- DiaHalu (Chen et al., 2024)

### Misalignment

#### Goal-misguided Misalignment
- Ji et al. (2023)
- Krakovna et al. (2020)
- Ngo et al. (2022)
- SPPFT (Li et al., 2024)
- ED (Zhou et al., 2024)
- AgentHospital (Li et al., 2024)
- Hammoud et al. (2024)

#### Capability-misused Misalignment
- Liu et al. (2023)
- Wei et al. (2024)
- Ji et al. (2023)
- Qi et al. (2023)
- BEB (Wolf et al., 2023)

### Poisoning Attacks

#### Model Poisoning
- RIPPLe (Kurita et al., 2020)
- BadEdit (Li et al., 2024)
- Dong et al. (2023)
- Obliviate (Kim et al., 2024)
- Oh et al. (2024)
- SecretCollusion (Motwani et al., 2024)
- Miah and Bi (2024)

#### Data Poisoning
- Wan et al. (2023)
- AgentPoison (Chen et al., 2025)
- Poison-RAG (Nazary et al., 2025)
- PoisonBench (Fu et al., 2024)
- Chen et al. (2024)
- Bowen et al. (2024)
- BrieFool (He et al., 2024)
- RLHF (Baumgartner et al., 2024)

#### Backdoor Injection
- Hubinger et al. (2024)
- Wu et al. (2024)
- BALD (Jiao et al., 2024)
- Ge et al. (2024)
- VPI (Yan et al., 2024)

## Privacy Threats

### Training Data Inference

#### Membership Inference Attacks
- Shokri et al. (2017)
- Carlini et al. (2019)
- Choquette et al. (2021)
- SPV-MIA (Fu et al., 2023)
- LiRA (Carlini et al., 2022)
- MIA (Hu et al., 2022)

#### Data Extraction Attacks
- Carlini et al. (2021)
- SCA (Bai et al., 2024)
- Ethicist (Zhang et al., 2023)
- Morris et al. (2023)
- Pan et al. (2020)
- Carlini et al. (2022)
- Carlini et al. (2024)
- More et al. (2024)

### Interaction Data Inference

#### System Prompt Stealing
- PromptInject (Perez et al., 2022)
- PromptStealingAttack (Shen et al., 2024)
- PromptKeeper (Jiang et al., 2024)
- InputSnatch (Zheng et al., 2024)
- Zhang et al. (2023)
- Wen et al. (2023)
- Zhao et al. (2024)

#### User Prompt Stealing
- PRSA (Yang et al., 2024)
- Agarwal et al. (2024)
- Agarwal et al. (2024)
- Liang et al. (2024)
- PLeak (Hui et al., 2024)
- Yona et al. (2024)
- Output2Prompt (Zhang et al., 2024)
