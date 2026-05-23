# In Situ Graph Reasoning and Knowledge Expansion Using Graph-PRefLexOR

Markus J. Buehler

Laboratory for Atomistic and Molecular Mechanics, MIT, Cambridge, Massachusetts, USA

Correspondence: Markus J. Buehler [\(mbuehler@MIT.EDU\)](mailto:mbuehler@MIT.EDU)

Received: 14 January 2025 | Revised: 27 March 2025 | Accepted: 30 March 2025

Keywords: artificial intelligence | category theory | engineering | graph theory | isomorphisms | language modeling | materials science | materiomics | reasoning | science

## ABSTRACT

The pursuit of automated scientific discovery has evolved from symbolic logic to modern AI, advancing reasoning and pattern recognition. While large language models (LLMs) excel in fluency and recall, they often struggle with structured reasoning and symbolic abstraction. We introduce Graph-PRefLexOR (Graph-based Preference-based Recursive Language Modeling for Exploratory Optimization of Reasoning), a framework that extends transformer-based LLMs with explicit graph construction, symbolic abstraction, and recursive refinement during answer generation. This architecture integrates graph-based representations into language generation, enabling interpretable, multistep reasoning. Inspired by reinforcement learning and category theory, Graph-PRefLexOR defines reasoning as a structured mapping from a user task T to knowledge graph G, patterns P, and answer A. Concepts become nodes, and relationships form edges, supporting hierarchical inference and adaptive learning. Demonstrations span hypothesis generation, materials design, and creative reasoning—such as linking mythological ideas like "thin places" to materials science—showcasing generalization beyond training domains. We propose a "knowledge garden growth" strategy to foster interdisciplinary insights. A 3-billion-parameter Graph-PRefLexOR outperforms standard LLMs in reasoning depth, structure, and applicability. This work lays the foundation for general-purpose, interpretable reasoning systems that can drive scientific discovery.

## 1 | Introduction

Discovery is driven by the ability to think deeply, reflect, and iteratively refine ideas before arriving at conclusions. Emergent artificial intelligence (AI) methods have achieved success in domains such as natural language processing [\[1](#page-36-0)–[7](#page-37-0)], materials science, and molecular biology, including protein folding [\[8](#page-37-0)], and others. However, many of these advances rely on models that produce outputs directly, without an explicit intermediate reasoning process akin to scientific thinking, and hence, generally lack self-awareness and capacity to reflect. For AI to ultimately accelerate discovery, we must develop systems capable of reasoning in relational and symbolical manners [\[9](#page-37-0)–[13](#page-37-0)], reflecting critically, and explaining their answers in an iterative process that mirrors scientific methods of exploration.

When we solve a problem by identifying how things are connected and flow together, we often discover a template that works far beyond our original situation, serving as a model that can be understood in a different context or applied to a new scenario to extrapolate. For example, the hierarchical structure of bone, combining stiffness and toughness across scales, has inspired the design of lightweight, durable materials for aerospace and architecture [[15](#page-37-0)–[17\]](#page-37-0). Category theory [[9](#page-37-0)–[11](#page-37-0), [18](#page-37-0), [19](#page-37-0)], a branch of mathematics that emphasizes how objects relate rather than their internal detail, can be a powerful tool to construct models of complex phenomena through a lens on relational aspects. At its core, category theory emphasizes morphisms—arrows that represent interactions or mappings between objects. Such an approach can reveal hidden patterns—where the same structures keep appearing in nature, human organizations, technology, and

This is an open access article under the terms of the [Creative Commons Attribution](http://creativecommons.org/licenses/by/4.0/) License, which permits use, distribution and reproduction in any medium, provided the original work is properly cited.

© 2025 The Author(s). Advanced Intelligent Discovery published by Wiley-VCH GmbH.

<span id="page-1-0"></span>science. By mapping out these connections and flows, we can unlock powerful solutions that have worked across many fields. We can view this like discovering a universal language that helps translate successful ideas from one domain to another, letting us recognize and apply useful, and transferrable, patterns wherever they might be useful (Figure 1). In the example depicted in the figure, we show how flaw-tolerance in alpha-helical protein meshes can be modeled as relational abstract patterns and then

FIGURE 1 | Visualization of generalization via abstraction. Top: Example, where a phenomenon in an original domain (here: protein materials fracture, specifically flaw-tolerance in alpha-helical protein meshes [[14\]](#page-37-0)) is modeled as relational abstract patterns and then used to describe distinct phenomena in other domains. The diagram shows how structural patterns in protein materials can be abstracted and applied across domains through categorical mappings and graph-based relationships. The three-level hierarchy demonstrates functorial relationships between source domain concepts (protein materials), abstract pattern recognition, and diverse applications in networks, social systems, biological systems, and musical composition. Bottom: Flowchart for visualizing the process from a task to a graph representation (with shared relational descriptions such as IS-A, RELATES-TO, and INFLUENCES), symbolic abstraction, hypothesis generation, and the final answer. The vertical dashed line with mathematical symbols (α, β, δ, → ) represents the shared representation of all problems in tokenized form, where the model learns to generalize representations across domains.

used to describe phenomena in other domains [[9](#page-37-0), [11](#page-37-0), [14](#page-37-0), [20](#page-37-0)]. The diverse domains explored in this work—mechanics, materials science, biology, music, and social systems—may initially appear unrelated, but they all share fundamental structural and relational motifs. These include hierarchical organization (e.g., the multiscale architecture of bone or proteins, social hierarchies, musical composition), feedback loops (e.g., cellular regulation, musical dynamics, mechanical control systems), and symmetry (e.g., structural balance in biological forms, harmonic symmetry in music, equilibrium in physical systems). These shared motifs make them excellent candidates for symbolic abstraction and cross-domain reasoning. Our prior work in bioinspired materials has repeatedly shown that concepts such as isomorphisms and structural analogies often bridge seemingly distant fields, suggesting that a reasoning framework capable of operating on these abstractions can naturally generalize across disciplines. By applying Graph-PReFLexOR to this spectrum of domains, we demonstrate not only its versatility, but also its ability to uncover hidden connections, promote creative insight, and support transferable scientific reasoning. A central concept is that of an isomorphism, a special type of morphism that establishes a structural equivalence between objects, enabling insights from one domain to be applied to another. For example, consider Newton's second law of motion, F = ma, which describes the proportionality between force F, mass m, and acceleration a. This relationship is isomorphic to Ohm's law in electrical circuits, V =IR, where voltage V plays the role of force, current I corresponds to acceleration, and resistance R serves as the proportionality factor akin to mass. Both laws can be abstracted into a general form, y=kx, where y represents the driving factor, x the response, and k the proportionality constant. This abstraction reveals that the structural relationships governing these systems are fundamentally equivalent, even though the physical quantities differ. By identifying such isomorphisms, category theory allows scientists to transfer insights across domains, uncovering deep connections between seemingly unrelated phenomena. This approach not only simplifies complex systems but also provides a systematic framework for discovering universal laws, as it emphasizes relational properties that transcend specific instances. Conversely, if we can discover isomorphisms directly from data, it opens the possibility of uncovering new theories and equations by identifying deep structural parallels across disparate domains, revealing universal principles that might otherwise remain hidden.

## 1.1 | Modeling Isomorphisms for Generalization

To model isomorphisms computationally, we require a neural network architecture capable of capturing the structural equivalences between relational systems. Graph Isomorphism Networks (GINs) are particularly well-suited for this purpose. GINs are designed to operate on graph-structured data, where nodes represent objects and edges capture relationships or interactions. Unlike traditional graph neural networks (GNNs), which may struggle to distinguish between non-isomorphic graphs, GINs achieve maximum expressiveness for distinguishing graph structures by leveraging a theoretically grounded update rule that closely resembles the Weisfeiler-Lehman graph isomorphism test [\[21](#page-37-0)]. In a GIN, the node update function aggregates features from neighboring nodes using a weighted sum and applies a learned multi-layer perceptron (MLP) to the result, ensuring the network can capture higher-order structural properties. For example, to model the isomorphism between Newton's second law, F = m × a, and Ohm's law, V =I × R, a GIN can represent the relationships between variables and operators (e.g., force, mass, acceleration, =, × ) as nodes and their interactions as edges. By learning the graph embeddings, the GIN can identify the shared relational structure underlying these systems, demonstrating their equivalence. After processing, the embeddings for both equations would be nearly identical in the latent space, reflecting their structural similarity and isomorphic relationship as they can be mapped to the same representation (see Section S2 and Figure S4, for an example). This capability to recognize and model isomorphisms makes GINs a powerful tool for tasks that involve reasoning over relational data, such as predicting properties of molecules, designing robust materials, or uncovering universal scientific laws. How does this relate to transformerbased large language models (LLMs)? Recent work has shown that transformers implicitly function as graph isomorphism neural networks [[13](#page-37-0)], which provides a powerful lens for explicitly integrating graph-based reasoning into these architectures. Whereas ref. [[13](#page-37-0)] focused on theoretically grounded advancements in the transformer architecture itself to strengthen its expressive capacity, in this paper, we focus on utilizing a standard transformer interpreted with the expressive power of a GIN and endow it with explicit capabilities to conduct in situ graph reasoning and symbolic deduction. We hypothesize that by explicitly leveraging graph theory within transformers, it can unlock new levels of performance, generalization, and interpretability, accelerating progress in various scientific domains. Moreover, modeling complex principles can be inspired from biology, specifically the use of hierarchically organized systems that utilize some of the core ideas of category theory in materials and scientific exploration, where natural systems are found to often reuse relational patterns and mechanisms such as amino acids, DNA, and others. Building on these ideas, it has been suggested that inspiration for such advances may come from biology, such as emergent hierarchical intelligence based on agentic systems and other strategies that mimic biological materials and design principles [\[22](#page-37-0)–26]. Related principles include concerted problem solving as illustrated in a recent study of ant dynamics [\[27](#page-37-0)], providing ample examples for latent opportunities to discover mechanisms that can be translated across domains. LLMs have demonstrated capabilities in generating human-like text and extracting insights from scientific literature. A grand challenge in the application of AI models in science in particular remains to develop algorithms that yield more sophisticated reasoning mechanisms [\[25](#page-37-0), [28](#page-37-0)–[35\]](#page-37-0). These models often excel in fluency and pattern recognition, yet typically lack the ability to engage in structured reasoning. Addressing this limitation, earlier work like PRefLexOR (abbreviation of: Preference-based Recursive Language Modeling for Exploratory Optimization of Reasoning) [[36](#page-37-0)] introduced a framework where models were developed that innately develop a thinking and reflection strategy before answering, leveraging reinforcement learning (RL) to refine reasoning through iterative reflection and dynamic feedback loops, inspired by earlier research [\[37](#page-37-0)–[40](#page-37-0)]. A key goal of the work reported in this paper is to develop a strategy that allows AI models to not simply retrieve information but to learn underlying abstractions and relational motives, and then use these insights to generalize across new domains and ultimately to expand its understanding and capabilities to generalize. Building on this foundation, we <span id="page-3-0"></span>expand the concept of "thinking before answering" to incorporate in situ graph-based reasoning. Graphs provide a powerful framework for representing relationships, causality, and structures inherent in scientific problems (Figure 2). From molecular interactions, dynamics, to material properties, many scientific phenomena are naturally represented as graphs. By enabling models to autonomously construct and manipulate these symbolic graph representations, we create a model that mimics the reflective, iterative reasoning processes integral to scientific inquiry. In transformers, knowledge is represented as a superposition of potentialities that collapse into specific outputs when conditioned on tasks. This interpretation offers a framework for enhancing both reasoning and creativity. We show how, by balancing structured coherence with divergent exploration, models can traverse new ideas, generating insights while maintaining rigorous, task-aligned solutions.

## 1.2 | Foundations of In Situ Graph Reasoning

In this work, we introduce a novel approach that unifies the linguistic fluency of LLMs with the relational reasoning capabilities that have been quite successful in architectures such as GNNs. Our model extends the reflective capabilities introduced in the original PRefLexOR model, enabling the construction of graph-based intermediate reasoning mechanisms, by taking advantage of the flexible capabilities of the transformer architecture. Through these symbolic representations via special tokens, the model engages in a "thinking phase," reasoning over the graph to refine its understanding before generating an answer. This approach not only enhances interpretability but also improves the model's ability to solve complex scientific problems requiring relational and structural reasoning. We achieve this by formalizing reasoning as a structured mapping:

$$\mathcal{M}: \mathcal{T} \to (\mathcal{G}, \mathcal{P}, \mathcal{A}),$$
 (1)

where a task T produces a knowledge graph G, abstract patterns P, and final answers A. The knowledge graph

$$\mathcal{G} = (V, E) \tag{2}$$

encodes concepts as nodes V and relationships as directed edges E. By explicitly constructing and abstracting relational graphs, the model can encode structural information that standard next-token prediction often overlooks or treats only implicitly.

FIGURE 2 | Overview of the approach used in this paper, presenting the concept of multistep reflection (panel a), graph-based modeling of context and tasks (panel b), abstract pattern formulation (panel c), and finally, integrated in the multistage reasoning mechanisms (panel d).

4 of 39 Advanced Intelligent Discovery, 2025

In this setup, each entity and its relations become first-class objects in the learned representation, enabling the network to detect and exploit common subgraph motifs and isomorphisms that recur across different inputs. Unlike pure sequence-based transformers, which must infer latent structure solely from token order and distributional cues, a graph-driven approach anchors learning in explicitly linked entities, preserving both local and global connectivity. As a result, it is not only easier for the model to discover universal features (such as repeated subgraphs or underlying algebraic forms) but also more straightforward to apply symbolic abstractions, since the graph representation makes these recurring structures and higher-level patterns more salient and amenable to consistent transformations. Our model thereby offers opportunities for diverse use cases in AI for science and beyond. Tasks such as multistep reasoning, hypothesis generation, and causal inference become more robust and explainable. This work bridges connectionist and symbolic paradigms, pushing the boundaries of what AI can achieve in scientific domains [\[41](#page-37-0)–[43](#page-37-0)]. By creating models that have a more explicit process of relational deliberation, both symbolically and structurally, before answering, we take an important step toward AI systems capable of true scientific reasoning and discovery.

## 1.3 | Outline of This Paper

The plan of the paper is as follows. First, we briefly review the PRefLexOR architecture published in earlier work [\[36](#page-37-0)], then describe the training process of developing Graph-PRefLexOR, and then review several case studies of how the method can be applied. We conclude with final thoughts and a discussion on future opportunities.

## 2 | Results and Discussion

We report the results of a series of experiments and analyses conducted based on the model. We start with a review of the PRefLexOR framework [[36\]](#page-37-0), how we adapted it for graph reasoning capability, and then move on to the results of the experiments conducted. We conclude with a discussion of the results, interpretations, and an outlook to future opportunities. The goal of the following results is to evaluate the Graph-PReFLexOR framework's ability to perform structured reasoning across domains using three core mechanisms: (1) symbolic abstraction through graph and pattern construction, (2) recursive refinement using a multiagent feedback loop, and (3) generalization beyond the training distribution. We evaluate these capabilities through tasks that span materials science, biology, art, and philosophy —domains not explicitly represented in the training corpus. These examples were chosen to demonstrate how the model autonomously constructs knowledge graphs and symbolic representations, applies abstraction to derive relational patterns, and refines its responses via multi-step recursive reasoning. The inclusion of a critic model in a multiagent powered reflection loop enables systematic improvement, and the outputs are benchmarked qualitatively and quantitatively, showcasing the depth, adaptability, and structure of the reasoning process.

## 2.1 | Review of PRefLexOR

PRefLexOR, Preference-based Recursive Language Modeling for Exploratory Optimization of Reasoning [[36\]](#page-37-0), is a framework that enhances the reasoning capabilities of language models by integrating preference optimization and recursive reasoning (Figure 3). The approach relies on the introduction of special tokens such as <|thinking|>..<|/thinking|> and <|reflect|>..<|/reflect|>, which explicitly mark distinct phases of reasoning, enabling structured generation and iterative refinement. During training, the model undergoes two stages: Structured Thought Integration, where reasoning processes are guided using special tokens and optimized through Odds Ratio Preference Optimization (ORPO), and Independent Reasoning Development, which leverages masking of thinking tokens and applies Efficient Exact Optimization (EXO) [\[44](#page-38-0)] to align final outputs with preferred reasoning patterns without prescribing the specific reasoning steps. Thinking tokens allow the model to simulate intermediate reasoning steps, while masking ensures that the model infers the best reasoning pathways without direct supervision. We note that EXO's reverse Kullback–Leibler divergence [\[45\]](#page-38-0) objective promotes mode-seeking behavior, emphasizing dominant reasoning strategies over diluted alternatives. By dynamically generating datasets and employing recursive feedback loops, PRefLexOR enables models to self-teach, iteratively refining their reasoning through on-the-fly task generation and reflection. This combination of token-driven structure, recursive refinement, and preference alignment makes PRefLexOR a suitable foundation for reasoning-intensive applications.

As was shown in the original paper [\[36\]](#page-37-0), with a trained model the reasoning process can be iteratively refined at inference time,

FIGURE 3 | Overview of the PRefLexOR framework as reported in [[36](#page-37-0)], presented here for completeness. The training process involves two stages: (1) Structured Thought Integration Training, focusing on incorporating structured reasoning components, and (2) Independent Reasoning Development, aimed at fostering model autonomy in reasoning. During inference, the Recursive Reasoning Algorithm is employed to iteratively refine responses. Below, the role of reasoning components is depicted in the two training phases, showing transitions from unmasked to masked reasoning. As shown in the original PRefLexOR paper, context can be enriched with additional data during inference, implementing retrieval augmented generation (RAG).

<span id="page-5-0"></span>where the model generates intermediate reasoning (thinking phase), the critic evaluates it (reflection phase) and then produces improved responses. Formally, the reasoning at step i, Ri, is represented as:

$$\mathbf{R}_{i+1} = f_{\text{critic}}(\mathbf{R}_i, \mathbf{F}_i), \tag{3}$$

where f critic applies feedback F<sup>i</sup> to refine the intermediate reasoning. At each step, reflection is guided by feedback on gaps or inconsistencies in Ri, enabling the next iteration to improve alignment with the desired output. The final response A is derived after N refinement steps as:

$$\mathbf{A} = g(\mathbf{R}_N), \tag{4}$$

where g extracts the synthesized final answer. During training, masking is applied to tokens embedded within <|thinking|>.. <|/thinking|>, requiring the model to infer reasoning paths indirectly while optimizing the final output. This iterative combination of thinking and reflection, structured via tokens and recursive refinement, ensures that the model autonomously enhances its reasoning capabilities, providing accurate and wellaligned outputs. Here is an example:

```
Basic structure of the reasoning strategy using a
thinking phase before answering.
```

User: [User question or task]

Assistant: <|thinking|>

… [GRAPH]

…

[ABSTRACT PATTERNS] …

[ADDITIONAL THINKING STEPS]

…

<|/thinking|> [Answer]

# 2.2 | Graph-PRefLexOR Algorithm Design and Training

Here we use the original PRefLexOR framework but create a different function to develop the structured training data by teaching the model to explicitly construct graphs and symbolic representations in its thinking phase. Details are included in the Materials and Methods section, but we review highlights here. All training is done on top of the meta-llama/Llama-3.2-3B-Instruct model, a small but performant base LLM that serves as the foundational platform for development. Structured reasoning is generated on-the-fly during training as described in the Materials and Methods section. At its core, the method constructs dynamic knowledge graphs in situ <sup>G</sup>=ðV, <sup>E</sup>Þ, where nodes <sup>V</sup> represent key concepts extracted from enriched context, and edges E encode relationships such as IS-A, RELATES-TO, and INFLUENCES, common and shared relational descriptions. These relationships are quantified using a semantic scoring function <sup>f</sup>ðci, cj;rkÞ, which evaluates the significance of relationships rk between concepts ci and cj. Retrieval-Augmented Generation (RAG) is employed to enrich the input context and identify abstract patterns and dependencies within the graph, enabling higher-order reasoning. The thinking section organizes reasoning into structured headings, including "Core Concepts and Relationships," "Reasoning Steps," and "Abstract Patterns," providing a framework for logical progression and shared token representation to encourage generation (see Figure [1](#page-1-0) and the discussion in Section S2). For example, the "Core Concepts and Relationships" heading defines the key nodes and their connections in the knowledge graph, while "Reasoning Steps" outlines step-by-step deductions using the generalized formalisms. The "Abstract Patterns" section summarizes these relationships into symbolic transformations, such as α → β → γ → δ → ϵ, with proportional dependencies α ∝ ϵ indicating critical links. These structured headings not only guide the model's reasoning but also ensure interpretability and coherence in outputs and aid to boost extrapolation capacity in recursive reasoning steps. Special tokens delineate the reasoning phases, facilitating recursive refinement of the knowledge graph and symbolic abstractions during training and inference. By uniting graph-based reasoning, symbolic methods, and structured thinking sections, Graph-PRefLexOR achieves robust, interpretable performance in generating high-quality responses for reasoningintensive tasks. We train the model using the same multistage training process as done in the original model [[36](#page-37-0)], featuring first ORPO and then DPO-EXO. The training dataset is the same as in the original paper, consisting of raw text of around 1000 papers in the realm of biological and bio-inspired materials [[30](#page-37-0)].

## 2.3 | Detailed Analysis of Example Responses: Single Task-Response Pair

In the first example, we task the model as follows, challenging the model to answer a question that partially involves a materials-specific task but also invokes a domain that was not included in the training data, to test how well the model can extrapolate its structured reasoning to other domains that were not included in the training set. The task is:

# User task

User: Propose a new idea to relate music and materials.

The model responds first by thinking about the problem by outputting a series of "thoughts" enclosed within the <|thinking|> and <|/thinking|> tokens. The response closely follows the structured approach developed during training. First, the model provides a knowledge graph:

#### Knowledge graph

- \*\*Knowledge Graph:\*\*
- 1. \*\*Music\*\* -[IS-A]-> \*\*Audio Signal\*\*
- 2. \*\*Material\*\* -[IS-A]-> \*\*Physical Substance\*\*
- 3. \*\*Nonlinear Dynamic Response\*\* -[RELATES-TO]-> \*\*Material\*\*
- 4. \*\*Frequency\*\* -[RELATES-TO]-> \*\*Music\*\*
- 5. \*\*Nonlinear Dynamic Response\*\* -[INFLUENCES]->
  - \*\*Material's Mechanical Properties\*\*

- 6. \*\*Music\*\* -[INFLUENCES]-> \*\*Nonlinear Dynamic Response\*\* -[INFLUENCES]-> \*\*Material's Mechanical Properties\*\*
- 7. \*\*Frequency Spectrum\*\* -[RELATES-TO]-> \*\*Music\*\*
- 8. \*\*Frequency Spectrum\*\* -[RELATES-TO]-> \*\*Material's Mechanical Properties\*\*
- 9. \*\*Material\*\* -[INFLUENCES]-> \*\*Frequency
- Spectrum\*\* -[INFLUENCES]-> \*\*Music\*\*
- 10. \*\*Material Selection\*\* -[IS-A]-> \*\*Material\*\*
  - -[RELATES-TO]-> \*\*Music\*\* -[RELATES-TO]->
  - \*\*Material's Mechanical Properties\*\*

Next, the model develops an abstract reasoning pattern as follows:

#### Abstract pattern

- \*\*Abstract Pattern:\*\*
- α → β → γ
- \*\*Key Transformation Rule:\*\*
- α ∝ β
- \*\*Essential Condition:\*\*
- γ → α
- \*\*Explanation:\*\*
- α represents \*\*Music\*\*, β represents \*\*Material\*\*, and γ represents \*\*Material's Mechanical Properties\*\*.
- The abstract pattern α → β → γ shows the direct relationship between Music, Material, and Material's Mechanical Properties.
- The key transformation rule α ∝ β indicates that Music α has a proportional influence on Material's Mechanical Properties (β).
- The essential condition γ → α signifies that Material's Mechanical Properties (γ) are influenced by Music (α), creating a feedback loop in the system.

## Pattern Context:

Inspire a new method to tune and control the mechanical properties of materials by analyzing and manipulating the frequency spectrum of music. This method leverages the dynamic responses of materials to different frequencies, mimicking the effects of music on materials.

The model then proceeds to develop a series of additional structured mechanisms (reasoning steps, relevant materials, design principles, material properties, hypothesis, background), building on these initial steps. These are:

#### Additional thinking steps

- \*\*Reasoning Steps\*\*:
- 1. Music has dynamic responses on materials, similar to how it affects human emotions.
- 2. Analyze the frequency spectrum of music to understand its dynamic responses.
- 3. Use the frequency spectrum to tune and control the mechanical properties of materials.
- 4. This method leverages the relationship between music's frequency spectrum and material responses.

- \*\*Relevant Materials or Concepts\*\*:
- Nonlinear Dynamic Response
- Frequency Spectrum
- Material Selection
- Tuning and Control of Mechanical Properties
- \*\*Design Principles\*\*:
- \*\*Frequency Spectrum Analysis\*\*: Study the dynamic responses of materials to different frequencies.
- \*\*Material Selection\*\*: Choose materials with nonlinear dynamic responses to optimize mechanical property tuning.
- \*\*Frequency Spectrum Manipulation\*\*: Modify the frequency spectrum of music to influence material responses.
- \*\*Tuning and Control\*\*: Use the manipulated frequency spectrum to tune and control the mechanical properties of selected materials.
- \*\*Material Properties\*\*:
- \*\*Nonlinear Dynamic Response\*\*: Materials exhibit nonlinear responses to external stimuli, such as vibrations or frequencies.
- \*\*Frequency Dependence\*\*: Material responses vary with different frequencies, allowing for tuning and control.
- \*\*Mechanical Property Tuning\*\*: Materials' mechanical properties can be adjusted by manipulating their nonlinear dynamic responses to specific frequencies. \*\*Hypothesis\*\*:
- "Music's frequency spectrum can tune and control the nonlinear dynamic responses of materials, mimicking the effects of music on materials."
- \*\*Additional Background\*\*:

The relationship between music and materials involves analyzing and manipulating the dynamic responses of materials to different frequencies. This can be achieved by studying the nonlinear dynamic responses of materials to external stimuli, such as vibrations or frequencies. By understanding how materials react to various frequencies, it's possible to tune and control their mechanical properties. This approach can be applied to materials used in various applications, from engineering to biomedical devices. The key is to select materials that exhibit significant nonlinear responses to specific frequencies, allowing for targeted manipulation of their properties.

All of the above content was enclosed within thinking tokens. Once these are completed, the model produces the final answer, shown in Text Box [1.](#page-7-0) We can see that the output features rich content that we can analyze. First, Figure [4](#page-8-0) depicts a visualization of the knowledge graph and abstract reasoning pattern. We find that the relationship between music and materials is formalized through an abstract pattern derived from their fundamental interactions, where the model proposes the triple system ðα, <sup>β</sup>, <sup>γ</sup>Þ where <sup>α</sup> represents music as an audio signal, <sup>β</sup> represents the material as a physical substance, and γ represents the material's mechanical properties. The core relationship follows the pattern α → β → γ with the key transformation rule α ∝ β and essential condition γ → α, proposing a closed feedback loop. This system manifests through nonlinear dynamic responses of materials to different frequencies, where the frequency

## <span id="page-7-0"></span>Final answer (after thinking)

\*\*Proposed Idea: "Music-Inspired Material Tuning"\*\*

Inspire a new method to tune and control the mechanical properties of materials by analyzing and manipulating the frequency spectrum of music. This method leverages the dynamic responses of materials to different frequencies, mimicking the effects of music on materials.

- \*\*Key Concepts:\*\*
- 1. \*\*Nonlinear Dynamic Response\*\*: Materials exhibit nonlinear responses to external stimuli, such as vibrations or frequencies.
- 2. \*\*Frequency Dependence\*\*: Material responses vary with different frequencies, allowing for tuning and control.
- 3. \*\*Frequency Spectrum Analysis\*\*: Study the dynamic responses of materials to different frequencies by analyzing their frequency response curves or using techniques like dynamic mechanical analysis (DMA).
- 4. \*\*Material Selection\*\*: Choose materials with nonlinear dynamic responses to specific frequencies, such as those exhibiting high damping or stiffness at particular frequencies.
- 5. \*\*Music Frequency Spectrum Manipulation\*\*: Modify the frequency spectrum of music to influence material responses. This can be achieved by filtering or modifying the frequency content of the music to match the target material's nonlinear dynamic responses.
- 6. \*\*Tuning and Control\*\*: Use the manipulated frequency spectrum to tune and control the mechanical properties of selected materials, such as damping, stiffness, or damping capacity.
- \*\*Design Principles:\*\*
- 1. \*\*Frequency Spectrum Analysis\*\*: Study the dynamic responses of materials to different frequencies to understand their nonlinear dynamic responses.
- 2. \*\*Material Selection\*\*: Choose materials with nonlinear dynamic responses to specific frequencies, such as those exhibiting high damping or stiffness at particular frequencies.
- 3. \*\*Frequency Spectrum Manipulation\*\*: Modify the frequency spectrum of music to influence material responses, using techniques like filtering or modifying the frequency content of the music.
- 4. \*\*Tuning and Control\*\*: Use the manipulated frequency spectrum to tune and control the mechanical properties of selected materials, such as damping, stiffness, or damping capacity.
- \*\*Potential Applications:\*\*
- 1. \*\*Material Tuning\*\*: Develop a method to tune and control the mechanical properties of materials, such as damping or stiffness, by manipulating the frequency spectrum of music.
- 2. \*\*Nonlinear Dynamic Response Measurement\*\*: Use music's frequency spectrum to measure nonlinear dynamic responses of materials, providing a non-contact and non-destructive method.
- 3. \*\*Material Selection\*\*: Identify materials with nonlinear dynamic responses to specific frequencies, aiding in the selection of materials for specific applications.
- 4. \*\*Biomedical Applications\*\*: Apply the principle of Music-Inspired Material Tuning to develop novel, non-invasive, and nondestructive techniques for monitoring or treating diseases, such as cancer or neurological disorders.
- \*\*Challenges and Future Directions:\*\*
- 1. \*\*Material Selection\*\*: Identifying materials with nonlinear dynamic responses to specific frequencies can be challenging, requiring extensive testing and characterization.
- 2. \*\*Frequency Spectrum Manipulation\*\*: Modifying the frequency spectrum of music to influence material responses can be difficult, as it depends on the specific material's nonlinear dynamic responses and the desired mechanical properties.
- 3. \*\*Scalability\*\*: Scaling the Music-Inspired Material Tuning method to larger structures or more complex materials can be challenging, requiring further research and development.
- 4. \*\*Theoretical Modeling\*\*: Developing theoretical models to explain and predict the nonlinear dynamic responses of materials to music's frequency spectrum can be an ongoing research direction.
- \*\*Conclusion:\*\*

The proposed method, "Music-Inspired Material Tuning," offers a novel approach to tune and control the mechanical properties of materials by analyzing and manipulating the frequency spectrum of music. This method leverages the dynamic responses of materials to different frequencies, mimicking the effects of music on materials. By understanding and manipulating the nonlinear dynamic responses of materials to specific frequencies, this method can be applied to various materials and applications, providing a non-contact and non-destructive method for tuning and controlling material properties.

spectrum of music directly influences material properties. The relationship leverages the dynamic responses of materials to different frequencies, mimicking the effects of music on materials. Through frequency spectrum analysis and manipulation, the proposed method enables tuning and control of mechanical properties such as possibly damping, stiffness, and damping capacity. This approach opens possibilities for nondestructive material testing, property tuning, and biomedical applications, where the nonlinear dynamic response provides a mechanism for controlled material modification through specific frequency interactions. Thinking about the idea more deeply, the proposal's main novelty lies in its symmetrical conceptualization of music– material interaction as a feedback loop. Rather than just treating music as an input that affects materials (which would be more

<span id="page-8-0"></span>FIGURE 4 | Overview of the data generated in response to the question: Propose a new idea to relate music and materials. The knowledge graph (top) illustrates the relationships between core concepts: music (blue), material properties (purple), and frequency spectrum (green). Key relationships include IS-A hierarchies (e.g., Music is an Audio Signal) and influence paths through nonlinear dynamic responses. The abstract pattern (bottom) formalizes these interactions through a triple system (α, β, γ) with proportional influence (α ∝ β) and feedback loop (γ → α). The integration between graph and pattern manifests in multiple ways: the music-to-material influence path in the graph maps to α → β in the pattern; the material's mechanical properties feedback in the graph corresponds to the essential condition γ → α; and the frequency spectrum mediation shown in the graph provides the physical mechanism for the proportional influence (α ∝ β) in the pattern. Dotted lines explicitly map concrete elements to their abstract counterparts, demonstrating how the theoretical framework emerges from and guides the practical implementation. This dual representation captures both the detailed mechanisms of music–material interaction and its fundamental mathematical structure.

obvious), it proposes a cyclic system where material properties can also influence the musical response. The proposal relies heavily on well-known concepts like nonlinear dynamic responses and frequency analysis, but the practical implementation focuses on fairly standard approaches (frequency spectrum analysis, material selection). The proposed methodology presents notable innovations in relating music and materials, particularly through its biomimetic foundation that draws parallels between material responses and human emotional reactions to music (α → response). The approach extends beyond conventional frequency-response studies by establishing a comprehensive framework that encompasses specific technical implementations through dynamic mechanical analysis (DMA), concrete property targets (damping, stiffness), and notably, potential biomedical applications for cancer and neurological disorders. The methodology's novelty lies not only in its theoretical framework but in its practical manifestation through noncontact and nondestructive testing methods. The proposal demonstrates depth in addressing implementation challenges, particularly in material selection optimization and frequency spectrum manipulation. Critically, it acknowledges and provides direction for key technical hurdles, including the scalability of the method to larger structures, the complexity of identifying materials with appropriate nonlinear dynamic responses, and the need for robust theoretical models to predict material behavior under musical frequency

stimulation. The potential impact is amplified by its extension into biomedical applications, suggesting a bridge between materials science and medical diagnostics through music-inspired techniques. This comprehensive treatment of both theoretical foundations and practical challenges, coupled with the biomimetic inspiration and medical applications, establishes the proposal as a significant innovation in material property tuning methodologies, transcending traditional approaches to frequency-based material manipulation. The innovation's fundamental strength emerges from its adherence to the abstract pattern  $\alpha \to \beta \to \gamma \to \alpha$ , where the cyclic relationship enables a unique feedback system. The proportionality rule  $\alpha \propto \beta$  manifests in the biomimetic response mechanism, while the essential condition  $\gamma \rightarrow \alpha$  is realized through the material's frequency-dependent behavior. This creates a dynamic system where music  $(\alpha)$  influences material properties  $(\gamma)$  through material response  $(\beta)$ , while the resulting property changes create new response patterns, forming a continuous adaptive loop. As an interpretation, the relationship can be viewed as a coupled system where each component's influence is nonlinearly dependent on the others:

$$\frac{\partial \gamma}{\partial t} = f(\alpha, \beta)$$
 and  $\frac{\partial \beta}{\partial t} = g(\gamma, \alpha)$ ,

capturing both the immediate response and long-term evolution of material properties under musical influence. This mathematical framework elegantly supports the proposal's practical applications while maintaining the fundamental symmetry of the original abstract pattern.

# 2.4 | In Situ Graph Generation and Recursive Reasoning Through Multiagent Modeling

The earlier example showed that the model was able to expand its capabilities beyond the materials-science focused training data and was able to successfully integrate different domains into the structured reasoning paradigm. In the next experiment, we task the model as follows:

### User task

**User:** Integrate a snowflake and ant behavior to design a new tough material made from protein.

However, unlike before where one can respond with a single shot, here we use recursive reasoning by using a two-agent setup where the thinking steps are critiqued, then improved, and fed back to the model. Ultimately, an integrated response is developed by the model that incorporates the various ideas, concepts, and details developed during the recursive process. Figure 5 shows a flowchart of the process used. In the example discussed below, we use N=3 iterations until the final answer is produced through multiple refinement steps.

Each iteration in the algorithm produces an intermediate result that is critiqued and improved as delineated in Figure 5. The final answer after iteration N=3 is shown in Text Box 2. Once the maximum number of iterations has been reached, we develop an integrated answer that uses the responses from all three iterations to develop the final response to the task. For the example discussed here, the final, integrated answer is:

# Final integrated answer from recursive Graph-PRefLexOR

Designing a new tough material made from protein inspired by snowflake and ant behavior involves creating a protein-based material with fractal patterns and hierarchical structures, similar to snowflakes and ants, to enhance toughness, adaptability, and diversity in applications. The design principle for creating such a material is to mimic fractal patterns and hierarchical structures, inspired by snowflakes, and self-organization, inspired by ants, to enhance toughness, adaptability, and diversity in applications.

The key components of this design are:

- \*\*Fractal Patterns (Snowflake Inspiration):\*\* Incorporate fractal patterns and self-similarity to create intricate, complex structures. Use branching patterns and scaling laws to create patterns at multiple length scales. This will enhance the material's toughness and adaptability.
- 2. \*\*Self-Organization (Ant Inspiration):\*\* Design the protein material to mimic the self-organization and hierarchical structures found in ant colonies. Encourage self-organization through hierarchical structures at multiple length scales. This will promote adaptability and diversity in the material's properties.
- 3. \*\*Hierarchical Structure:\*\* Incorporate multiple length scales to create complex, tough materials. This will enhance the material's toughness and adaptability.

The design process involves:

- 1. \*\*Design Protein Material:\*\* Create a protein-based material with fractal patterns and hierarchical structures. Incorporate self-organization and hierarchical structures at multiple length scales.
- \*\*Test and Refine:\*\* Test the protein material's toughness, adaptability, and diversity in properties.
   Refine the design based on the test results to enhance the material's performance.

The material properties of the designed protein-based material are:

- \*\*Fractal Pattern Mimicry:\*\* Inspired by snowflakes, the protein material exhibits fractal patterns and selfsimilarity.
- \*\*Hierarchical Structure:\*\* Mimics ant behavior by creating hierarchical structures at multiple length scales.
- \*\*Self-Organization:\*\* Encourages self-organization, similar to ants, to enhance toughness and adaptability.
- \*\*Toughness Enhancement:\*\* Combines fractal patterns and hierarchical structures to create tough, protein-based materials.
- \*\*Adaptability:\*\* Designed for diverse applications by mimicking snowflake and ant patterns.

The potential applications of the designed protein-based material are:

- \*\*Biomedical Applications:\*\* Inspired by snowflake and ant patterns, protein-based materials can be used in biomedical applications such as tissue engineering, wound dressing, and drug delivery.
- \*\*Biodegradable Materials:\*\* Protein-based materials can be designed to be biodegradable, providing a sustainable alternative to traditional materials.

- \*\*Soft Robotics:\*\* Inspired by ant behavior, proteinbased materials can be used in soft robotics, enabling more flexible and adaptable robotic systems.
- \*\*Biomineralization:\*\* Protein-based materials can mimic natural biomineralization processes, providing insights into bone and tooth formation.

The challenges and opportunities associated with designing protein-based materials inspired by snowflake and ant behavior are:

- \*\*Scalability:\*\* Maintaining fractal patterns and hierarchical structures at multiple length scales while scaling up the material.
- \*\*Self-Organization:\*\* Encouraging self-organization in protein materials to enhance toughness and adaptability.
- \*\*Biocompatibility:\*\* Ensuring the protein material is biocompatible and non-toxic for potential applications.
- \*\*Synthesis Methods:\*\* Developing efficient and scalable methods for synthesizing protein-based materials with fractal patterns and hierarchical structures.

Future research directions include:

- \*\*Fractal Geometry of Protein Materials:\*\* Investigating the fractal geometry of protein materials and its role in enhancing toughness and adaptability.
- \*\*Ant-Inspired Protein Materials:\*\* Developing proteinbased materials that mimic ant behavior, such as selforganization and hierarchical structures.
- \*\*Biodegradable Protein Materials:\*\* Designing proteinbased materials that are biodegradable and sustainable.
- \*\*Synthesis Methods for Fractal Protein Materials:\*\* Developing efficient and scalable methods for synthesizing protein-based materials with fractal patterns and hierarchical structures.

In conclusion, designing protein-based materials inspired by snowflake and ant behavior offers a promising approach to creating tough, adaptable, and diverse materials. By mimicking fractal patterns and hierarchical structures, and self-organization, protein materials can be engineered to enhance toughness, adaptability, and diversity in applications. However, challenges such as scalability, selforganization, biocompatibility, and synthesis methods need to be addressed. Future research should focus on fractal geometry, ant-inspired protein materials, biodegradable protein materials, and efficient synthesis methods to unlock the full potential of protein-based materials.

For a comparison, the answer from a standard (nonfine-tuned) model is shown in Text Box S1. We note that this response does not include any thinking section (and hence, no graph reasoning or abstraction), and solely consists of the text as provided. We first compare the results from the first iteration of the graph reasoning model with the standard model. From the standpoint of a scientist at a high technical level, the graph reasoning-based response demonstrates superior intellectual depth and reasoning quality. The response provides a highly systematic framework for material design, detailing explicit parameters (e.g., protein selection, fractal pattern generation, hierarchical structure formation), and an iterative validation process that reflects rigorous scientific methodology and clear delineation in graph form (Figure [6](#page-15-0)). This structured approach aligns well with the expectations of graduate-level research, as it includes clear constraints, performance criteria, and testing protocols akin to what one might find in peer-reviewed scientific literature. By contrast, the response from the standard model showcases a more conceptual, broader-stroke approach and lacks the comprehensive methodological granularity seen in the other response. While it does incorporate topics such as self-healing peptides and adaptive camouflage, its overall elaboration remains more conceptual than procedural. Thus, the graph reasoning result more closely meets the benchmarks of high rigor by systematically addressing design parameters, experimental validation, and scalability concerns. We compare the two responses using GPT-4o as evaluator, with results shown in Table [1.](#page-16-0) Graph-PRefLexOR systematically addresses the design principles (fractals, hierarchical structures, self-organization) and walks through potential challenges (scalability, synthesis methods, biocompatibility) with a more methodical causeand-effect flow and hence offers a deeper theoretical reasoning overall. In contrast, the comparison with the nonfine-tuned model is much shorter and more general. It does not dig as deeply into the theoretical underpinnings, deep materials-specific reasoning, or longer-term challenges. The graph-based response offers more robust, multilayered reasoning.

## 2.5 | Growing a Knowledge Garden Via In Situ Graph Generation and Knowledge Expansion

In this next experiment, we use the capability of the graph reasoning model to generate graphs that describe phenomena in a relational manner and apply it to explore ever-expanding graph structures by repeatedly prompting the model. This allows us to recursively grow a graph, starting from an initial concept or response to a task, and to examine whether the model can blend creative and analytical reasoning. The initial task is:

## User task

User: Write a song by incorporating snow flakes and flower petals.

#### Knowledge graph

- \*\*Knowledge Graph:\*\*
  - 1. \*\*Snow Flakes\*\* -[IS-A]-> \*\*Natural Phenomenon\*\*
- 2. \*\*Flower Petals\*\* -[IS-A]-> \*\*Natural Phenomenon\*\*
- 3. \*\*Snow Flakes\*\* -[RELATES-TO]-> \*\*Winter\*\*
- 4. \*\*Flower Petals\*\* -[RELATES-TO]-> \*\*Spring\*\*
- 5. \*\*Snow Flakes\*\* -[INFLUENCES]-> \*\*Mood\*\* (e.g., Serene, Calming)
- 6. \*\*Flower Petals\*\* -[INFLUENCES]-> \*\*Mood\*\* (e.g., Delicate, Whimsical)
- 7. \*\*Snow Flakes\*\* -[INFLUENCES]-> \*\*Nature's Beauty\*\*
- 8. \*\*Flower Petals\*\* -[INFLUENCES]-> \*\*Nature's Beauty\*\*
- 9. \*\*Snow Flakes\*\* -[INFLUENCES]-> \*\*Seasonal Changes\*\*
- 10. \*\*Flower Petals\*\* -[INFLUENCES]-> \*\*Seasonal Changes\*\*

<span id="page-11-0"></span>FIGURE 5 | Graph-PRefLexOR Recursive Reasoning Algorithm, using graph reasoning and abstract representations of relational mechanics, using a multiagent system with Agent #1 being the Graph Reasoning model, and Agent #2 being a general-purpose critic model. The reflection is generated using the Critic agent and then used to improve the thinking process. This resembles an iterative approach leveraging the Reasoning Model and a general-purpose Critic Model to generate, refine, and optionally integrate responses. As before, the process ultimately involves generating initial responses, extracting reflections, improving thinking processes, and creating new responses based on refined thinking, with an optional final integration step. The algorithm relies on extracting thinking processes (indicated via <|thinking|> ..<|/thinking|> ) and reflection processes generated by the Critic. The sampled responses can either be used in their final state or integrated into an amalgamated response that shows very rich facets in the scientific process.

TEXT BOX 2 | Final answer after iteration 3 from recursive Graph-PRefLexOR.

## Final answer after iteration 3 from recursive Graph-PRefLexOR

Based on the provided context, here's a comprehensive design for a tough, protein-based material inspired by snowflake and ant behavior:

- \*\*Design Concept:\*\* "Snowflake-Ant Inspired Protein Material" (SAIPM)
- \*\*Design Principles:\*\*
- 1. \*\*Fractal Pattern Mimicry\*\*: Incorporate fractal patterns and self-similarity to enhance toughness and adaptability, inspired by snowflakes.
- 2. \*\*Hierarchical Structure\*\*: Create hierarchical structures at multiple length scales to mimic ant behavior, promoting self-organization and toughness.
- 3. \*\*Protein Selection\*\*: Choose proteins with high strength, toughness, and adaptability, such as collagen, silk, or silk-like proteins.
- 4. \*\*Molecular Assembly\*\*: Design a molecular assembly process that allows for easy incorporation of fractal patterns and hierarchical structures, inspired by snowflakes and ants.
- 5. \*\*Self-Assembly\*\*: Encourage self-assembly of protein molecules to form complex structures, mimicking ant behavior.
- \*\*Design Structure:\*\*
- 1. \*\*Fractal Pattern Generation\*\*:
  - Use computational models to generate fractal patterns inspired by snowflakes.
  - Incorporate these patterns into protein sequences or structures.

- Design a hierarchical structure at multiple length scales, inspired by ant behavior.
- Use techniques like layer-by-layer assembly or 3D printing to create these structures.
- 3. \*\*Protein Selection and Preparation\*\*:
  - Choose proteins with high strength, toughness, and adaptability.
  - Prepare these proteins in a form suitable for incorporation into the SAIPM structure.
- 4. \*\*Molecular Assembly\*\*:
  - Design a molecular assembly process that allows for easy incorporation of fractal patterns and hierarchical structures.
  - Use techniques like covalent bonding, cross-linking, or ionic interactions to assemble protein molecules.
- 5. \*\*Self-Assembly\*\*:
  - Encourage self-assembly of protein molecules to form complex structures.
- Use techniques like pH-induced unfolding, temperature-dependent assembly, or external stimuli to control the assembly process.
- \*\*Design Parameters:\*\*
- \*\*Fractal Pattern Parameters\*\*:
  - Order of fractal pattern (e.g., octaves)
  - Fractal pattern size (e.g., diameter, length)
  - Fractal pattern orientation (e.g., angle, direction)
- \*\*Hierarchical Structure Parameters\*\*:
  - Number of hierarchical levels
  - Length scale of each level (e.g., diameter, length)
  - Orientation of each level (e.g., angle, direction)
- \*\*Protein Parameters\*\*:
  - Protein type (e.g., collagen, silk)
  - Protein concentration
  - Protein processing or modification (e.g., denaturation, cross-linking)
- \*\*Molecular Assembly Parameters\*\*:
  - Assembly method (e.g., covalent bonding, ionic interactions)
  - Assembly conditions (e.g., pH, temperature, solvent)
- \*\*Self-Assembly Parameters\*\*:
  - Assembly trigger (e.g., pH, temperature, external stimuli)
  - Assembly duration or rate
- \*\*Design Evaluation Criteria:\*\*
- \*\*Toughness\*\*: Measure the SAIPM's resistance to mechanical stress, such as tensile strength, compressive strength, or impact resistance.
- \*\*Adaptability\*\*: Evaluate the SAIPM's ability to adapt to different environments or applications, such as self-healing, self-repair, or self-adaptation.
- \*\*Strength\*\*: Measure the SAIPM's strength in different directions or under various loading conditions.
- \*\*Energy Dissipation\*\*: Assess the SAIPM's ability to dissipate energy under mechanical stress, such as through viscoelasticity or crack deflection.
- \*\*Biocompatibility\*\*: Evaluate the SAIPM's compatibility with biological systems, including cell adhesion, cell growth, and biodegradation.
- \*\*Synthesis Efficiency\*\*: Assess the efficiency of the molecular assembly process and the SAIPM's production rate.
- \*\*Design Validation:\*\*
- \*\*Toughness Validation\*\*: Test the SAIPM under various mechanical stress conditions (e.g., tensile, compressive, impact) to confirm its toughness.
- \*\*Adaptability Validation\*\*: Evaluate the SAIPM's adaptability by exposing it to different environments or applying it to various applications to confirm its ability to adapt.
- \*\*Strength Validation\*\*: Measure the SAIPM's strength in different directions or under various loading conditions to confirm its strength.
- \*\*Energy Dissipation Validation\*\*: Assess the SAIPM's ability to dissipate energy under mechanical stress by analyzing its viscoelastic properties or observing crack deflection.
- \*\*Biocompatibility Validation\*\*: Test the SAIPM's biocompatibility by observing cell adhesion, cell growth, and biodegradation.
- \*\*Synthesis Efficiency Validation\*\*: Evaluate the efficiency of the molecular assembly process and the SAIPM's production rate by comparing the synthesized material to theoretical predictions.
- \*\*Design Iteration:\*\*
- \*\*Feedback Loop\*\*: Establish a feedback loop to collect data on the SAIPM's performance, including mechanical stress, adaptability, strength, energy dissipation, biocompatibility, and synthesis efficiency.
- \*\*Design Revisions\*\*: Use the collected data to revise the design, adjusting parameters such as fractal pattern order, hierarchical level number, protein type, assembly method, or assembly conditions to optimize the SAIPM's performance.

29439981, 2025, 3, Downloaded from https://advanced.onlinelibrary.wiley.com/doi/10.1002/aidi.202500006 by Massachusetts Institute of Technolo, Wiley Online Library on [16/02/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License

- \*\*Reiteration\*\*: Repeat the design iteration process until the SAIPM meets the desired performance criteria or until further improvements are not feasible.
- \*\*Design Requirements:\*\*
- \*\*Toughness\*\*: The SAIPM should exhibit high toughness, as measured by tensile strength, compressive strength, or impact resistance.
- \*\*Adaptability\*\*: The SAIPM should demonstrate adaptability, such as self-healing, self-repair, or self-adaptation, to meet the specific requirements of various applications.
- \*\*Strength\*\*: The SAIPM should exhibit high strength in different directions or under various loading conditions.
- \*\*Energy Dissipation\*\*: The SAIPM should be able to dissipate energy under mechanical stress, such as through viscoelasticity or crack deflection.
- \*\*Biocompatibility\*\*: The SAIPM should be compatible with biological systems, including cell adhesion, cell growth, and biodegradation.
- \*\*Synthesis Efficiency\*\*: The SAIPM's molecular assembly process should be efficient and scalable for large-scale production.
- \*\*Design Constraints:\*\*
- \*\*Protein Selection\*\*: The choice of protein is limited by its strength, toughness, and adaptability.
- \*\*Molecular Assembly\*\*: The assembly method and conditions are constrained by the need for efficient and scalable production.
- \*\*Self-Assembly\*\*: The assembly process is limited by the trigger and duration or rate of assembly.
- \*\*Fractal Pattern Generation\*\*: The fractal pattern generation process is limited by the complexity of the patterns and the computational resources required.
- \*\*Hierarchical Structure Creation\*\*: The hierarchical structure creation process is limited by the number of hierarchical levels and the length scale of each level.
- \*\*Design Requirements and Constraints:\*\*
- \*\*Design Requirements:\*\*
  - Toughness: High tensile strength, compressive strength, or impact resistance.
  - Adaptability: Self-healing, self-repair, or self-adaptation.
  - Strength: High in different directions or under various loading conditions.
  - Energy Dissipation: Ability to dissipate energy under mechanical stress.
  - Biocompatibility: Compatibility with biological systems.
  - Synthesis Efficiency: Efficient and scalable molecular assembly process.
- \*\*Design Constraints:\*\*
  - Protein Selection: Limited by protein strength, toughness, and adaptability.
  - Molecular Assembly: Limited by assembly method and conditions.
  - Self-Assembly: Limited by assembly trigger and duration or rate.
  - Fractal Pattern Generation: Limited by fractal pattern order and size.
  - Hierarchical Structure Creation: Limited by number of hierarchical levels and length scale.
- Protein Preparation: Limited by protein processing or modification.
- \*\*Design Principles and Materials:\*\*

## Design Principles:

- 1. \*\*Fractal Pattern Mimicry\*\*: Incorporate fractal patterns and self-similarity to enhance toughness and adaptability.
- 2. \*\*Hierarchical Structure\*\*: Create hierarchical structures at multiple length scales to promote self-organization and toughness.
- 3. \*\*Protein Selection\*\*: Choose proteins with high strength, toughness, and adaptability.
- 4. \*\*Molecular Assembly\*\*: Design a molecular assembly process that allows for easy incorporation of fractal patterns and hierarchical structures.
- 5. \*\*Self-Assembly\*\*: Encourage self-assembly of protein molecules to form complex structures.

# Materials:

- \*\*Protein\*\*: Collagen, silk, or silk-like proteins.
- \*\*Molecular Assembly Method\*\*: Covalent bonding, ionic interactions, or other techniques that allow for easy incorporation of fractal patterns and hierarchical structures.
- \*\*Assembly Conditions\*\*: pH, temperature, solvent, or other conditions that facilitate self-assembly.
- \*\*Design Requirements and Materials:\*\*

## Design Requirements:

- \*\*Protein\*\*: High strength, toughness, and adaptability (e.g., collagen, silk).
- \*\*Molecular Assembly Method\*\*: Efficient and scalable (e.g., covalent bonding, ionic interactions).
- \*\*Assembly Conditions\*\*: Triggered by pH, temperature, or external stimuli.
- \*\*Design Principles and Materials:\*\*

### Design Principles:

- 1. \*\*Fractal Pattern Mimicry\*\*: Incorporate fractal patterns and self-similarity to enhance toughness and adaptability.
- 2. \*\*Hierarchical Structure\*\*: Create hierarchical structures at multiple length scales to promote self-organization and toughness.
- 3. \*\*Protein Selection\*\*: Choose proteins with high strength, toughness, and adaptability.

- 4. \*\*Molecular Assembly\*\*: Design a molecular assembly process that allows for easy incorporation of fractal patterns and hierarchical structures.
- 5. \*\*Self-Assembly\*\*: Encourage self-assembly of protein molecules to form complex structures.

## Materials:

- \*\*Protein\*\*: Collagen, silk, or silk-like proteins.
- \*\*Molecular Assembly Method\*\*: Covalent bonding, ionic interactions, or other techniques that allow for easy incorporation of fractal patterns and hierarchical structures.
- \*\*Assembly Conditions\*\*: pH, temperature, solvent, or other conditions that facilitate self-assembly.
- \*\*Design Requirements and Materials:\*\*

#### Design Requirements:

- \*\*Protein\*\*: High strength, toughness, and adaptability (e.g., collagen, silk).
- \*\*Molecular Assembly Method\*\*: Efficient and scalable (e.g., covalent bonding, ionic interactions).
- \*\*Assembly Conditions\*\*: Triggered by pH, temperature, or external stimuli.
- \*\*Design Principles and Materials:\*\*

#### Design Principles:

- 1. \*\*Fractal Pattern Mimicry\*\*: Incorporate fractal patterns and self-similarity to enhance toughness and adaptability.
- 2. \*\*Hierarchical Structure\*\*: Create hierarchical structures at multiple length scales to promote self-organization and toughness.
- 3. \*\*Protein Selection\*\*: Choose proteins with high strength, toughness, and adaptability.
- 4. \*\*Molecular Assembly\*\*: Design a molecular assembly process that allows for easy incorporation of fractal patterns and hierarchical structures.
- 5. \*\*Self-Assembly\*\*: Encourage self-assembly of protein molecules to form complex structures.

## Materials:

- \*\*Protein\*\*: Collagen, silk, or silk-like proteins.
- \*\*Molecular Assembly Method\*\*: Covalent bonding, ionic interactions, or other techniques that allow for easy incorporation of fractal patterns and hierarchical structures.
- \*\*Assembly Conditions\*\*: pH, temperature, solvent, or other conditions that facilitate self-assembly.
- \*\*Design Requirements and Materials:\*\*

## Design Requirements:

- \*\*Protein\*\*: High strength, toughness, and adaptability (e.g., collagen, silk).
- \*\*Molecular Assembly Method\*\*: Efficient and scalable (e.g., covalent bonding, ionic interactions).
- \*\*Assembly Conditions\*\*: Triggered by pH, temperature, or external stimuli.
- \*\*Design Principles and Materials:\*\*

#### Design Principles:

- 1. \*\*Fractal Pattern Mimicry\*\*: Incorporate fractal patterns and self-similarity to enhance toughness and adaptability.
- 2. \*\*Hierarchical Structure\*\*: Create hierarchical structures at multiple length scales to promote self-organization and toughness.
- 3. \*\*Protein Selection\*\*: Choose proteins with high strength, toughness, and adaptability.
- 4. \*\*Molecular Assembly\*\*: Design a molecular assembly process that allows for easy incorporation of fractal patterns and hierarchical structures.
- 5. \*\*Self-Assembly\*\*: Encourage self-assembly of protein molecules to form complex structures.

#### Materials:

- \*\*Protein\*\*: Collagen, silk, or silk-like proteins.
- \*\*Molecular Assembly Method\*\*: Covalent bonding, ionic interactions, or other techniques that allow for easy incorporation of fractal patterns and hierarchical structures.
- \*\*Assembly Conditions\*\*: pH, temperature, solvent, or other conditions that facilitate self-assembly.
- \*\*Design Requirements and Materials:\*\*

#### Design Requirements:

- \*\*Protein\*\*: High strength, toughness, and adaptability (e.g., collagen, silk).
- \*\*Molecular Assembly Method\*\*: Efficient and scalable (e.g., covalent bonding, ionic interactions).
- \*\*Assembly Conditions\*\*: Triggered by pH, temperature, or external stimuli.
- \*\*Design Principles and Materials:\*\*

## Design Principles:

- 1. \*\*Fractal Pattern Mimicry\*\*: Incorporate fractal patterns and self-similarity to enhance toughness and adaptability.
- 2. \*\*Hierarchical Structure\*\*: Create hierarchical structures at multiple length scales to promote self-organization and toughness.
- 3. \*\*Protein Selection\*\*: Choose proteins with high strength, toughness, and adaptability.
- 4. \*\*Molecular Assembly\*\*: Design a molecular assembly process that allows for easy incorporation of fractal patterns and hierarchical structures.
- 5. \*\*Self-Assembly\*\*: Encourage self-assembly of protein molecules to form complex structures.

# Materials:

– \*\*Protein\*\*: Collagen, silk, or silk-like proteins.

- <span id="page-15-0"></span>– \*\*Molecular Assembly Method\*\*: Covalent bonding, ionic interactions, or other techniques that allow for easy incorporation of fractal patterns and hierarchical structures.
- \*\*Assembly Conditions\*\*: pH, temperature, solvent, or other conditions that facilitate self-assembly.
- \*\*Design Requirements and Materials:\*\*

## Design Requirements:

- \*\*Protein\*\*: High strength, toughness, and adaptability (e.g., collagen, silk).
- \*\*Molecular Assembly Method\*\*: Efficient and scalable (e.g., covalent bonding, ionic interactions).
- \*\*Assembly Conditions\*\*: Triggered by pH, temperature, or external stimuli.
- \*\*Design Principles and Materials:\*\*

## Design Principles:

- 1. \*\*Fractal Pattern Mimicry\*\*: Incorporate fractal patterns and self-similarity to enhance toughness and adaptability.
- 2. \*\*Hierarchical Structure\*\*: Create hierarchical structures at multiple length scales to promote self-organization and toughness

FIGURE 6 | Overview of the graph data generated in response to the question: Integrate a snowflake and ant behavior to design a new tough material made from protein. The figure depicts both the knowledge graph alone (panel a) and an integration of the knowledge graph with the abstract pattern representation (panel b). Results after the first iteration, i =1, are shown.

<span id="page-16-0"></span>TABLE 1 | Benchmark comparison of Graph-PReFLexOR versus a nonfine-tuned baseline model, evaluated by GPT-4o. The evaluation covers intellectual depth, reasoning quality, logic, academic rigor, and graph integration. GPT-4o was used as a standardized evaluation agent to ensure consistency in qualitative assessment. Graph-PReFLexOR achieves significantly higher scores across all criteria, particularly in structured reasoning and integration of graph-based methods, with an average score of 9.7 versus 6.4.

| Criteria                     | Graph<br>reasoning<br>response<br>(Response 1) | Reference<br>model<br>(Response 2) | Explanation                                                                                                                                                                                          |
|------------------------------|------------------------------------------------|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Intellectual depth           | 10/10                                          | 7/10                               | Response 1 combines text and the graph, providing a multilayered,<br>systematic approach to design. Response 2 lacks visual support and<br>focuses on textual descriptions.                          |
| Reasoning quality            | 10/10                                          | 8/10                               | The graph in Response 1 supports robust reasoning, connecting<br>abstract concepts to practical design principles. Response 2<br>demonstrates logical reasoning but without visual reinforcement.    |
| Logic                        | 10/10                                          | 7/10                               | Response 1's<br>graph enhances the logical connections between<br>abstract patterns, protein structures, and material properties.<br>Response 2 lacks a structured framework to visually link ideas. |
| Use of academic measures     | 10/10                                          | 6/10                               | Response 1 leverages the graph to convey systematic relationships<br>and integrates validation criteria effectively. Response 2 is<br>descriptive and less rigorous in methodology.                  |
| Clarity and organization     | 8/10                                           | 9/10                               | While Response 1 provides depth, the graph may overwhelm casual<br>readers. Response 2 is simpler and more accessible but less<br>comprehensive.                                                     |
| Potential applicability      | 10/10                                          | 8/10                               | Response 1 explicitly ties abstract and biological principles to<br>practical applications, supported by the graph. Response 2<br>mentions potential applications but without detailed pathways.     |
| Overall integration of graph | 10/10                                          | 0/10                               | The graph in Response 1 enhances intellectual depth, reasoning,<br>and academic rigor. Response 2 lacks a graph entirely.                                                                            |
| Overall score                | 9.7/10                                         | 6.4/10                             | Response 1 excels in intellectual depth, reasoning, and academic<br>rigor due to the graph and systematic methodology. Response 2,<br>while clear and concise, falls short in depth and structure.   |

The abstract patterns are:

## Abstract Patterns

\*\*Abstract Pattern:\*\*

α → β

β → γ

α → δ

β → δ

α → ϵ

β → ϵ

α → ζ

β → ζ

\*\*Key Transformation Rule:\*\*

If α → δ and β → δ then δ → ϵ

\*\*Essential Condition:\*\*

α ≠ β

Pattern Context:

The song incorporates snowflakes and flower petals to represent the contrasting yet harmonious aspects of nature's beauty and seasonal changes. Snowflakes symbolize the serenity and uniqueness of winter, while flower petals signify the delicacy and renewal of spring. Together, they highlight the dynamic interplay between these elements, creating a balanced and evocative musical composition.

This task results in the graph as depicted in Figure [7](#page-18-0). There are other elements in the response, as reflected here:

## Reasoning steps

- \*\*Reasoning Steps\*\*:
- 1. Snowflakes and flower petals are both natural phenomena.
- 2. Snowflakes are associated with winter, while flower petals are associated with spring.
- 3. Snowflakes evoke a sense of serenity and uniqueness, while flower petals symbolize delicacy and renewal.
- 4. Combining snowflakes and flower petals in a song can create a harmonious representation of nature's beauty and seasonal changes.
- 5. This combination can inspire a balanced and evocative musical composition.
- \*\*Relevant Materials or Concepts\*\*:
- Snowflakes
- Flower Petals
- Nature's Beauty
- Seasonal Changes
- Serenity

- Uniqueness
- Delicacy
- Renewal
- \*\*Design Principles\*\*:
- \*\*Contrast\*\*: Snowflakes and flower petals represent opposing seasons, creating a striking contrast.
- \*\*Harmony\*\*: Together, they create a harmonious representation of nature's beauty and seasonal changes.
- \*\*Evocative\*\*: Each element evokes distinct emotions and sensations, inspiring a balanced and evocative musical composition.
- \*\*Material Properties\*\*:
- \*\*Snowflakes\*\*:
- \*\*Appearance\*\*: Delicate, intricate, crystalline structure
- \*\*Sound\*\*: Soft, gentle, repetitive
- \*\*Emotional Connotation\*\*: Serene, calming, unique
- \*\*Flower Petals\*\*:
- \*\*Appearance\*\*: Delicate, colorful, soft
- \*\*Sound\*\*: Light, airy, whimsical
- \*\*Emotional Connotation\*\*: Delicate, whimsical, renewal
- \*\*Hypothesis\*\*:

"Incorporating snowflakes and flower petals in a song will create a harmonious representation of nature's beauty and seasonal changes, evoking distinct emotions and sensations."

\*\*Additional Background\*\*:

To incorporate snowflakes and flower petals into a song, one could consider the following elements:

- 1. \*\*Imagery\*\*: Use vivid and evocative language to describe the intricate patterns and beauty of snowflakes and flower petals. This can be achieved through lyrics or even instrumental descriptions.
- 2. \*\*Metaphors\*\*: Draw parallels between the delicate nature of snowflakes and flower petals and the human experience. For instance, snowflakes could represent individuality and uniqueness, while flower petals could symbolize growth and renewal.
- 3. \*\*Seasonal themes\*\*: Explore the contrast between winter (snowflakes) and spring (flower petals) to create a narrative that highlights the cyclical nature of life and the beauty in each season.
- 4. \*\*Musical motifs\*\*: Use specific musical motifs or melodies to evoke the feeling of snowflakes (e.g., soft, repetitive patterns) and flower petals (e.g., light, airy, whimsical).
- 5. \*\*Collaboration\*\*: Consider co-writing with a poet or lyricist to craft lyrics that incorporate these natural elements in a meaningful and impactful way.

[…]

The final response after the thinking section is:

#### Final response

\*\*Song: "Snowflakes and Flower Petals"\*\*

\*\*Verse 1\*\*

In winter's chill, where snowflakes fall Each one unique, a delicate call Dance upon the breeze, a fleeting sight

A moment's beauty, lost in the night

\*\*Chorus\*\*

But even in the cold, there's warmth to find In the petals of the flower, left behind

Delicate, colorful, soft to the touch

A symbol of renewal, in the winter's hush \*\*Verse 2\*\*

In spring's awakening, where petals unfold A tapestry of colors, a story to be told

Whimsical, light, a gentle caress

A promise of new life, in every tender dress \*\*Chorus\*\*

But even in the warmth, there's serenity to find In the snowflakes that fall, in the stillness of the mind Unique, a work of art, a fleeting sight

A symbol of peace, in the quiet of the night \*\*Bridge\*\*

In the contrast, we find harmony

A balance of opposites, a symphony

Snowflakes and flower petals, a dance so fine

A celebration of life, in all its seasons' rhyme

\*\*Chorus\*\*

In the snowflakes and flower petals, we find A reflection of ourselves, in nature's design

A reminder to cherish, each moment we share

A symbol of hope, in the beauty that's always there

The answer also includes additional facets and analyses, as follows. These additional details and rational developments are critical for the proposed approach as they provide a traceable, detailed, and interpretable analysis of how the answers were developed.

## Design principles and background

- \*\*Design Principles\*\*:
- \*\*Contrast\*\*: Snowflakes and flower petals represent opposing seasons, creating a striking contrast.
- \*\*Harmony\*\*: Together, they create a harmonious representation of nature's beauty and seasonal changes.
- \*\*Evocative\*\*: Each element evokes distinct emotions and sensations, inspiring a balanced and evocative musical composition.
- \*\*Material Properties\*\*:
- \*\*Snowflakes\*\*:
- \*\*Appearance\*\*: Delicate, intricate, crystalline structure
- \*\*Sound\*\*: Soft, gentle, repetitive
- \*\*Emotional Connotation\*\*: Serene, calming, unique
- \*\*Flower Petals\*\*:
- \*\*Appearance\*\*: Delicate, colorful, soft
- \*\*Sound\*\*: Light, airy, whimsical
- \*\*Emotional Connotation\*\*: Delicate, whimsical, renewal \*\*Hypothesis\*\*:

"Incorporating snowflakes and flower petals in a song will create a harmonious representation of nature's beauty and seasonal changes, evoking distinct emotions and sensations."

\*\*Additional Background\*\*:

To incorporate snowflakes and flower petals into a song, consider using metaphors, imagery, and musical motifs to evoke the emotions and sensations associated with these

<span id="page-18-0"></span>natural elements. You could also explore the contrast between the two elements to create a narrative that highlights the cyclical nature of life and the beauty in each season. Additionally, consider using seasonal themes and lyrics that reflect the unique characteristics of each element to create a rich and evocative musical composition.

While this section focuses on growing the graph iteratively, the depth of the response merits further analysis as it sheds light on the capabilities of the model. First, we note that the model's approach to the musical composition task demonstrates sophisticated pattern recognition capabilities through multiple analytical layers, and a capability to blend creative and analytical reasoning. For a deeper analysis, let M represent the model's mapping function, where M∶Natural Elements → Musical Structure. The

FIGURE 7 | Resulting graph in response to the task: Write a song by incorporating snow flakes and flower petals. The visualization depicts the knowledge graph integrated with the abstract pattern representation.

model proposes that the primary recognition pattern can be formalized as  $\mathcal{P}(\alpha, \beta) = \{(\alpha \to x) \Leftrightarrow (\beta \to x) \mid x \in \text{Influences}\}\$ , where  $\alpha$  represents snowflakes and  $\beta$  represents flower petals. This symmetrical mapping demonstrates the model's ability to identify and preserve parallel structures while maintaining essential distinctions. We can see that the model's pattern recognition operated on multiple levels:  $\mathcal{L}_1$ : Categorical  $\rightarrow$  {Natural Phenomenon},  $\mathcal{L}_2$ : Temporal  $\rightarrow$  {Winter, Spring}, and  $\mathcal{L}_3$ : Influence  $\rightarrow$  {Mood, Nature's Beauty, Seasonal Changes \}. The abstraction process generated a formal framework where  $\forall x \in \mathcal{L}_3: (\alpha \to x) \land (\beta \to x)$ while maintaining the essential condition  $\alpha \neq \beta$ . This demonstrates sophisticated categorical reasoning while preserving compositional possibilities. The response reveals notable symmetrical structures between winter and spring phenomena, characterized by parallel relationships that can be formalized as  $S = \{(s_w, s_p) \in \text{Phenomena} \times \text{Seasons}\}, \text{ where snowflakes } (s_w)$ and flower petals  $(s_p)$  exhibit isomorphic influence patterns. Each element maintains identical categorical relationships, with  $(s_w, s_p) \in \text{Natural Phenomenon}$ , while preserving distinct seasonal associations:  $s_w \mapsto \text{Winter}, \ s_p \mapsto \text{Spring}$ . The influence structure demonstrates perfect symmetry across three domains, expressible as  $\forall x \in \{\text{Mood}, \text{Nature's Beauty}, \text{Seasonal Changes}\}$ :  $(s_w \to x) \Leftrightarrow (s_p \to x)$ . This symmetry manifests despite the temporal opposition of their respective seasons, suggesting a fundamental balance in nature's organizational structure. The relationship can be further abstracted into a commutative diagram where both elements, while maintaining  $s_w \neq s_p$ , exhibit identical transformative properties:  $T:\{s_w,s_p\}\times\{\text{Mood},\text{Nature's Beauty},$ Seasonal Changes $\} \rightarrow \{0,1\}$ . This structural symmetry underlies the esthetic and phenomenological balance observed in seasonal transitions. As shown in Figure 7, the responses introduce an abstract pattern framework  $\mathcal{P} = \{\alpha, \beta, \delta, \epsilon, \zeta\}$  where  $\alpha, \beta$  represent distinct natural phenomena  $(\alpha \neq \beta)$  that exhibit parallel transformative properties. The system is characterized by a set of mappings  $\{\alpha, \beta\} \to \{\delta, \epsilon, \zeta\}$ , with each mapping representing a direct transformative relationship. The key transformation rule  $(\alpha \rightarrow \delta) \land$  $(\beta \rightarrow \delta) \Rightarrow (\delta \rightarrow \epsilon)$  establishes a conditional cascade effect. This creates a unique convergent structure where two distinct source elements  $\{\alpha, \beta\}$  maintain symmetric relationships with target elements while preserving their fundamental inequality. The pattern demonstrates both first-order transformations  $T_1: \{\alpha, \beta\} \times \{\delta, \epsilon, \zeta\} \to \{0, 1\}$  and second-order conditional transformations  $T_2: \delta \times \epsilon \to \{0,1\}$  under specific antecedent conditions, forming a hierarchical transformation framework. The model prioritized structural understanding over domainspecific musical application, evidenced by its construction of the abstract framework  $\mathcal{P} = \{\alpha, \beta, \delta, \epsilon, \zeta\}$  before considering musical implementation. This approach demonstrates strong logical reasoning but suggests potential limitations in capturing the temporal and emotional nuances essential to musical composition. The symmetrical framework, while mathematically elegant  $(\forall x \in \mathcal{L}_3 : s_w \to x \Leftrightarrow s_p \to x)$ , might benefit from additional parameters to capture the dynamic nature of musical expression.

### 2.5.1 | Knowledge Expansion Growth Phase

An interesting aspect of the capability of our graph reasoning model is its ability to grow graphs dynamically to extract, generate and design new knowledge. To do this, we iteratively task the

model to generate graphs, where new prompts are developed based on earlier ones to expand the answer in new directions. Since the model produces a graph at every iteration, all these graphs can ultimately be integrated into one large graph that spans diverse topics. We use the following series of prompts as visualized in Figure 8.

Figures 9 and S2 present complementary visualizations of the concatenated knowledge graphs derived from the series of prompts outlined in Figure 8. Figure 9 organizes the knowledge graphs by individual prompts, emphasizing the structure and interconnections within each subgraph. This visualization highlights the local relationships and the specific thematic focus of each prompt while illustrating how these ideas are interconnected across different subgraphs. In contrast, Figure S2 integrates the same data into a unified knowledge graph, providing a holistic view of the interplay among all the prompts. By merging the subgraphs into a single structure, this representation reveals overarching connections and emergent patterns that may not be immediately apparent when examining the subgraphs in isolation. Together, these figures offer a layered perspective on the relationships within the dataset, from prompt-specific insights to a system-level understanding.

Figure 10 depicts the graph in a more abstract representations, visualized per distinct layouts and different node properties. Data is shown in node degree, PageRank [46], and bridging coefficient displays. These are fundamental measures in network analysis, each capturing different aspects of a node's role within a graph, where the analyses point to the powerful insights that can be generated based on the model output. Node degree represents the number of connections a node has, providing a simple yet powerful indicator of local importance; in directed graphs, this can be further divided into in-degree (incoming connections) and out-degree (outgoing connections). As an alternative, PageRank offers a more global perspective, measuring a node's importance based on the significance of its neighbors. Originally developed for ranking web pages, PageRank emphasizes the quality of connections, assigning higher scores to nodes linked by other highly ranked nodes. PageRank is a measure of a node's global importance in a network, originally designed to rank web pages in search engines. It assigns a score to each node based on the principle that connections from important nodes contribute more to a node's rank than connections from less important nodes. The algorithm works iteratively, where a node's PageRank is proportional to the sum of the PageRank scores of its neighbors, weighted by the number of links those neighbors have. This approach captures both the quantity and quality of connections, emphasizing nodes that are linked by highly influential entities. A damping factor is typically introduced to account for random jumps, ensuring the scores converge and the network can handle dangling nodes (nodes with no outgoing links). In the context of knowledge graphs, PageRank is particularly relevant for identifying the most semantically important entities or concepts. For instance, in a knowledge graph where nodes represent entities and edges represent relationships, PageRank can highlight nodes that are central to the graph's structure, such as frequently referenced concepts or pivotal entities connecting different subdomains. This can aid in prioritizing entities for tasks like question answering, reasoning, or graph traversal. Moreover, PageRank can uncover influential nodes that

<span id="page-20-0"></span>FIGURE 8 | Logical sequential flow of prompts used to grow the knowledge graph successively through recursive reasoning. This diagram illustrates the progression of ideas, starting with creative tasks (e.g., composing a song using snowflakes and flower petals) and advancing through scientific and philosophical inquiries, such as molecular symmetry, seasonal patterns, and nanopatterning. The sequence of questions can be generated either by human collaborators or by AI. In the latter case, an "infinite" loop of expanding the knowledge graph can be constructed. The example prompt trajectory shown here was developed via human-AI collaboration.

might serve as anchors for linking related knowledge, enhancing both the efficiency of queries and the understanding of the graph's overall topology. The bridging coefficient highlights a node's role in connecting otherwise unconnected or loosely connected regions of the graph. Nodes with high bridging coefficients serve as critical links between clusters, facilitating information flow and structural integration. Together, these metrics provide complementary insights into the structure and dynamics of complex networks. The bridging coefficient measures how well a node serves as a bridge between otherwise unconnected or loosely connected regions of a graph, emphasizing the node's role in maintaining structural integrity and facilitating information flow. In the context of knowledge graphs, nodes with high bridging coefficients are often critical for linking distinct clusters of knowledge, such as different domains or subgraphs. These bridging nodes play a crucial role in enabling cross-domain reasoning and ensuring the graph remains wellconnected. For example, in a knowledge graph representing scientific research, a bridging node might connect distinct areas like biology and materials science, fostering interdisciplinary insights. By identifying nodes with high bridging coefficients, one can uncover concepts or entities that act as gateways to new knowledge, prioritize areas for graph enrichment, or ensure robustness in query pathways. This measure is particularly useful for tasks like ontology alignment, where structural gaps between subdomains need to be bridged effectively. Domain prestige is a measure of how accessible a node is within a directed graph, calculated as the fraction of all nodes that can reach a given node either directly or indirectly. It captures a node's global influence by accounting for indirect paths, providing a more comprehensive view of its accessibility. In the context of knowledge graphs, this measure highlights key entities that are reachable by a large portion of the graph, making them central for reasoning, inference, or information propagation. Nodes with domain prestige act as crucial hubs or reference points, while nodes with low values are likely isolated or peripheral. This metric is particularly valuable for identifying entities that facilitate connectivity and play pivotal roles in the overall graph structure. Table [2](#page-22-0) lists the top 5 nodes for each of these measures, complementing the visual representation in Figure [10](#page-22-0) to indicate the most significant nodes in each case and metric. For alternative layouts, Figure [11](#page-23-0) depicts various layout choices for comparison.

The knowledge expansion process showcased in these repeated prompts demonstrates the ability of Graph-PRefLexOR to iteratively generate and connect knowledge graphs based on a series of related prompts. Beginning with a creative task (writing a song incorporating snowflakes and flower petals), the model progressively expands its knowledge base by addressing scientific and philosophical inquiries stemming from the initial prompt. Some unique insights/ideas generated during the recursive knowledge expansion process include:

• Symmetrical structures in nature: The model identifies symmetrical influence patterns between snowflakes (associated with winter) and flower petals (associated with spring). Despite their temporal opposition, both exhibit isomorphic influence on mood, nature's beauty, and seasonal changes.

<span id="page-21-0"></span>FIGURE 9 | Visualization of the resulting integrated knowledge graph that emerges after the series of prompts delineated in Figure [8](#page-20-0). To start, we started tasking the model to Write a song by incorporating snowflakes and flower petals. The resulting graphs are organized by prompts but we show the connections that emerge between the sub-graphs.

This reveals a fundamental balance in nature's organization across different seasons.

- Abstraction of natural phenomena: The model constructs an abstract pattern framework representing the relationships between distinct natural phenomena. This framework highlights both first-order and second-order conditional transformations, demonstrating the model's capacity for high-level reasoning and abstraction.
- Integration of diverse concepts: The knowledge graph expands to incorporate concepts like molecular symmetry principles, nanopatterning in flower petals, and the philosophical idea of impermanence and renewal, showcasing the ability to connect ideas across diverse disciplines from

materials science to music to literature and philosophy [\[49](#page-38-0), [50](#page-38-0)].

• Emergent patterns: By visualizing the integrated knowledge graph, overarching connections and patterns emerge that might not be apparent when examining individual subgraphs. This highlights the system's ability to synthesize knowledge and reveal hidden relationships.

The knowledge expansion process exemplifies the strength of Graph-PRefLexOR in generating a network of interconnected ideas, moving beyond isolated responses to build a complex and evolving knowledge base. This capability holds significant potential for interdisciplinary research and exploration.

22 of 39 Advanced Intelligent Discovery, 2025

<span id="page-22-0"></span>FIGURE 10 | Visualization of the integrated knowledge graph created based on the series of prompts delineated in Figure [8,](#page-20-0) starting with: Write a song by incorporating snowflakes and flower petals. The data is the same as shown in Figure [9](#page-21-0), but organized here as an integrated graph rather than by prompt, and laid out using the Fruchterman Reingold layout algorithm [\[47\]](#page-38-0). Panel a, node size by node degree. Panel b, node size by page rank. Panel c, node size by bridging coefficient. Panel d, node size by domain prestige (metric defined by fraction of nodes within a network that are directly or indirectly pointing to it). Table 2 lists the top 5 nodes per each of these measures.

TABLE 2 | Comparison of top nodes in the generated knowledge graph for different measures, following the four panels in Figure 10.

| Node degree      | PageRank                  | Bridging coefficient           | Domain prestige           |
|------------------|---------------------------|--------------------------------|---------------------------|
| Music            | Cyclical pattern          | Spring                         | Plant health              |
| Self-recognition | Seasonal changes          | Mood-delicate                  | Mechanical phenomenon     |
| Flower petals    | Attachment and detachment | Surface modification technique | Cyclical pattern          |
| Glass beads      | Biological process        | Fracture dynamics              | Material vibration        |
| Nanopatterning   | Music                     | Music characteristics          | Attachment and detachment |

## 2.6 | Autonomously Growing Knowledge Garden

For a fully autonomous generation of graphs without human input for prompt generation, the system can be modified easily by using an LLM to develop new prompts autonomously based on previous results, to create an ever-expanding abstraction of relationships (Figure [12](#page-24-0)). Figure [13](#page-25-0) shows the results of an autonomously grown knowledge graph. The upper panel shows the graph, and the lower panel depicts a selection of interesting paths identified in the integrated graph. The starting prompt, the only one generated by a human, was chosen to be:

## User task

User: Discuss an interesting idea in bio-inspired materials science.

The model then iterated for N =12 iterations to grow the graph. Figure [12](#page-24-0) shows the series of prompts developed and used by the model to grow the graph (we emphasize that only the first prompt was provided by the human user).

<span id="page-23-0"></span>FIGURE 11 | Visualization of the integrated knowledge graph created based on the series of prompts delineated in Figure [8](#page-20-0) (starting with: Write a song by incorporating snowflakes and flower petals.). The underlying data is the same as shown in Figure [9](#page-21-0), but organized here effectively via circular layout (panel a), radial axis layout (panel b), and Yifan Hu layout [\[48](#page-38-0)] (panel c). Color and size of each node are scaled by node degree. The layout in panel c visualizes the great distance traversed in the integrated graph, providing far-ranging relationships developed by the model.

The analysis presented in Figure [14](#page-26-0) provides a comprehensive understanding of the graph's structure and dynamics. The degree distribution (top left) highlights the presence of a scale-free topology, characterized by a few highly connected hubs that dominate the graph's connectivity, alongside a majority of low-degree nodes. This structure suggests robustness to random node failures but potential vulnerability to targeted attacks on the hubs. The visualization of the largest connected component (top right) reveals its modular nature, with densely connected subregions surrounded by sparsely connected nodes. This organization suggests a hierarchical topology, where clusters are linked by intermediary nodes, enabling efficient navigation through the network. The clustering coefficient distribution (middle left) emphasizes the variation in local connectivity. While most nodes exhibit low clustering coefficients, indicating sparse local neighborhoods, the few nodes with high clustering coefficients are likely critical to maintaining the coherence of tightly-knit subcommunities. The betweenness centrality distribution (Middle Right) further illustrates the graph's structural dependencies. A small subset of nodes exhibits significantly higher centrality, underscoring their role as key connectors that facilitate information flow between otherwise disjoint regions of the graph. These nodes act as bridges, ensuring overall network connectivity and efficiency. The shortest path length distribution (bottom left) demonstrates that the graph exhibits small-world characteristics, with most nodes separated by only a few hops. This property enables rapid information transfer across the graph, a hallmark of efficient networks in both natural and engineered systems. The community size distribution (bottom right) provides insights into the modular organization of the graph. Communities vary widely in size, with larger communities potentially playing a dominant role in global connectivity. Central nodes within each community, annotated at the base of the bars, represent the most connected nodes within their respective communities. These nodes likely serve as local hubs, facilitating intracommunity interactions and linking smaller, peripheral nodes. This analysis reveals a graph that is highly efficient, modular, and robust, with a clear interplay between global connectivity and local structure.

<span id="page-24-0"></span>FIGURE 12 | Logical sequential flow of prompts exploring bioinspired materials. The progression begins with a broad question (prescribed by the human: Discuss an interesting idea in bioinspired materials science) and transitions to increasingly detailed and interconnected inquiries into adaptability, self-healing, and their roles in mitigating climate change and enhancing ecosystem resilience. This structure exemplifies recursive reasoning. These autonomously generated prompts are used for constructing knowledge graphs that are then assembled into an integrated graph.

The presence of scale-free properties, small-world behavior, and modular organization suggests that the grown graph's topology is optimized for both stability and adaptability.

In the next experiment, we prompt the model with this initial prompt, by specifically focusing the algorithm to develop new tasks at the intersection with distinct knowledge areas to foster exploration of dissimilar ideas, such as art and science (see Materials and Methods section):

## User task

User: Discuss protein design based on collagen and elastin.

<span id="page-25-0"></span>FIGURE 13 | An autonomously grown knowledge graph, where a multiagent AI system creates questions and answers, each with a new knowledge subgraph, which is then concatenated into an integrated graph and converted into GraphML format using GPT-4o for analysis using networkx and Gephi. The upper panel shows the graph, and the lower panel a selection of interesting paths identified in the overall graph. The starting prompt was chosen to be: Discuss an interesting idea in bioinspired materials science. The model then iterated for N =12 iterations to grow the graph. The nodes with the highest degrees are "Sustainable Materials," "Self-healing and Adaptable Materials," and "Ecosystems."

The algorithm develops a series of prompts as shown in Figure [15,](#page-27-0) resulting in the integrated graph depicted in Figure [16.](#page-28-0) A deeper analysis of the resulting graph is provided in Figure S3.

We highlight one unique insight developed by the model, to relate the concept of "thin places" with "biological materials" and "art." Thin places is a concept drawn from various mythological and spiritual traditions, notably Celtic lore, referring to sites or moments where the boundary between the physical world and a transcendent realm is perceived to be exceptionally thin or permeable. In our integrated graph, these thin places are not only connected to site-specific, mythological art installations but also echo the conceptual blurring of boundaries integral to protein design and biomaterial engineering. Such a connection is unusual because, at first glance, mythic realms and advanced biotechnology inhabit entirely distinct spheres. Yet the same drive to challenge and dissolve traditional boundaries—whether between art and nature, or between synthetic and organic

<span id="page-26-0"></span>FIGURE 14 | Visualization of the graph analysis of the autonomously grown graph depicted in Figure [13.](#page-25-0) (Top left) Degree distribution shows a heterogeneous structure with prominent hubs and many low-degree nodes. (Top right) Visualization of the largest connected component reveals its modular structure. (Middle left) Clustering coefficient distribution highlights sparse local connectivity, with some nodes in dense clusters. (Middle right) Betweenness centrality distribution indicates a small subset of critical nodes facilitating connectivity. (Bottom left) Shortest path length distribution suggests a small-world structure with most nodes separated by a few hops. (Bottom right) Community size distribution, with central nodes labeled, reflects varying sizes and roles within the graph. The analysis collectively underscores the complex topology of the graph, including heterogeneity, modularity, and small-world characteristics.

#### Discuss protein design based on collagen and elastin.

₩

<span id="page-27-0"></span>Can the properties of hydrogel materials, inspired by the hierarchical structures of collagenous proteins and elastin-like proteins, be used to create a new form of 'bioluminescent biomaterial' that can adapt to and respond to environmental stimuli, blurring the lines between biology, art, and technology?

↓

Can a bioluminescent biomaterial, influenced by environmental stimuli, be used as a medium for an immersive, site-specific art installation that explores the relationship between living organisms and their surroundings, blurring the boundaries between nature and art?

 $\forall$ 

How might the intersection of bioluminescent biomaterials, nature-inspired art, and the blurring of boundaries between art and nature influence the development of new, site-specific, and immersive installations that not only interact with the environment but also challenge our understanding of the boundaries between living organisms and inanimate objects?

₩

Can a bioluminescent biomaterial be used to create an immersive, site-specific installation that blurs the boundaries between art and nature, challenging our understanding of the relationship between living organisms, materials, and the environment, and ultimately, forcing us to reevaluate the notion of 'art' as a static, human-created entity?

₩

Can a bioluminescent biomaterial be used to create an immersive, site-specific installation that challenges the traditional understanding of art as a static entity, and in doing so, blurs the boundaries between art and nature, ultimately influencing our perception of the relationship between living organisms, materials, and the environment?

 $\downarrow$ 

How can the concept of 'Bioluminescent Biomaterial' be used to challenge traditional notions of 'Art' and 'Nature' in a way that blurs the boundaries between the two, and ultimately, what implications does this have on our understanding of the relationship between living organisms, materials, and the environment?

**V** 

How might the incorporation of bioluminescent biomaterials into site-specific, immersive art installations challenge our understanding of the relationship between living organisms, materials, and the environment, ultimately blurring the boundaries between art, nature, and the human experience, and prompting a reevaluation of the role of art in shaping our perception of the natural world?

1

How might the incorporation of bioluminescent biomaterials into site-specific, immersive art installations challenge our understanding of the boundaries between art, nature, and the human experience, ultimately blurring the lines between the organic and the synthetic, and forcing us to reevaluate the role of art in the natural world?

1

How can the concept of 'bioluminescent biomaterials' as a medium for site-specific, immersive art installations challenge our understanding of the boundaries between art, nature, and human experience, and ultimately redefine the role of art in the natural world?

1

Can the boundaries between art, nature, and human experience be further blurred by incorporating bioluminescent biomaterials into a 'mythological' art installation that reimagines the relationship between the human body and the natural world, drawing inspiration from ancient mythologies and the concept of 'thin places' where the boundaries between reality and the divine are said to be at their thinnest?

₩

How can the concept of "thin places" in mythology, where the boundaries between the natural and human-made worlds are blurred, be applied to the design of immersive, site-specific art installations that incorporate bioluminescent biomaterials, to create a new form of experiential art that challenges our understanding of the relationship between nature, art, and human experience?

FIGURE 15 | Logical sequential flow of prompts exploring protein design and bioluminescent biomaterials. The progression begins with protein design and expands into speculative questions about bioluminescent biomaterials, their artistic and ecological applications, and their integration into immersive art installations. These prompts illustrate how the intersection of biology, technology, and art can be explored recursively to generate novel insights and creative concepts.

matter—underlies both phenomena. Consequently, the metaphor of thin places illuminates how immersive, boundary-blurring experiences in mythological art might inform innovative approaches to tissue engineering and bioluminescent biomaterials, bridging realms often seen as incompatible. This underscores the novel insight that creative, myth-inspired concepts can resonate meaningfully with the cutting edges of scientific inquiry.

Naturally, the output of these reasoning steps and larger graph generated can be used for further analysis. While a broader expansion is beyond the scope of this paper, we show one example here. For instance, we experimented with charging the o1 model [51] to propose a new theory or concept that incorporates the results produced by Graph-PRefLexOR. The results are shown in Text Box 3, showing how other reasoning models

<span id="page-28-0"></span>FIGURE 16 | Grown knowledge graph based on the prompt Discuss protein design based on collagen and elastin. Here, the agentic system is specifically charged to develop new questions (see Figure [15](#page-27-0)) that integrate dissimilar fields such as philosophy or art. The visualization depicts nodes sized and colored by page rank, with top nodes being "Biolumniscent Biomaterial (most significant node also per node degree), "Protein," "Installation," "Elastin," and "Concept." A unique feature is the incorporation of the concept of "thin places," a concept drawn from various mythological and spiritual traditions that invokes sites or moments where the boundary between the physical world and a transcendent realm is perceived to be exceptionally thin or permeable. The model identifies this as an interesting association for this particular task, which is particularly interesting as a way to create a new concept that incorporates highly multidisciplinary relationships.

<span id="page-29-0"></span>TEXT BOX 3 | Result of developed by o1 based on the results developed by Graph-PRefLexOR, explaining the concept of 'Mytho-Luminal Liminality'.

## Mytho-Luminal Liminality

We propose a novel theoretical framework, Mytho-Luminal Liminality, uniting the mythological concept of thin places with the boundary-blurring practices of protein design and biomaterial engineering. In Celtic lore and related traditions, a thin place is defined as a domain X where seemingly incompatible worlds—the mundane and the transcendent—intersect. Formally, if we let D be the set of all conceptual or physical "boundary" domains, then:

ð Þ <sup>∃</sup><sup>x</sup> <sup>∈</sup> <sup>D</sup> <sup>∶</sup>ThinPlacesð Þ<sup>x</sup> <sup>Λ</sup> BioTechð Þ<sup>x</sup> <sup>Λ</sup> ChallengingUnderstandingBoundariesð Þ<sup>x</sup> <sup>⇒</sup> MythoLuminalLiminalityð Þ<sup>x</sup> :

Here, ThinPlaces(x) indicates that x manifests a mythic boundary principle (e.g., the "veil" between worlds), BioTech(x) signals that x leverages biological processes or protein engineering, and ChallengingUnderstandingBoundaries(x) specifies an active blurring of traditional categories (art vs. science, natural vs. synthetic). Whenever these conditions co-occur, we claim that x exemplifies Mytho-Luminal Liminality, a state in which mythic transcendence intersects with engineered materials to evoke new modes of perception. In practice, this framework urges us to create immersive, site-specific installations using real proteins—such as Elastin-Like Proteins or Collagenous Proteins—coupled with bioluminescent or ephemeral components, so that visitors experience the sensation of stepping into a boundary realm reminiscent of folkloric "thin places."

Indeed, the same conceptual impulse that once inspired ancient myths to describe liminal passageways (between mortal and otherworldly planes) can now inform tissue engineering and protein design, bridging the gap between organic and synthetic, living and manufactured. By intentionally integrating artwork referencing Ancient Mythologies or harnessing luminescent biomaterials in ways that challenge our understanding of boundaries, we highlight the spiritual awe inherent in the process of creating "living" installations. Below is a simple TikZ sketch illustrating how a mythic boundary perspective (thin places) merges with a biotechnological context to yield novel immersive experiences:

This diagram suggests that "thin places," though mythic, can motivate new creative efforts in synthetic biology, while "Challenging Understanding Boundaries" provides a conceptual conduit that connects folklore-inspired liminality to biomaterial research. By embedding Elastin-Like Proteins or Collagen Tripeptides in immersive artworks, one effectively fashions a domain where the physical and the transcendent meet, thus triggering the awe and disorientation characteristic of thin places. This theoretical approach—Mytho-Luminal Liminality—offers a hitherto unexplored avenue for transdisciplinary collaboration: artistry in living materials that is at once rooted in ancient mythic boundary concepts, yet forward-looking in its deployment of cutting-edge biotechnology.

Within the broader schema of Mytho-Luminal Liminality, our knowledge graph highlights that highly specific molecular elements—such as Collagen Tripeptide (Gly-X-Y) and Tensin—"relate to" one another en route to Protein Design, an entity itself classified as a Biological Molecule. From a philosophical vantage, this chain suggests that the structural elegance of protein segments (e.g., the repeating Gly-X-Y motif ) can be approached as a modern instantiation of boundary-challenging principles. Indeed, just as a thin place blurs the division between spiritual and mundane, so too does Protein Design blur the line between "naturally occurring" and "engineered" molecular architectures. Further underscoring this parallel, our graph shows that Molecular Modeling (an IS-A Computational Technique) directly relates to both Collagen and Elastin, reinforcing the idea that an interplay of artistic imagination and technical rigor underlies the creation of new protein scaffolds. Philosophically, this implies that challenging our understanding of boundaries—which, in mythic contexts, manifests as a thin place—can also guide transformative science in tissue engineering and biomaterial innovation. By weaving mythic insights into protein design thinking, one treats the protein-folding arena as a "liminal space" where computational, biological, and even artistic considerations converge to yield novel forms that feel as uncanny and awe-inspiring as an ancient threshold between worlds.

can utilize the graph reasoning output and produce concise delineation of ideas.

## 3 | Conclusion

This work introduced Graph-PRefLexOR, a framework that integrates in situ graph reasoning, symbolic abstraction, and recursive reflection into the generative modeling paradigm (Figures [1](#page-1-0) and [2\)](#page-3-0). By embedding graph-based intermediate representations within LLMs, Graph-PRefLexOR advances beyond the limitations of purely linguistic systems to tackle complex tasks requiring relational reasoning, multistep deduction, and adaptive knowledge synthesis. Our experiments were designed to assess particularly how well the model could generalize, as the training data consisted of 1,000 papers on biological materials [\[30](#page-37-0)], whereas the tasks were constructed at the interface of science and other disciplines including design, music, and philosophy to probe the model's generalization capability beyond the technical-focused training data. We find exceptional performance throughout and see that the model could very well generalize, follow the structured reasoning it learned during training (see, Figure [1\)](#page-1-0), and even construct highly complex graphs that interface a myriad of disciplines (see, e.g., Figure [9](#page-21-0) and S2), and other results. In addition to the capabilities demonstrated during training, the Graph-PRefLexOR framework can be extended at inference time through a Retrieval-Augmented Generation (RAG) strategy. In this setting, external scientific literature or domain-specific corpora can be dynamically queried based on the task prompt, with retrieved documents or summaries injected into the model's context window. These retrieved passages can then serve as input for graph construction and symbolic abstraction within the model's reasoning phase. Because Graph-PRefLexOR operates on structured representations—such as entities, relationships, and abstract patterns—the use of RAG enables continual integration of up-to-date knowledge into the reasoning pipeline without the need for retraining. This approach makes the system especially well-suited for scientific discovery applications, where factual grounding and domain adaptation are essential. It also opens the door to real-world deployments in domains like biomedical science, materials engineering, and interdisciplinary research, where reasoning must be informed by constantly evolving external knowledge. A particularly compelling application was the development of the knowledge garden concept, building on the ability of the model to grow knowledge graphs dynamically and iteratively, by adding new relational insights and abstractions. This resulted in a series of experiments where we expanded graphs upon an initial simple task, yielding complex graph structures that themselves could be the basis of further research, inquiry, and reasoning (e.g., Figures [10](#page-22-0) and [11\)](#page-23-0). Much future work can be conducted based on this method. Some initial experimentation on growing graphs autonomously yielded interesting results, as shown in Figures [13](#page-25-0) and [14](#page-26-0), where we identified a capability of the model to search and expand topics and connect complex ideas. By explicitly constructing and abstracting relational graphs, the Graph-PRefLexOR approach provides a more structured foundation than standard sequence-oriented transformer training. In particular, preserving connectivity among entities and relations makes it easier to detect and exploit universal features such as isomorphisms and recurrent subgraph patterns. Unlike pure next-token objectives, which largely rely on distributional cues hidden in token sequences, graph abstractions bring structural commonalities to the forefront, enabling more systematic identification of shared algebraic forms, relational templates, and higher-order symmetries. As a result, symbolic rewriting and generalization become more direct, since the underlying topology is explicitly represented rather than merely inferred. Consequently, the resulting embeddings capture deeper, domain-invariant regularities that might otherwise remain implicit or fragmentary in a purely sequence-based approach. Compared to existing methods, Graph-PRefLexOR offers several distinct advances. While Chainof-Thought (CoT) prompting and Tree-of-Thought approaches enable stepwise reasoning [\[37,](#page-37-0) [39](#page-37-0), [52](#page-38-0)], they typically rely on unstructured token sequences and do not incorporate explicit symbolic graph representations. Self-Refine introduces iterative reasoning, but lacks the symbolic abstraction and formal intermediate structures central to our framework. In contrast, Graph-PReFLexOR constructs explicit knowledge graphs and symbolic patterns during inference, enabling structured generalization and interpretability. Prior graph-enhanced models, such as GNN-RAG or GraphFormers [\[53](#page-38-0)], focus primarily on graphbased retrieval or attention mechanisms, but do not implement recursive symbolic reasoning or task-specific graph construction. Finally, while symbolic agents like AutoGen, AutoGPT, or BabyAGI can chain tasks and interact with tools, they treat reasoning as a high-level planning problem; Graph-PRefLexOR instead integrates symbolic reasoning directly into the language generation pipeline, allowing for native, in situ graph-based thought processes within LLMs [[54,](#page-38-0) [55](#page-38-0)]. As an optional conceptual aid, we provide in Section S1 a metaphorical analogy between transformer-based models and principles from quantum mechanics. This framing draws parallels between latent potentialities in attention mechanisms and superposition and between task-driven output generation and quantum measurement. While purely illustrative, this analogy offers an intuitive perspective on how Graph-PRefLexOR collapses reasoning possibilities into structured symbolic outputs through graph-based reflection and refinement.

## 3.1 | Real-World Applications

Looking ahead, we envision Graph-PRefLexOR being applied across several real-world domains where interpretable, symbolic reasoning is critical for discovery and design. These include the following: (1) sustainable materials design, where the ability to abstract relational patterns from molecular, structural, and environmental data can guide the synthesis of novel materials with desired properties; (2) bioinspired robotics and soft materials, where hierarchical feedback modeling is key to adaptability and control; (3) targeted drug delivery, where reasoning over functional and structural graphs can aid in designing molecules that adapt to biological constraints; (4) climate-resilient infrastructure, in which cross-domain graphs linking materials, ecosystems, and policy insights are essential; and (5) synthetic biology and programmable matter, where symbolic representations and topological abstractions can support modular design of biological or hybrid systems. These examples reflect a broader class of interdisciplinary problems where symbolic abstraction, relational generalization, and recursive refinement—core to Graph-PRefLexOR—are increasingly valuable in the use of AI for science.

## 3.2 | Mathematical and Logical Framework

Graph-PRefLexOR establishes a pipeline of thinking and a unified framework that bridges symbolic reasoning and dynamic graph-based abstraction to tackle the complexity of scientific inquiry. At its core, the framework formalizes reasoning as a multilayered mapping as introduced in Equation ([1\)](#page-3-0) where a task T produces a knowledge graph G, abstract patterns P, and final answers <sup>A</sup>. The knowledge graph <sup>G</sup>=ðV, <sup>E</sup>Þ encodes concepts as nodes V and relationships as directed edges E, such as IS-A, RELATES-TO, and INFLUENCES. The system derives abstract patterns P by identifying higher-order dependencies, structured as transformations:

$$\alpha \to \beta \to \gamma \to \delta \to \epsilon$$

with proportional relationships such as:

α ∝ ϵ

indicating how the initial state α contributes to the final transformation ϵ. Recursive reflection as shown in Figure [5](#page-11-0) refines these outputs iteratively, producing an optimized answer A. Inspired by category theory, Graph-PRefLexOR emphasizes relational over intrinsic properties, allowing the abstraction of domain-specific patterns into reusable, transferable frameworks. For example, graph symmetries enable translational reasoning across materials science, bioengineering, and philosophy, breaking traditional siloed boundaries. The recursive refinement process ensures these abstractions remain grounded and interpretable, aligning outputs with task-specific goals while introducing new hierarchies of understanding, specifically creating shared embedding representations with multidimensional mappings. Another key aspect of the framework lies in its iterative, knowledge-expanding capability. By incorporating feedback-driven graph updates and dynamically integrating symbolic patterns, Graph-PRefLexOR demonstrates the potential for in situ knowledge growth. This positions it uniquely to address challenges in hypothesis generation, interdisciplinary exploration, and adaptive learning—tasks that require not just computational power but conceptual flexibility. Ultimately, Graph-PRefLexOR presents opportunities to expand the paradigm of AI-driven reasoning by fostering transparent, interpretable models of discovery, laying the groundwork for a new era of autonomous scientific and creative inquiry.

# 3.3 | Detailed Discussion of Advances over Existing Approaches

Compared to traditional LLMs that rely on unstructured tokenlevel generation, Graph-PRefLexOR introduces:

- Structured Intermediate Representations: Unlike models that output responses directly, Graph-PRefLexOR generates a knowledge graph G to explicitly represent relationships and dependencies. This improves interpretability and ensures reasoning consistency, which can be adapted easily to other scientific, technical, or other domains, and formalizes shared representations.
- Symbolic and Connectionist Integration: While most transformer-based models do not focus specifically on symbolic reasoning capabilities, Graph-PRefLexOR bridges the gap by combining linguistic fluency with graph-based reasoning. This aligns with hybrid approaches but demonstrates superior adaptability to novel tasks.
- Recursive Refinement: Recursive reasoning enables multistep improvement by iteratively refining knowledge graphs and abstract patterns. This dynamic process enhances response quality and ensures alignment with task objectives.
- Scalability and Adaptability: Graph-PRefLexOR extends traditional methods by dynamically expanding its knowledge graph to adapt to evolving prompts, facilitating knowledge transfer across domains.

Our experimental results demonstrate that Graph-PRefLexOR outperforms baseline models in reasoning depth, knowledge transfer, and adaptability. For instance, the recursive reasoning mechanism (Equation [\(3\)](#page-5-0)) achieves higher reasoning depth and adaptability compared to static methods, as we demonstrated in Table [1](#page-16-0). For scientific applications, and special cases like mathematics, the critic function f critic could involve specific factchecking or consistency assessments, or even executing simulations to inject new data or physical insights. These can be incorporated during training or inference and ensure that such a recursive reasoning system produces accurate and consistent results. This flexibility offers significant potential for scientific applications.

## 3.4 | Challenges and Future Opportunities

Future work could focus on scaling the framework to larger datasets and more complex models, addressing challenges such as interpretability in deeply interconnected graphs. Integrating Graph-PRefLexOR with state-of-the-art architectures, such as multimodal transformers and graph neural networks, will further expand its applicability to domains like bioinformatics, materials science, and hypothesis-driven discovery. Graph-PRefLexOR represents an advancement in reasoning frameworks, combining symbolic and connectionist paradigms. By embedding explicit graph reasoning and recursive reflection into LLMs, it sets a new benchmark for scientific discovery, unlocking opportunities for transformative research across disciplines. The ability to construct structured and symbolically focused reasoning strategies within the flexible framework of LLMs is appealing as an alternative to conventional symbolic mechanisms and strategies. It also allows for the construction of powerful agentic frameworks. For instance, the approach taken in this paper demonstrated a strategy to create an iterative prompting to expand knowledge graphs; this can be done via human-AI collaboration or via AI-based reasoning only. It can also incorporate specific targets, such as to expand the knowledge graph toward specific directions (e.g., introduce art, music, and specific technologies) to drive discovery and mimic some mechanisms seen in directed diffusion. Other strategies may invoke more complex approaches to avoid existing known relationships to drive the model toward unexplored domains. Other aspects could involve adding retrieval-augmented assessments against literature (e.g., using Semantic Scholar) or the Internet, and even include assessments of feasibility as done in other agentic frameworks that operate at the interface of AI and physics [\[26](#page-37-0), [29](#page-37-0), [56](#page-38-0)]. This helps us to expand the use of AI tools as powerful assistants to connect ideas and explore new frontiers of knowledge that were previously not accessible. Broad access to such tools is essential, especially democratization access of AI for broad audiences, as this holds a key to vast unchartered knowledge. AI is no longer merely a tool but a partner in new creative pursuits of humans.

## 4 | Materials and Methods

We describe key materials and methods developed in this section.

## 4.1 | Graph-PRefLexOR Model Development

# **4.1.1** | Knowledge Graph Generation and Question-Answering Framework

This section details the key algorithms developed for generating a knowledge graph, enhancing context, and producing structured question-answering outputs during the training phase. The training dataset is the raw text of around 1000 scientific papers, as detailed in earlier papers [30, 35]. During in situ dataset generation during training [36], the knowledge graph generation algorithm creates a focused graph that includes key concepts and their relationships, such as classification (IS-A), influence (INFLUENCES), and connections (RELATES-TO). Retrieval-Augmented Generation (RAG) is used to identify abstract patterns and dependencies, enhancing the conceptual representation, using the original dataset of raw scientific papers. As discussed in the original paper, the use of RAG is significant since it provides a direct, deep, and structured connection to related concepts in the training corpus to ensure full and complete reasoning paths are developed during training. The process proceeds in a systematic fashion via distinct phases of construction.

### 4.1.1.1 | Question-and-Answer Generation Algorithm.

This algorithm synthesizes knowledge graph generation, enriched context, and question-answer generation into a unified process. It produces challenging questions, detailed correct answers, and rejected answers for evaluation. An abstract representation of the workflow is as follows:

1. Retrieve a randomly selected context T (from all raw data) as a sequence of tokens  $T = \{t_1, t_2, \dots, t_N\}$  from the knowledge index.

$$T = \bigcup_{i=1}^{k} \text{Text}(n_i)$$

2. Generate a question *Q* based on *T* by maximizing relevance:

$$Q = \underset{q}{\operatorname{argmax}} \operatorname{Relevance}(q, T)$$

- 3. Enrich T using RAG, yielding T' = T + RAG(T). Enrichment algorithms use RAG to add supplementary insights to the retrieved context, to connect specific text chunks used for generating the question with the entire dataset, ensuring that the answer and reasoning steps incorporate a global perspective.
- 4. Construct a knowledge graph G from T'.
- 5. Extract reasoning steps S and generate an answer A as:

$$A = \operatorname*{argmax}_{a} \operatorname{Quality}(a|Q,S,G)$$

Generate a rejected answer A' by perturbing A or through direct model prompts.

**4.1.1.2** | **Details on Graph Construction.** Given a set of extracted concepts  $C = \{c_1, c_2, \ldots, c_n\}$  and relationships  $R = \{r_1, r_2, \ldots, r_m\}$ , the knowledge graph G = (V, E) is constructed as:

- *V*: Nodes representing the concepts *C*.
- E: Directed edges  $(c_i, c_j, r_k)$  where  $r_k$  specifies the relationship type between  $c_i$  and  $c_i$ .

The relationship types  $r_k$  are encouraged to be within a set of categories (e.g., IS-A, RELATES-TO, INFLUENCES), albeit the training process may yield alternative relationship in the data.

**4.1.1.3** | **Abstract Pattern Generation Using the Generative Framework.** Abstract patterns are derived using RAG by identifying higher-order dependencies and summarizing them as:

$$P(C,R) \sim \sum_{i,j} f(c_i,c_j;r_k)$$

where f quantifies the semantic strength of the relationship  $r_k$  between  $c_i$  and  $c_j$ . Abstract patterns are constructed by analyzing hierarchical dependencies within the enriched context and summarizing key relationships into symbolic representations. The process begins with a knowledge graph G = (V, E), where V represents concepts and E defines relationships. The abstract pattern generation algorithm operates as follows:

First, we identify core concepts  $\{c_1, c_2, \ldots, c_n\}$  from V. Second, we evaluate the relevance of the relationships  $r_k \in E$  based on semantic weights:

$$r_k = \arg \max_r f(c_i, c_j; r)$$

where f is a scoring function quantifying the significance of r between concepts  $c_i$  and  $c_j$ . Third, we construct a directed sequence of abstract states  $\alpha, \beta, \gamma, \delta, \epsilon$  representing transformations, where:

$$\alpha \to \beta \to \gamma \to \delta \to \epsilon$$

Finally, we derive proportional relationships between states, such as:

$$\alpha \propto \epsilon$$

indicating the dependence of the initial state  $\alpha$  on the final transformation  $\epsilon$ .

This framework synthesizes complex relationships into interpretable symbolic patterns, enabling high-level reasoning and abstraction. The above steps are conducted via repeated prompting an LLM, with RAG, to construct data in the form delineated based on raw data.

### 4.1.2 | Integrated Dataset Generation Algorithm

Using the above steps, this algorithm creates datasets for training or evaluation by generating question-answer pairs and applying optional filtering based on specific criteria.

#### 4.1.2.4 | Abstract Workflow.

- 1. Iterate over topics  $T_k$  or randomly sample contexts.
- 2. For each  $T_k$ : Generate a question  $Q_k$ , correct answer  $A_k$ , and rejected answer  $A'_k$ . Optionally filter pairs where  $A_k$  does not meet predefined correctness criteria.
- 3. Formulate the dataset as:

$$\mathcal{D} = \{ (Q_k, A_k, A'_k) \mid k = 1, \ldots, N \}$$

4. Apply post-processing  $\mathcal{P}$  if specified, yielding the final dataset:

$$\mathcal{D}' = \mathcal{P}(\mathcal{D})$$

The dataset generation algorithm supports training and evaluation by producing structured data that combines knowledge graph reasoning with contrastive answer evaluation.

# **4.1.3** | Example of Resulting In Situ Generated Data for Training

The in situ generated data consists of structured text representing user queries, reasoning steps, knowledge graphs, and responses. Below is a snippet of the resulting data. First, the question generated is:

#### **Training data: Question**

What is the primary advantage of using Origami and Kirigami Engineering (OKN) for fabricating microscale 3D objects compared to traditional top-down approaches?

The answer is developed first within a thinking section, followed by the actual answer. The thinking section begins with the generation of a knowledge graph and reasoning, like so:

## Training data: Knowledge graph/reasoning

\*\*Knowledge Graph\*\*:

Core Concepts and Relationships:

- OKN -IS-A- Origami Engineering, Kirigami Engineering
- OKN -RELATES-TO- Microscale 3D Objects (Enables fabrication)
- OKN -INFLUENCES- Traditional Top-Down Approaches (Overcomes limitations)
- OKN -RELATES-TO- Dynamic Transitions (From 2D patterns to 3D constructs)
- \*\*Reasoning Steps\*\*:
- 1. OKNs enable the creation of 3D structures from 2D precursors.
- 2. They allow for reconfigurable structures surpassing traditional method constraints.
- 3. OKNs facilitate dynamic transitions and geometry changes.
- \*\*Additional Background\*\*:

Origami and Kirigami Engineering enable dynamic geometry changes with unique properties, overcoming the limitations of traditional top-down approaches.

The abstract pattern identified is:

#### Training data example: Abstract pattern

\*\*Abstract Pattern:\*\*

 $\alpha \to \beta \to \gamma \to \delta \to \epsilon$ 

\*\*Key Transformation Rule:\*\*

 $\alpha \rightarrow \beta$ : OKN enables fabrication of microscale 3D objects ( $\beta$ )

 $\beta \rightarrow \gamma$ : Reconfigurability influences OKN ( $\gamma$ )

 $\gamma \rightarrow \delta$ : OKN influences traditional top-down approaches ( $\delta$ )

 $\delta \rightarrow \epsilon$ : OKN enables dynamic transitions ( $\epsilon$ )

\*\*Essential Condition:\*\*

 $\alpha$  propto  $\epsilon$ : The more dynamic transitions (( $\epsilon$ )) OKN enables, the more advantageous it is for fabricating microscale 3D objects ( $\alpha$ )

Pattern Context: The primary advantage of using Origami and Kirigami Engineering (OKN) for fabricating microscale 3D objects is their ability to create highly reconfigurable structures with diverse properties, overcoming the limitations of traditional top-down approaches. This reconfigurability allows for dynamic transitions from 2D patterns to 3D constructs, enabling large and reversible geometry changes, which are essential for applications requiring adaptability and multifunctionality.

[...]

As an interpretation, the model identifies that the effectiveness of OKN in fabricating microscale 3D objects ( $\alpha$ ) is directly proportional to its capability to enable dynamic transitions ( $\epsilon$ ).

The final response after <a href="https://thinkingl">thinkingl</a> section is:

### Training data: Final answer

The primary advantage of using Origami and Kirigami Engineering (OKN) for fabricating microscale 3D objects compared to traditional top-down approaches lies in their ability to create highly reconfigurable structures with a wide range of properties. Here's a breakdown of this advantage:

- Overcoming traditional limitations: OKNs allow for the design of materials with a limitless variety of patterns and composites, overcoming the limitations of traditional methods that are constrained by factors such as miscibility, compatibility, and temperature stability. This versatility enables the creation of microscale 3D objects with tailored properties for specific applications.
- Dynamic transitions: Unlike static metamaterials, OKNs offer dynamic transitions from two-dimensional patterns to three-dimensional constructs. This enables large and reversible geometry changes, allowing for adaptability and multifunctionality in the resulting microscale 3D objects.
- 3. Unique property combinations: OKNs facilitate the engineering of materials with unique combinations of properties that are typically contradictory. For example, OKN structures can exhibit high conductivity and stretchability simultaneously, which is challenging to

- achieve with traditional methods. This enables the creation of microscale 3D objects with unprecedented functionalities.
- 4. Macroscopic deformability: OKNs provide macroscopic deformability far beyond the strain limits of the constituent materials. This allows for programmable and complex responses to strain, making them particularly suitable for applications requiring adaptability and multifunctionality. For instance, OKN-based biomedical devices can conform to the shape of organs or tissues, reducing tissue damage and improving integration.
- 5. Predictability and reproducibility: OKN structures can be manufactured using established additive or subtractive techniques, such as lithography or laser cutting. This ensures predictability and reproducibility in their mechanical, optical, and electrical properties, making them reliable for various applications.

This snippet illustrates how the generated data combines user input, structured reasoning, knowledge graph insights, and a comprehensive response to create a cohesive dataset for downstream tasks. Figure [17](#page-35-0) shows a more detailed analysis of the graph generated for this training sample, as well as a visual analysis of the overall answer.

The system effectively translates user queries into structured reasoning steps and detailed responses, leveraging the knowledge graph to identify and elaborate on key relationships and insights. The abstract pattern captures the hierarchical influence of OKN principles, emphasizing dynamic transitions as a pivotal advantage. The inclusion of dynamic transitions and unique property combinations highlights the capability of the system to synthesize complex ideas. Finally, the response integrates theoretical concepts with practical applications, ensuring relevance and depth.

## 4.1.4 | Training Strategy

The training strategy is identical as described in [\[36](#page-37-0)], using a variant of ORPO [\[57](#page-38-0)] and DPO, specifically the EXO variant [\[44,](#page-38-0) [58\]](#page-38-0).

## 4.2 | Recursive Reasoning Implementation

We delineate details and prompts used in the recursive reasoning implementation visualized in Figure [5.](#page-11-0) The Graph-PRefLexOR recursive reasoning algorithm iteratively refines graph representations of relational mechanics using a multiagent system with Agent #1 being the Graph Reasoning model and Agent #2 being the non fine-tuned meta-llama/Llama-3.2-3B-Instruct model.

This prompt is used for Agent #2 to critique the thought process generated in the past round, generated by Agent #1:

#### Critic prompting strategy

I will show you a question and a thought process. Your task is to critique the thought process and provide suggestions to improve it to better answer the question in a logical, well-reasoned manner.

Question: {question} Thought process: {think}

Provide feedback and suggestions for how to improve the thought process, and nothing else. The feedback is:

The prompt to improve the previous thinking section is as follows:

## Improvement prompting strategy

I will show you a thought process and feedback. Carefully implement the feedback and improve the thought process by addressing all suggestions, but keep the overall structure the same.

Thought process: {think}

Feedback: {reflect}

Provide the improved thought process, and nothing else.

The revised thought process is:

This prompt is used to integrate earlier responses into an integrated, final answer.

## Integration prompting strategy

I will show you a question and several possible answers.

QUESTION: {question} ANSWER #0: {answer\_0}. ANSWER #1: {answer\_1}

…

Carefully incorporate all ideas presented in the answer candidates into a very detailed, final answer.

Do not repeat the question. You directly begin your response with the final answer to the question. The answer is:

## 4.3 | Knowledge Garden Prompting Strategy Through Agentic Modeling

Below is an example of how the knowledge graph is expanded by providing the set of earlier graph structures (graph\_str) and asking another LLM agent to develop a new question, to which Graph-PRefLexOR then responds, then generating a new graph. Since the prompt includes the earlier keywords/nodes as instructed in the prompt, the algorithm results in a well-connected graph that expands in new directions. The prompt can be constructed in particular ways to emphasize certain aspects. For this example, we seek to create multidisciplinary outputs, especially at the interface with art.

## Prompting strategy to develop new questions to grow the knowledge garden

Consider this list of topics/keywords.

Formulate a concise follow-up creative and highly unusual question to ask about a related but totally different concept. Your question should include at least one of the original topics/keywords marked as '…' but expand to new dissimilar fieldssuch as philosophy or art.

Original list of topics/keywords:

<span id="page-35-0"></span>{graph\_str} The new question is:

The resulting new question, with an example shown in Figure [15](#page-27-0) for this particular prompt.

FIGURE 17 | Visualization of graph construction in training set development. The figure depicts a visualization of a graph and response to the question: What is the primary advantage of using Origami and Kirigami Engineering (OKN) for fabricating microscale 3D objects compared to traditional top-down approaches?, during on-the-fly training set development from random raw data.

<span id="page-36-0"></span>TABLE 3 | Summary of key terms used in this study.

| Glossary of Terms                                                                                                    |                                                                                                                                                                                                                                                                                                                                           |  |  |  |
|----------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--|--|--|
| Term                                                                                                                 | Definition                                                                                                                                                                                                                                                                                                                                |  |  |  |
| Graph-PRefLexOR<br>(Preference-based<br>Recursive Language Modeling<br>for Exploratory Optimization of<br>Reasoning) | A generative framework that performs dynamic graph reasoning and iteratively expands<br>domain knowledge using a reinforcement learning inspired approach. It leverages<br>detailed knowledge graphs and abstract representations, allowing for hierarchical<br>reasoning and adaptive learning, following the approach detailed in [36]. |  |  |  |
| PRefLexOR                                                                                                            | A framework that enhances the reasoning capabilities of language models by integrating<br>preference optimization and recursive reasoning.                                                                                                                                                                                                |  |  |  |
| Knowledge Graph (KG)                                                                                                 | A graph representation of knowledge where nodes (V) represent concepts and edges (E)<br>represent relationships between concepts. In Graph-PRefLexOR, relationships include<br>"IS-A,"<br>"RELATES-TO,"<br>and "INFLUENCES."                                                                                                              |  |  |  |
| Abstract Pattern (P)                                                                                                 | Symbolic representations that summarize key relationships and dependencies identified<br>within a knowledge graph, often represented as a series of transformations.                                                                                                                                                                      |  |  |  |
| Recursive Reasoning                                                                                                  | An iterative process of refining reasoning by applying feedback to improve alignment<br>with the desired output.                                                                                                                                                                                                                          |  |  |  |
| Thinking Phase                                                                                                       | A stage where the model simulates intermediate reasoning steps, marked by the tokens<br>< thinking > and < /thinking > .                                                                                                                                                                                                                  |  |  |  |
| Reflection Phase                                                                                                     | A stage where the model evaluates the intermediate reasoning generated during the<br>thinking phase and provides feedback for improvement, here achieved via a second<br>agent.                                                                                                                                                           |  |  |  |
| Retrieval-Augmented Generation (RAG)                                                                                 | A technique used to enrich the input context by retrieving relevant information from a<br>knowledge index, enabling higher-order reasoning.                                                                                                                                                                                               |  |  |  |
| Semantic Scoring Function (f )                                                                                       | A function that evaluates the significance of relationships between concepts in a<br>knowledge graph.                                                                                                                                                                                                                                     |  |  |  |
| Base LLM, here: meta-llama/Llama<br>3.2-3B-Instruct                                                                  | The foundational large language model (LLM) used to train Graph-PRefLexOR.                                                                                                                                                                                                                                                                |  |  |  |
| Odds Ratio Preference Optimization<br>(ORPO)                                                                         | An optimization method used during the training of PRefLexOR [36] to guide reasoning<br>processes using special tokens.                                                                                                                                                                                                                   |  |  |  |
| Independent Reasoning Development                                                                                    | A stage in PRefLexOR training that leverages masking of thinking tokens to ensure the<br>model infers reasoning pathways without direct supervision.                                                                                                                                                                                      |  |  |  |
| Efficient Exact Optimization<br>(EXO)                                                                                | An optimization method used in PRefLexOR to align final outputs with preferred<br>reasoning patterns.                                                                                                                                                                                                                                     |  |  |  |

# 4.4 | Graph Visualization

We use mermaid.js to visualize graphs, where the output of the graph reasoning model is converted to the scripting language using Claude 3.5 Sonnet, then exported using the web editor [\(https://mermaid.live/\)](https://mermaid.live/). Additional visualizations and analyses are conducted using Gephi [\[59\]](#page-38-0) ([https://gephi.org/\)](https://gephi.org/) and various plugins therein.

## 4.5 | Glossary and Key Terms

Table 3 shows a summary of key terms and definitions used in this paper.

#### Acknowledgements

The author acknowledges support from the MIT Generative AI initiative.

#### Conflicts of Interest

The author declare no conflicts of interest.

### Data Availability Statement

Code, data, model weights availability, and additional materials are available at: <https://huggingface.co/lamm-mit> and [https://github.com/lamm](https://github.com/lamm-mit/PRefLexOR.)[mit/PRefLexOR.](https://github.com/lamm-mit/PRefLexOR.) The model is available at [https://huggingface.co/lamm](https://huggingface.co/lamm-mit/Graph-Preflexor_01062025)[mit/Graph-Preflexor\\_01062025](https://huggingface.co/lamm-mit/Graph-Preflexor_01062025).

#### References

- 1. A. Vaswani, N. Shazeer, N. Parmar, et al., "Attention is All you Need," (2017), <https://papers.nips.cc/paper/7181-attention-is-all-you-need>.
- 2. A. Radford , K. Narasimhan, T. Salimans, and I. Sutskeve, "Improving Language Understanding by Generative Pre-Training," [https://](https://gluebenchmark.com/leaderboard) [gluebenchmark.com/leaderboard.](https://gluebenchmark.com/leaderboard)
- 3. L. Xue, A. Barua, N. Constant, et al., "ByT5: Towards a Token-Free Future with Pre-Trained Byte-to-Byte Models," Transactions of the Association for Computational Linguistics 10 (2021): 291–306, [https://](https://arxiv.org/abs/2105.13626v3) [arxiv.org/abs/2105.13626v3.](https://arxiv.org/abs/2105.13626v3)

- <span id="page-37-0"></span>4. A. Q. Jiang, et al., Mistral 7B (2023), <http://arxiv.org/abs/2310.06825>.
- 5. Phi-2: The Surprising Power of Small Language Models Microsoft Research, [https://www.microsoft.com/en-us/research/blog/phi-2-the](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)[surprising-power-of-small-language-models/](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/).
- 6. A. Dubey, et al., "The llama 3 herd of Models," (2024), [https://arxiv.org/](https://arxiv.org/abs/2407.21783) [abs/2407.21783.](https://arxiv.org/abs/2407.21783)
- 7. T. B. Brown, B. Mann, N. Ryder, et al., "Language Models are Few-Shot Learners," (2020).
- 8. J. Jumper, R. Evans, A. Pritzel, et al., "Highly Accurate Protein Structure Prediction with AlphaFold," Nature (2021): 1-12.
- 9. D. Spivak, T. Giesa, E. Wood, and M. Buehler, "Category Theoretic Analysis of Hierarchical Protein Materials and Social Networks," PLoS ONE 6 (2011): e23911.
- 10. T. Giesa, D. I. Spivak and M. J. Buehler, "Reoccurring Patterns in Hierarchical Protein Materials and Music: The Power of Analogies," BioNanoScience 1 (2011): 153–161.
- 11. T. Giesa, D. Spivak, and M. Buehler, "Category Theory Based Solution for the Building Block Replacement Problem in Materials Design," Advanced Engineering Materials 14 (2012): 810–817.
- 12. V. Abbott and G. Zardini, "Flashattention on a Napkin: A Diagrammatic Approach to Deep Learning Io-Awareness," (2024), [https://arxiv.org/abs/2412.03317.](https://arxiv.org/abs/2412.03317)
- 13. M. J. Buehler, "Graph-Aware Isomorphic Attention for Adaptive Dynamics in Transformers," (2025), [https://arxiv.org/abs/2501.02393.](https://arxiv.org/abs/2501.02393)
- 14. T. Ackbarow, D. Sen, C. Thaulow, and M. Buehler, "Alpha-Helical Protein Networks Are Self-Protective and Flaw-Tolerant," PLoS ONE 4 (2009): e6015.
- 15. D. Nepal, et al., "Hierarchically Structured Bioinspired Nanocomposites," Nature Materials 2022 (2022): 1–18, [https://www.](https://www.nature.com/articles/s41563-022-01384-1) [nature.com/articles/s41563-022-01384-1.](https://www.nature.com/articles/s41563-022-01384-1)
- 16. R. Ritchie, M. Buehler, and P. Hansma, "Plasticity and Toughness in Bone," Physics Today 62 (2009): 41–47.
- 17. M. E. Launey, M. J. Buehler, and R. O. Ritchie, "On the Mechanistic Origins of Toughness in Bone," Annual Review of Materials Research 40 (2010): 25–53.
- 18. S. Eilenberg and S. MacLane, "Group Extensions and Homology," Annals of Mathematics 43 (1942): 757–831, [https://www.jstor.org/](https://www.jstor.org/stable/1968966#id-name=JSTORhttps://www.jstor.org/stable/1968966) stable/1968966#id-name=[JSTORhttps://www.jstor.org/stable/1968966.](https://www.jstor.org/stable/1968966#id-name=JSTORhttps://www.jstor.org/stable/1968966)
- 19. S. Eilenberg and S. Mac Lane, "General Theory of Natural Equivalences," Transactions of the American Mathematical Society 58 (1945): 247, [https://www.ams.org/journals/tran/1945-058-00/S0002-](https://www.ams.org/journals/tran/1945-058-00/S0002-9947-1945-0013131-6/S0002-9947-1945-0013131-6.pdf) [9947-1945-0013131-6/S0002-9947-1945-0013131-6.pdf](https://www.ams.org/journals/tran/1945-058-00/S0002-9947-1945-0013131-6/S0002-9947-1945-0013131-6.pdf).
- 20. T. Giesa, R. Jagadeesan, D. I. Spivak, and M. J. Matriarch Buehler, "A Python Library for Materials Architecture," ACS Biomaterials Science & Engineering (2015): 1009–1015, [http://pubs.acs.org/doi/10.1021/](http://pubs.acs.org/doi/10.1021/acsbiomaterials.5b00251) [acsbiomaterials.5b00251.](http://pubs.acs.org/doi/10.1021/acsbiomaterials.5b00251)
- 21. B. Weisfeiler and A. A. Leman, "Reduction of a Graph to a Canonical Form and an Algebra Arising during This Reduction," Nauchno-Technicheskaya Informatsia 2 (1968): 12–16.
- 22. S. W. Cranford and M. J. Buehler, Biomateriomics (Springer, 2012.
- 23. N. Groen, S. Cranford, J. de Boer, M. Buehler, and C. Van Blitterswijk, Materiomics: High-Throughput Screening of Biomaterial Properties (Cambridge University Press 2013), 1–12.
- 24. S. E. Arevalo and M. J. Buehler, "Learning from Nature by Leveraging Integrative Biomateriomics Modeling toward Adaptive and Functional Materials," MRS Bulletin 2023 (2023): 1–14, [https://link.springer.com/](https://link.springer.com/article/10.1557/s43577-023-00610-8) [article/10.1557/s43577-023-00610-8.](https://link.springer.com/article/10.1557/s43577-023-00610-8)
- 25. M. J. Buehler, "MeLM, a Generative Pretrained Language Modeling Framework that Solves Forward and Inverse Mechanics Problems,"

- Journal of the Mechanics and Physics of Solids 181 (2023): 105454, [https://linkinghub.elsevier.com/retrieve/pii/S0022509623002582.](https://linkinghub.elsevier.com/retrieve/pii/S0022509623002582)
- 26. A. Ghafarollahi and M. J. Buehler, "Sciagents: Automating Scientific Discovery through Multi-Agent Intelligent Graph Reasoning," Advanced Materieals (2024), [https://advanced.onlinelibrary.wiley.com/doi/full/10.](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/adma.202413523) [1002/adma.202413523](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/adma.202413523).
- 27. T. Dreyer, et al., "Comparing Cooperative Geometric Puzzle Solving in Ants versus Humans," Proceedings of the National Academy of Sciences of the United States of America 122 (2025): e2414274121.[https://www.pnas.](https://www.pnas.org/doi/abs/10.1073/pnas.2414274121) [org/doi/abs/10.1073/pnas.2414274121.](https://www.pnas.org/doi/abs/10.1073/pnas.2414274121)
- 28. K. Singhal, et al., Large Language Models Encode Clinical Knowledge, Nature (2023): 172–180, [https://www.nature.com/articles/s41586-023-](https://www.nature.com/articles/s41586-023-06048-6) [06048-6](https://www.nature.com/articles/s41586-023-06048-6).
- 29. A. Ghafarollahi and M. J. Buehler, "Protagents: Protein Discovery via Large Language Model Multi-Agent Collaborations Combining Physics and Machine Learning," Digital Discovery 3 (2024): 1389–1409, [https://](https://doi.org/10.1039/D4DD00013G-10.1039/D4DD00013G) [doi.org/10.1039/D4DD00013G-10.1039/D4DD00013G](https://doi.org/10.1039/D4DD00013G-10.1039/D4DD00013G).
- 30. R. K. Luu and M. J. Buehler, "BioinspiredLLM:Conversational Large Language Model for the Mechanics of Biological and Bio-Inspired Materials," Advanced Science (2023): 2306724, [https://doi.org/10.1002/](https://doi.org/10.1002/advs.202306724-10.1002/advs.202306724) [advs.202306724-10.1002/advs.202306724](https://doi.org/10.1002/advs.202306724-10.1002/advs.202306724).
- 31. S. Yu, N. Ran, and J. Liu, "Large-Language Models: The Game-Changers for Materials Science Research. AI in Chemical Engineering," Under a Creative Commons License (2024): 100076, [https://doi.org/10.](https://doi.org/10.1016/j.aichem.2024.100076) [1016/j.aichem.2024.100076](https://doi.org/10.1016/j.aichem.2024.100076) .
- 32. Y. Hu and M. J. Buehler, "Deep Language Models for Interpretative and Predictive Materials Science," APL Machine Learning 1 (2023): 010901, <https://aip.scitation.org/doi/abs/10.1063/5.0134317>.
- 33. E. L. Buehler and M. J. Buehler, "X-LoRA: Mixture of Low-Rank Adapter Experts, a Flexible Framework for Large Language Models with Applications in Protein Mechanics and Design," (2024), [https://arxiv.](https://arxiv.org/abs/2402.07148v1) [org/abs/2402.07148v1.](https://arxiv.org/abs/2402.07148v1)
- 34. M. J. MechGPT. Buehler, "A Language-Based Strategy for Mechanics and Materials Modeling that Connects Knowledge across Scales, Disciplines and Modalities," Applied Mechanics Reviews, 76 (2024): 021001, <https://doi.org/10.1115/1.4063843-10.1115/1.4063843>.
- 35. W. Lu, R. K. Luu, and M. J. Buehler, "Fine-Tuning Large Language Models for Domain Adaptation: Exploration of Training Strategies, Scaling, Model Merging and Synergistic Capabilities," NPJ Computational Materials, 11 (2025), 84, [https://www.nature.com/](https://www.nature.com/articles/s41524-025-01564-y) [articles/s41524-025-01564-y](https://www.nature.com/articles/s41524-025-01564-y).
- 36. M. J. Preflexor Buehler, "Preference-Based Recursive Language Modeling for Exploratory Optimization of Reasoning and Agentic Thinking," NPJ Artificial Intelligence (2025), (in press) [https://arxiv.](https://arxiv.org/abs/2410.12375) [org/abs/2410.12375](https://arxiv.org/abs/2410.12375).
- 37. E. Zelikman, Y. Wu, J. Mu, and N. D. Goodman, "Star: Bootstrapping Reasoning with Reasoning," (2022).
- 38. E. Zelikman, et al., "Quiet-Star: Language Models Can Teach Themselves to Think before Speaking," (2024), [https://arxiv.org/abs/](https://arxiv.org/abs/2403.09629) [2403.09629](https://arxiv.org/abs/2403.09629).
- 39. J. Wei, et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models, (2023), <https://arxiv.org/abs/2201.11903>.
- 40. T. Kojima, S. S. Gu, M. Reid, Y. Matsuo, and Y. Iwasawa, "Large Language Models Are Zero-Shot Reasoners, (2023), [https://arxiv.org/](https://arxiv.org/abs/2205.11916) [abs/2205.11916](https://arxiv.org/abs/2205.11916).
- 41. D. E. Rumelhart, G. E. Hinton, and R. J. Williams, "Learning Representations by Back-Propagating Errors," Nature 323 (1986): 533–536.
- 42. A. Newell and H. A. Simon, Human Problem Solving (Prentice-Hall, 1972).
- 43. J. McCarthy, "Programs with Common Sense," vol 1, (RLE and MIT Computation Center, 1960), 77–84.

- <span id="page-38-0"></span>44. H. Ji, et al., "Towards Efficient Exact Optimization of Language Model Alignment" (2024), <https://arxiv.org/abs/2402.00856>.
- 45. S. Kullback and R. A. Leibler, "On Information and Sufficiency," Annals of Mathematical Statistics 22 (1951): 79–86.
- 46. S. Brin and L. Page, "The anatomy of a large-scale hypertextual web search engine," in Proceedings of the Seventh International Conference on the World Wide Web (WWW 1998) (Elsevier Science Publishers B.V, 1998), 107–117.
- 47. T. M. J. Fruchterman and E. M. Reingold, "Graph Drawing by Force-Directed Placement," Software: Practice and Experience 21 (1991): 1129–1164.
- 48. Y. F. Hu, "Efficient and High Quality Force-Directed Graph Drawing," The Mathematica Journal 10 (2005): 37–71.
- 49. H. Hesse, the Glass Bead Game (Vintage Classics, 2000).
- 50. T. Ziolkowski, the Novels of Hermann Hesse: A Study in Theme and Structure, Includes Analysis of the Glass Bead Game and Its Conceptual Influence (Princeton University Press, 1965).
- 51. OpenAI, "OpenAI o1 System Card," (2024), [https://arxiv.org/abs/2412.](https://arxiv.org/abs/2412.16720) [16720.](https://arxiv.org/abs/2412.16720)
- 52. A. Madaan, N. Tandon, P. Gupta, et al., "Self-Refine: Iterative Refinement with Self-Feedback," (2023), <https://arxiv.org/abs/2303.17651>.
- 53. V. P. Dwivedi and X. Bresson, "A Generalization of Transformer Networks to Graphs," ( 2021), [https://arxiv.org/abs/2012.09699.](https://arxiv.org/abs/2012.09699)
- 54. Q. Wu, G. Bansal, J. Zhang, et al., "AutoGisomoren: Enabling Next-Gen Llm Applications via Multi-Agent Conversation," (2023), [https://arxiv.org/](https://arxiv.org/abs/2308.08155) [abs/2308.08155.](https://arxiv.org/abs/2308.08155)
- 55. Y. Wu, T. Yue, S. Zhang, C. Wang, and Q. Wu, "Stateflow: Enhancing Llm Task-Solving through State-Driven Workflows," (2024), [https://arxiv.](https://arxiv.org/abs/2403.11322) [org/abs/2403.11322](https://arxiv.org/abs/2403.11322).
- 56. A. Ghafarollahi and M. J. Buehler, "Atomagents: Alloy Design and Discovery through Physics-Aware Multi-Modal Multi-Agent Artificial Intelligence," Proceedings of the National Academy of Sciences USA 122, (2025): e2414074122 <https://doi.org/10.1073/pnas.2414074122>.
- 57. J. Hong, N. Lee, and J. Thorne, "ORPO: Monolithic Preference Optimization without Reference Model," (2024), [https://arxiv.org/abs/](https://arxiv.org/abs/2403.07691) [2403.07691](https://arxiv.org/abs/2403.07691).
- 58. R. Rafailov, et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," (2024), [https://arxiv.org/abs/2305.](https://arxiv.org/abs/2305.18290) [18290.](https://arxiv.org/abs/2305.18290)
- 59. M. Bastian, S. Heymann, and M. Jacomy, "Gephi: An Open Source Software for Exploring and Manipulating Networks," (2009), [http://](http://www.aaai.org/ocs/index.php/ICWSM/09/paper/view/154) [www.aaai.org/ocs/index.php/ICWSM/09/paper/view/154.](http://www.aaai.org/ocs/index.php/ICWSM/09/paper/view/154)

### Supporting Information

Additional supporting information can be found online in the Supporting Information section.