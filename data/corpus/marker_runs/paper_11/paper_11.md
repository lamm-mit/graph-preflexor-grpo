## SWARMS OF LARGE LANGUAGE MODEL AGENTS FOR PROTEIN SEQUENCE DESIGN WITH EXPERIMENTAL VALIDATION

Fiona Y. Wang<sup>1</sup> Di Sheng Lee<sup>2</sup> David L. Kaplan<sup>2</sup> Markus J. Buehler3,#

<sup>1</sup>Laboratory for Atomistic and Molecular Mechanics (LAMM), Department of Biological Engineering, Massachusetts Institute of Technology, Cambridge, MA 02139, USA <sup>2</sup>Department of Biomedical Engineering, Tufts University, Medford, MA, 02155 <sup>3</sup>Laboratory for Atomistic and Molecular Mechanics (LAMM), Department of Civil and Environmental Engineering, Department of Mechanical Engineering, Center for Computational Science and Engineering, Schwarzman College of Computing, Massachusetts Institute of Technology, Cambridge, MA 02139, USA

Corresponding author: # mbuehler@mit.edu

## ABSTRACT

Designing proteins *de novo* with tailored structural, physicochemical, and functional properties remains a grand challenge in biotechnology, medicine, and materials science, due to the vastness of sequence space and the complex coupling between sequence, structure, and function. Current state-of-the-art generative methods, such as protein language models (PLMs) and diffusion-based architectures, often require extensive fine-tuning, task-specific data, or model reconfiguration to support objective-directed design, thereby limiting their flexibility and scalability. To overcome these limitations, we present a decentralized, agent-based framework inspired by swarm intelligence for *de novo* protein design. In this approach, multiple large language model (LLM) agents operate in parallel, each assigned to a specific residue position. These agents iteratively propose contextaware mutations by integrating design objectives, local neighborhood interactions, and memory and feedback from previous iterations. This position-wise, decentralized coordination enables emergent design of diverse, well-defined sequences without reliance on motif scaffolds or multiple sequence alignments, validated with experiments on proteins with alpha helix and coil structures. Through analyses of residue conservation, structure-based metrics, and sequence convergence and embeddings, we demonstrate that the framework exhibits emergent behaviors and effective navigation of the protein fitness landscape. Our method achieves efficient, objective-directed designs within a few GPU-hours and operates entirely without fine-tuning or specialized training, offering a generalizable and adaptable solution for protein design. Beyond proteins, the approach lays the groundwork for collective LLM-driven design across biomolecular systems and other scientific discovery tasks.

Keywords protein design · swarm intelligence · large language models · agent-based systems

## 1 Introduction

The ability to computationally design novel proteins with precisely tailored structural, physicochemical, and functional properties is a grand challenge at the forefront of modern biotechnology [\[1](#page-15-0)[–6\]](#page-15-1), holding immense promise for advancements across medicine [\[7–](#page-15-2)[9\]](#page-15-3), materials science [\[10](#page-15-4)[–12\]](#page-15-5), and synthetic biology [\[13,](#page-15-6) [14\]](#page-16-0). *De novo* protein design, which aims to create entirely new amino acid sequences that fold into desired three-dimensional structures and perform specific tasks, offers a powerful alternative to modifying existing natural proteins [\[15\]](#page-16-1). However, navigating the astronomically vast protein sequence space, coupled with the intricate and often non-intuitive relationship between sequence, structure, and function, renders this a formidable computational and experimental endeavor.

Traditional computational protein design methods have made significant strides, often relying on physics-based energy functions or statistical potentials to guide sequence optimization within a predefined structural scaffold [\[16,](#page-16-2) [17\]](#page-16-3). More recently, the advent of deep learning has revolutionized protein science, giving rise to powerful neural networks [\[18–](#page-16-4)[22\]](#page-16-5), protein language models (PLMs) [\[23–](#page-16-6)[26\]](#page-16-7), and denoising diffusion probabilistic models [\[27](#page-16-8)[–29\]](#page-16-9). While these models excel at tasks like protein folding prediction or generating natural-like sequences, their application to targeted *de novo* protein design often presents significant limitations. Specifically, many state-of-the-art generative models typically demand extensive fine-tuning on large, task-specific datasets or intricate architectural modifications to achieve multiobjective design (Figure [1\)](#page-2-0). This reliance on specialized training data and computationally intensive learning phases restricts their generalizability, adaptability to novel design objectives, and overall efficiency for rapid prototyping in diverse design scenarios. Consequently, a significant gap remains in developing highly adaptable, generalizable, and computationally efficient *de novo* protein design frameworks that do not require extensive pre-training or fine-tuning for each new objective.

Large Language Models (LLMs), powerful neural networks initially designed for natural language processing tasks, have demonstrated an unprecedented ability to learn complex patterns, capture intricate relationships within sequential data, and generate coherent, contextually relevant outputs [\[30,](#page-16-10) [31\]](#page-16-11). Their success stems from their transformer-based architectures, which enable them to process long-range dependencies and learn rich representations from vast amounts of unsupervised data. This inherent capacity for pattern recognition and sequence generation has naturally extended their application to diverse scientific domains, including the automation of scientific discovery through bio-inspired multi-agent intelligent graph reasoning [\[32\]](#page-16-12), the exploratory optimization of reasoning and agentic thinking [\[33\]](#page-16-13), the discovery of protein design principles [\[34\]](#page-16-14), protein discovery via physics- and machine learning-informed multi-agent collaborations [\[35\]](#page-17-0), the solving of complex mechanics problems and knowledge integration [\[36\]](#page-17-1), and molecular analysis and design using generative AI via multi-agent modeling [\[37\]](#page-17-2).

To address limitations of the current computational protein design methods and leveraging the advantages of LLMs, we introduce a novel, decentralized, and collaborative agent-based framework for *de novo* protein sequence design, drawing inspiration from the principles of swarm intelligence [\[38\]](#page-17-3). Our approach harnesses the emergent collective behavior of multiple LLM agents, each assigned to a specific residue position within the protein sequence. These LLM agents iteratively propose context-aware mutations, integrating rich information including explicit design objectives, fundamental protein folding principles, local neighborhood information, a memory of prior iterations, and evaluation to guide their decisions. This unique position-wise, decentralized strategy facilitates the emergent design of structured, diverse, and multi-objective protein sequences without the need for explicit motif scaffolds or resource-intensive multiple sequence alignments.

## 2 Results and Discussion

#### 2.1 Swarm Framework

Our protein sequence design framework leverages a decentralized, agent-based approach inspired by swarm intelligence (swarm framework), where individual LLM agents collaboratively optimize a protein sequence to meet predefined design objectives. The overall framework, depicted in Figure [1b](#page-2-0), comprises of an objective and shared reasoning hubs, a group of parallel LLM agents, and an evaluation module.

The process begins with the user-defined objective and input sequence (Figure [1c](#page-2-0)). This information is passed to the group of parallel LLM agents (Figure [1b](#page-2-0)), each in charge of a single residue position, e.g. a1, a2, . . . , an. Rather than relying on costly weight updates, these agents perform on-the-fly specialization, using the information provided by the objective, shared reasoning hubs, local context, and evaluation feedback to propose mutations for the current sequence (Figures S1-S5). By concatenating the proposed mutations from all agents, we obtain an updated sequence, which is then passed to the evaluation module, which converts the sequence into a PDB file and evaluates it against the design objectives. Local context is extracted from the PDB file and passed to the next iteration, allowing the swarm to learn from its history and generate emergent novelty. Memory history is extracted from the proposed sequence and its corresponding evaluation results, allowing the swarm to learn from its past decisions. The proposed sequence, local context, memory history, and evaluation feedback are then fed back into the prompt for the next iteration, allowing the LLM agents to collectively explore the design space, learn from its history and neighborhood, and improve on the evaluation results to achieve the final optimized sequence.

## 2.2 Swarm Framework Achieves Diverse Structural Objectives

As shown in Figure [2,](#page-3-0) the proposed framework generates protein sequences that demonstrate diverse secondary structures in accordance with predefined design objectives.

<span id="page-2-0"></span>Figure 1: a. Comparison of swarm framework with conventional single-model framework. b. Multiple LLM agents share the same design objective and reasoning hubs, each proposing mutations for a single residue position, producing updated local context and evaluation feedback. c. Starting with a design objective and input sequence, the agents propose mutations for each residue position, producing an updated sequence which is evaluated. The previously proposed sequences are stored in memory for future iterations. The local context, memory history, and evaluation feedback are used to guide the next round of mutations. d. Input prompt consists of the agent's role and task, local neighborhood and context, design goal and energy, and memory history. Output consists of reasoning and the proposed mutation.

<span id="page-3-0"></span>Figure 2: Design objective, start sequence, best sequence, its respective 3D structure, and sequence logo returned from 64 iterations with GPT-4o for four structural design objectives.

When challenged to form α-helices from an initial SSSS... sequence, the framework generates helical structures under two constraints: one set containing a diversity of hydrophilic residues (top panel) and another with repeating alanine, leucine, and glutamate (ALE) motifs (bottom panel). The corresponding sequences (SDEEDAAAQAKETESSES and ALAAALELAAAAEAALAAEAE) and sequence logos illustrate these preferences, showing enrichment for hydrophilic helical residues for the top panel, and enrichment for the ALE pattern for the bottom panel, in alignment with well-known helix stabilization rules [\[39,](#page-17-4) [40\]](#page-17-5). The framework also successfully engineers other topologies, as seen by its ability to design a sequence (STTSTVTSTTTTVIGTSTYYT) that forms a beta-strand by placing alternating hydrophobic and polar residues. The sequence logo illustrates the enrichment for hydrophobic (red) and polar (yellow) residues, in alignment with principles of β-sheet formation [\[41\]](#page-17-6). Furthermore, it can optimize for random structures, such as generating a "loose, extended coil" (KTEKTQQKTN) from a hydrophobic LLLL... sequence. The sequence logo illustrates the enrichment for charged (green and blue) and polar (yellow) residues, in alignment with principles of coil formation [\[42,](#page-17-7)[43\]](#page-17-8). In each example, the 3D-folded structures confirm that the sequence indeed achieves the desired structural motif.

To experimentally validate that the designed sequences indeed form the intended secondary structure motifs, circular dichroism (CD) spectroscopy was performed on two hydrophilic peptide examples that can be readily synthesized: the hydrophilic α-helix and the coil sequence. CD spectroscopy is a standard biophysical technique for probing the secondary structure content of peptides and proteins in solution [\[44\]](#page-17-9).

For the hydrophilic helix design, the CD spectrum in Figure [3a](#page-4-0) displayed characteristic double minima near 208 nm and 222 nm, which is a hallmark of α-helical conformation [\[45\]](#page-17-10). This confirms that the designed sequence adopts a helical structure in aqueous solution, in agreement with the in silico prediction. Similarly, the sequence designed to form a random coil exhibited in Figure [3b](#page-4-0) a very low ellipticity above 210 nm and a negative band near 195 nm, consistent with the spectral signature of a predominantly disordered (random coil) conformation [\[46\]](#page-17-11). These experimental results validate that the swarm framework can produce sequences with robust secondary structure content.

<span id="page-4-0"></span>Figure 3: CD spectra of the best sequence for the a. hydrophilic helix and the b. coil sequence.

Interestingly, the iterative design process, powered by LLMs such as GPT-4o [\[48\]](#page-17-12), reveals a dynamic balance between convergence and exploration over 64 iterations, as illustrated in Figure [4.](#page-5-0) This figure tracks Rosetta Energy [\[47,](#page-17-13) [49,](#page-17-14) [50\]](#page-17-15) (red), which reports the physical plausibility of each structure, and Structural Score (blue), which reflects how closely the predicted fold matches the target motif. Early iterations (blue, 40-45) push the sequence rapidly toward high Structural Scores (often >75%) while lowering Rosetta Energy; later iterations (orange, 48-64) keep the motif intact yet continue probing alternative, lower-energy sequences, underscoring the framework's alternating converge-and-explore strategy [\[51\]](#page-17-16).

## 2.3 Swarm Framework Achieves Physiochemical, Functional, and Multi-Domain Objectives

Beyond structural motifs, the swarm framework exhibits strong adaptability in achieving physiochemical, functional, and multi-domain design objectives such as matching specific vibrational frequency distributions, designing proteins with metal-binding pockets, and designing proteins with multi-domain structures.

As shown in the top panel of Figure [5,](#page-6-0) the framework accurately reproduces a predefined target normalized frequency distribution across six mode indices, which represent the lowest-frequency collective motions obtained from an

<span id="page-5-0"></span>Figure 4: Evolution of calculated Rosetta energy [\[47\]](#page-17-13) (red) and Structure Score (blue) over 64 iterations with GPT-4o for the design objective: choose residues that mirror their left and right neighbors to promote local symmetry. This plot visualizes the dynamic interplay between convergence (blue shaded regions, where Structure Score stabilizes at a high level and Energy stabilizes at a low level) and exploration (orange shaded regions, where Energy fluctuates more significantly) during the iterative design process.

Anisotropic Network Model (ANM) of the protein [\[52\]](#page-17-17). These frequencies correspond to the square roots of the eigenvalues derived from the Hessian matrix of the ANM [\[52\]](#page-17-17). The computed frequencies of the designed protein (blue) closely align with the target frequencies (red), with a cosine similarity of 0.991 and a mean squared error (MSE) of 6.57e-04, approaching almost perfect agreement, highlighting the precision with which the swarm framework can optimize for frequency distribution objectives.

Further insight into the design process of target frequency distribution is provided in Figure [6,](#page-7-0) which visualizes the evolution of Energy (blue) and Frequency Score (red) over time. Three distinct phases of convergence (blue) alternate with exploratory phases (orange), where energy fluctuations increase as the agents sample broader regions of sequence space. This dynamic oscillation between refinement and exploration enables the framework to identify optimized solutions while maintaining diversity in the designed sequences.

The middle panel of Figure [5](#page-6-0) presents the successful transformation from a beta hairpin into a metal-binding motif by introducing histidine, cysteine, and methionine residues. The highlighted four cysteine residues (shown in sticks) form a coordination pocket for the metal ion, which is consistent with the design objective. The sequence logo highlighted common metal-binding motifs such as CXXC [\[53\]](#page-17-18) without additional knowledge input, highlighting the framework's ability to achieve functional objectives with minimal guidance.

The bottom panel of Figure [5](#page-6-0) presents the transformation of a protein with N-terminal beta sheet and C-terminal alpha helix into a protein with N-terminal alpha helix and C-terminal beta sheet, which is consistent with the design objective. This indicates the framework is able to design longer sequences (136 residues) and potentially longer sequences when memory space permits.

#### 2.4 LLM Comparison

The choice of the underlying LLM substantially affects both convergence dynamics and exploration diversity within the swarm framework, as demonstrated in Figure [7.](#page-7-1) This figure provides a comparative evaluation across six distinct language models: grok-3-mini, GPT-4o-mini, Mistral-8B, GPT-4.1, GPT-4o, and Llama-3.2-3B, under the same local symmetry design objective. For each model, two visualizations are presented: on the left, a Hamming distance heatmap quantifies sequence similarity and convergence, while on the right, a UMAP projection clusters sequences based on their physicochemical properties, across design iterations.

The Hamming distance heatmaps (left panels of Figure [7\)](#page-7-1) visualize pairwise dissimilarity between sequences across iterations. Darker regions indicate smaller Hamming distances (higher similarity), while lighter regions represent

<span id="page-6-0"></span>Figure 5: Design objective, start sequence, best sequence, evidence of objective achievement, and sequence logo returned from 16 iterations with GPT-4o for three diverse design objectives.

<span id="page-7-0"></span>Figure 6: Evolution of calculated Rosetta energy [\[47\]](#page-17-13) (red) and Frequency Score (blue) over 64 iterations with GPT-4o for the design objective: design protein with vibrational frequency spectrum matching target=[0.1, 0.15, 0.5, 0.6, 0.7, 0.8]. This plot illustrates the multi-round convergence (blue shaded regions, where Frequency Score is high and stable and Energy is low) and exploration (orange shaded regions, where Frequency Score and Energy fluctuates) dynamics of the swarm framework over multiple iterative cycles for a non-structural design objective.

<span id="page-7-1"></span>Figure 7: Comparison of 6 LLMs for the local symmetry design objective. The left panels show Hamming distance heatmaps of sequences from 64 iterations, illustrating mutation convergence and diversification (darker colors indicate higher convergence, lighter (red) colors indicate less convergence). The right panels show UMAP clustering of sequences from 64 iterations based on physicochemical properties, with points colored by iteration, visualizing the exploration and convergence patterns for grok-3-mini, GPT-o4-mini, Mistral-8B, GPT-4.1, GPT-4o, and Llama-3.2-3B.

greater diversity. Examining these plots reveals that different language models lead to distinct convergence profiles in the sequence space, influenced by each model's reasoning style, architectural traits, and training distributions:

- grok-3-mini This model exhibits the strongest convergence behavior, with its heatmap showing uniformly dark shades, indicating that sequence diversity rapidly collapses. This is likely due to both its relatively small model parameter size and the low setting of reasoning\_effort = low, which together result in shallow sampling, limited hypothesis testing, and reduced capacity to explore multiple local optima [\[54\]](#page-17-19).
- GPT-o4-mini Although also a small model, GPT-o4-mini [\[55,](#page-17-20) [56\]](#page-18-0) demonstrates slightly more exploration early on before converging. Its pattern reflects a stronger prior for structured reasoning than grok-3-mini, enabling initial variation followed by deterministic settling. This balance may stem from more optimized architectural tuning in the GPT-o4-mini compared to grok.
  - Mistral-8B Mistral displays structured convergence with well-defined similarity blocks, indicating consistent and deterministic behavior. Its early-phase trajectories are less diverse, leaning towards convergence than exploration, consistent with its robust pattern-matching capabilities [\[57,](#page-18-1) [58\]](#page-18-2).
    - GPT-4.1 GPT-4.1 exhibits less convergence than Mistral-8B, with clearer signs of early-stage exploration and multiple regions of moderate similarity, suggesting a more distributed search strategy. Its abstraction and generalization capabilities may enable it to explore multiple promising directions in parallel before converging [\[59\]](#page-18-3).
    - GPT-4o GPT-4o demonstrates more exploratory behavior, as its Hamming distance heatmap shows a gradual gradient of similarity interweaved by intermittent convergent steps. This pattern reflects its real-time reasoning and multimodal flexibility which allows the model to adaptively explore input space before committing to final outputs [\[60\]](#page-18-4). Its ability to directly integrate and process rich sensory data enables a more nuanced exploration-convergence balance compared to models constrained to unimodal pipelines [\[61\]](#page-18-5).
- Llama-3.2-3B Llama-3.2 shows the highest level of exploration, with diffuse and lightly saturated patterns indicating minimal convergence. This sustained diversity may arise from weaker inductive biases or alignment constraints, favoring broader sampling at the cost of slower optimization toward design objectives [\[62–](#page-18-6)[64\]](#page-18-7).

The UMAP clustering plots (right panels of Figure [7\)](#page-7-1) offer a dimensionality-reduced representation of the generated sequences, projected according to their physicochemical properties. Points are color-coded by iteration, enabling a comparative view of how each language model navigates, samples, and ultimately converges within the sequence landscape over time.

- grok-3-mini The UMAP projection reveals a single, loosely defined cluster, with minimal differentiation between early and late iterations. This high degree of convergence in physicochemical space likely reflects the model's relatively low reasoning effort, resulting in a more deterministic and less exploratory trajectory toward a stable solution.
- GPT-4o-mini Convergence occurs around iteration 27, after which sequences segregate into two distinct clusters. This pattern suggests an initial exploratory phase followed by bifurcation into two dominant local minima within the physicochemical space.
  - Mistral-8B Two well-defined convergence clusters occur from iterations 2–25 and 28–64. The clear separation between early and late iteration clusters suggests that the model identifies two distinct regions in the design space, with an intermediate exploratory phase between them.
    - GPT-4.1 Two convergence regions are observed from iterations 5–27 and 29–64 though the separation between clusters is less distinct than that of other models. This suggests a more gradual transition between solution spaces, consistent with a less discrete and more continuous mode of convergence.
    - GPT-4o Three convergence phases are evident from iterations 2–27, 27–52, and 53–64. However, clusters are less sharply defined, suggesting a more continuous convergent process. This fluid progression aligns with GPT-4o's real-time reasoning and multimodal adaptability [\[60\]](#page-18-4).
- Llama-3.2-3B This model exhibits minimal convergence, with early and late sequences distributed homogeneously across the UMAP space. The lack of cluster formation reflects a persistent exploratory behavior, supported by the model's diffuse sampling strategy across the design landscape [\[62\]](#page-18-6).

Our comparative analysis of LLMs using both Hamming distance heatmaps and UMAP clustering reveals diverse behaviors in convergence and exploration, which influences the trajectory of protein sequence optimization within our swarm framework. Our results demonstrate that the balance between convergence and exploration can be tuned by selecting the appropriate LLM: models like grok-3-mini yield highly convergent optimization, while choices such as Llama-3.2-3B favor broader, more exploratory search in sequence space. Thus, model selection provides a practical lever to modulate search dynamics in protein design.

Behavioral differences between models stem not only from architecture but also from transparency. Proprietary models like GPT series are "black boxes," limiting reproducibility [\[65](#page-18-8)[–67\]](#page-18-9). In contrast, open-weight models like Mistral-8B and Llama-3.2-3B provide more transparency, but still withhold key details [\[68,](#page-18-10) [69\]](#page-18-11). Thus, the convergence behavior observed in swarm optimization is shaped by both the internal architecture of the model and its transparency [\[70\]](#page-18-12). The precise mechanistic basis underlying these convergence differences can only be determined through further investigation, a current limitation of our study.

#### 2.5 Benchmark

To validate the efficacy and unique advantages of our swarm framework, we evalutaed its performance against several established protein design methodologies, including structural prediction models, autoregressive protein language models (PLMs), and denoising diffusion probabilistic models. As shown in Figure [8a](#page-10-0), design freedom increases progressively across the spectrum of protein engineering approaches: starting with structural prediction models, which allow the least flexibility by evaluating fixed sequences; then to PLMs, which can generate sequences but are generally limited to patterns learned from natural proteins; followed by denoising diffusion probabilistic models, which offer more generative capability for sequence and backbone design. Our swarm framework builds upon and exceeds these levels of design freedom, providing the ability to impose diverse objectives without requiring retraining or modification for new tasks.

Figure [8b](#page-10-0) compares swarm framework's ability to design a helix–turn–helix motif to structural models such as AlphaFold [\[71,](#page-18-13) [72\]](#page-18-14). While AlphaFold excels in protein structure prediction, it is not inherently suited for *de novo* sequence design. Beginning from a poly-arginine sequence, the swarm framework applied structure-informed mutations to yield a novel sequence (IKPILRAKPPIIRIKAARIK) that, when folded using AlphaFold, adopted the desired helix–turn–helix conformation. This result demonstrates the generative power of the swarm framework in producing sequences that fold into specific target structures.

Figure [8c](#page-10-0) compares the swarm framework to autoregressive PLMs such as ProtGPT2 [\[23\]](#page-16-6) in designing sequences with specific patterns. While ProtGPT2 generates natural-like sequences, it offers limited control over explicit design constraints without substantial fine-tuning. In contrast, when assigned a specific sequence design objective of "every 4 residues must follow the pattern of Hydrophobic–Polar–Glycine–Aromatic", the swarm framework successfully produced a sequence (VSGFATGFINGYVSGYASGF) that strictly adheres to this rule. This highlights the swarm framework's capacity to incorporate user-defined, custom rules without requiring retraining or architectural changes.

Figure [8d](#page-10-0) compares the swarm framework against RFdiffusion [\[27\]](#page-16-8), a representative denoising diffusion model for protein backbone generation, combined with ProteinMPNN [\[21\]](#page-16-15) for sequence design. While diffusion-based approaches are powerful, they often require specialized modifications for multi-objective design. In our example, the swarm framework was tasked with simultaneously achieving two objectives: introducing flexibility via turn-promoting residues and generating a repeating sequence pattern. Starting from a poly-valine sequence, the swarm designed a sequence (GGPPIGIGGIGGPGIIIGGGG) that met both criteria (mostly loops and repeating GG pairs). This result demonstrates the swarm framework's capability for multi-objective optimization.

These comparisons underscore the design capability, adaptability, and generalizability of our swarm framework for diverse protein design tasks.

#### Swarm Explores Novel Regions of Sequence Space

To evaluate the novelty and biological plausibility of sequences generated by the swarm framework, we compared them to both naturally occurring proteins and sequences generated by established *de novo* design methods.

We compiled a dataset of sequences consisting of:

- 5,000 natural protein sequences from the SCOPe database (v2.08) [\[73\]](#page-18-15)
- 200 sequences generated by ProteinMPNN [\[21\]](#page-16-15) from 100 backbones generated by RFdiffusion [\[27\]](#page-16-8) (two sequences per backbone) with unconditioned diffusion
- 640 sequences generated by our swarm framework across 10 distinct design objectives (64 sequences per objective)

Figure [9a](#page-11-0) visualizes how swarm framework, ProteinMPNN, and natural protein sequences from the SCOPe database distribute in a unified feature space, using t-distributed Stochastic Neighbor Embedding (*t*-SNE) for dimensionality reduction. Each protein sequence was encoded as a feature vector encompassing amino acid composition, molecular weight, and aromaticity. The swarm framework samples sequence space broadly, generating sequences that are natural-

<span id="page-10-0"></span>Figure 8: a. Design freedom increases from structural prediction models, autoregressive protein language models, denoising diffusion probabilistic models to swarm framework. b. The swarm framework is capable of making informed mutations compared to static structural prediction models. c. The swarm framework can design sequences according to custom rules compared to autoregressive protein language models. d. The swarm framework optimize multiple objectives compared to denoising diffusion probabilistic models.

<span id="page-11-0"></span>Figure 9: a. *t*-SNE visualization comparing swarm sequences (blue), SCOPe natural proteins (red), and ProteinMPNN designs (wheat). b. Tree using neighbor-joining on numerical feature vectors with the same legend as in *t*-SNE visualization.

like, others resembling *de novo* designs from ProteinMPNN, and, notably, an additional distinct set that occupies regions unpopulated by either SCOPe or ProteinMPNN sequences.

The tree in Figure [9b](#page-11-0) visualizes the hierarchical relationships among sequences generated by the swarm framework, ProteinMPNN, and those derived from natural proteins in the SCOPe database. Construction of the tree is based on the same 22-dimensional numerical feature vectors. Sequence-to-sequence dissimilarities were computed, and the neighbor-joining algorithm was employed to infer the tree topology. The tree reveals that the swarm framework produces a wide array of sequence types, including those near natural-like proteins and *de novo* designs from ProteinMPNN, and others branching into novel regions of the sequence space. These results demonstrate that the swarm framework can generate both familiar and novel protein designs, extending beyond the reach of current natural and model-generated proteins.

#### Swarm Framework is Computationally Efficient

<span id="page-11-1"></span>A key advantage of the swarm framework is its computational efficiency, when compared to leading protein structure prediction and design models (Table [1\)](#page-11-1).

| Model              | Training Time        | Inference Time per Prediction |
|--------------------|----------------------|-------------------------------|
| AlphaFold [71, 72] | about 1400 TPU-days  | minutes to hours              |
| ProtGPT2 [23]      | 512 GPU-days         | minutes                       |
| ESM2 [74–76]       | about 1800 GPU-days  | 10 GPU-hours                  |
| RFdiffusion [27]   | about 1800 GPU-days  | minutes                       |
| ProteinMPNN [21]   | about 10 GPU-days    | minutes                       |
| Swarm (this work)  | no training required | a few GPU-hours               |

Table 1: Comparative compute resource for protein structure prediction and design models, comparing training time and inference time per prediction.

Unlike methods such as AlphaFold (about 1400 TPU-days) [\[71,](#page-18-13) [72\]](#page-18-14), ProtGPT2 (512 GPU-days) [\[23\]](#page-16-6), ESM2 (about 1800 GPU-days) [\[74–](#page-18-16)[76\]](#page-18-17), RFdiffusion (about 1800 GPU-days) [\[27\]](#page-16-8), and ProteinMPNN [\[21\]](#page-16-15) which rely on heavy pretraining, the swarm framework requires no training. By eliminating the need for pretraining, the swarm framework lowers the computational overhead and democratizes access to *de novo* protein design capabilities.

For inference, the swarm framework completes the full iterative design process within a few GPU-hours, with total runtime primarily determined by language model API response time. Compared to ESM2 [\[74](#page-18-16)[–76\]](#page-18-17) which may take approximately 10 GPU-hours per prediction, the swarm framework's GPU-hour estimate is more cost-effective. This combination of zero training cost and efficient inference renders the swarm framework an accessible and scalable approach for objective-directed protein design.

#### 3 Conclusion

We introduced a decentralized, collaborative agent-based framework for protein sequence design, inspired by swarm intelligence. By assigning individual LLM agents to specific residue positions and allowing them to iteratively propose context-aware mutations, the swarm framework demonstrates versatility across diverse design objectives.

Structurally, the swarm framework achieved precise designs of  $\alpha$ -helices,  $\beta$ -strands, and coils, with high fidelity to residue conservation patterns (Figure 2). Alpha-helices and coils were validated by CD spectroscopy (Figure 3). Functionally, the swarm framework engineered sequences to match the predefined vibrational frequency spectrum, satisfy metal-binding pockets, and form multi-domain structures (Figure 5). The iterative design process enables both convergence and exploration (Figure 4, 6). To tune these design dynamics, different LLMs can be used to control the convergence and exploration behaviors (Figure 7).

Benchmarking against state-of-the-art protein engineering methods, including structure prediction models, autoregressive PLMs, and diffusion models, highlighted the swarm framework's strengths in precise design control, adaptability on-the-fly, and multi-objective optimization (Figure 8). More importantly, the designed sequences explore biologically plausible and novel regions of protein space (Figure 9). In addition, the swarm framework requires no training and completes the design process within a few GPU-hours (Table 1).

The swarm framework departs from traditional, monolithic deep-learning paradigms as it employs a no-training generative approach where complex, multi-objective optimization emerges from the collaborative, local interactions of multiple specialized agents (Figure 1). This swarm-of-agents principle is highly generalizable and holds considerable promise for other complex design domains, where locally-informed agents collaborate to achieve global objectives.

#### 4 Methods

Codes and data are available at https://github.com/lamm-mit/ProteinSwarm. The swarm framework runs on a workstation equipped with a high-end CUDA-compatible GPU (Quadro RTX 5000, NVidia, Santa Clara, CA, USA).

#### 4.1 Swarm Framework

The swarm framework consists of four core components: a multi-agent swarm system where each residue position is managed by an autonomous LLM agent, a memory and learning system that tracks global and local patterns, a structure evaluation unit with OmegaFold [77] for structure prediction and Rosetta [47] for energy scoring, and a decision-making algorithm that accepts or rejects complete iterations based on structural and energetic criteria.

Each protein sequence is represented as a grid where each position i in the sequence  $S=(s_1,s_2,\ldots,s_n)$  is managed by an autonomous agent  $A_i$ . The agents operate on the amino acid sequence space  $\mathcal{A}^{20}$ , where each position can be occupied by any of the 20 standard amino acids. Each agent receives local context including linear sequence neighbors within a radius r:  $\mathcal{N}_i = \{s_{i-r}, \ldots, s_{i-1}, s_{i+1}, \ldots, s_{i+r}\}$ , spatial neighbors from the previous iteration's structure:  $\mathcal{S}_i = \{(j, d_{ij}) : d_{ij} < \text{cutoff}\}$ , solvent exposure information:  $E_i = \text{exposure}(i, S, D)$ , position-specific structural context from secondary structure analysis using DSSP software [78], and memory-based learning insights from previous iterations, where D is the  $C\alpha$  distance matrix from the previous iteration's folded structure.

The optimization proceeds through a cleanly separated four-phase loop:

**Phase 1: Agent Collection.** All agents simultaneously propose amino acid changes based on their local context and memory insights. Each agent  $A_i$  receives input context  $C_i = (\mathcal{N}_i, \mathcal{S}_i, E_i, M_i)$ , where  $M_i$  represents memory-based learning insights, and outputs a proposed amino acid  $a_i' \in \mathcal{A}^{20}$ . The system collects all proposals  $\mathbf{P} = (a_1', a_2', \dots, a_n')$  without any structure computation during this phase.

**Phase 2: Apply Changes.** The proposed sequence  $S' = (a'_1, a'_2, \dots, a'_n)$  is constructed from all agent proposals and the structure is predicted using OmegaFold:

$$S' \xrightarrow{\text{OmegaFold}} PDB' \tag{1}$$

The folding process used CUDA GPU acceleration with a subbatch size of 1 optimized for short sequences, outputting standard PDB coordinates.

**Phase 3: Structure Evaluation.** The folded structure is evaluated using multiple criteria including Rosetta energy [47] scoring ( $E_{\text{total}} = E_{\text{vdw}} + E_{\text{hbond}} + E_{\text{elec}} + \ldots$ ), secondary structure analysis using DSSP [78], and design objective-specific evaluation metrics.

Phase 4: Decision and Memory Update. The system decides whether to accept the proposed sequence based on:

$$\text{Accept} = \begin{cases} \text{True} & \text{Objective Score}(S') > \text{Objective Score}(S) \\ \text{True} & \text{if } E_{\text{total}}(S') < E_{\text{total}}(S) \text{ and Objective Score}(S') \approx \text{Objective Score}(S) \\ \text{False} & \text{otherwise} \end{cases}$$

If accepted,  $S \leftarrow S'$  and the memory system is updated with the successful pattern. If rejected, the original sequence is retained and the memory system records the failed attempt.

The framework implements a memory system that enables agents to learn from both global and local patterns.

**Global Memory.** The framework tracks system-wide patterns including accepted and rejected sequences ( $\mathcal{S}_{accepted}$ ,  $\mathcal{S}_{rejected}$ ), successful mutation patterns ( $\mathcal{P}_{success} = \{(p,c) : success\_rate(p) > \theta\}$ ), energy progression trends ( $\mathcal{T}_{energy} = \{(i,E_i) : i \in iterations\}$ ), and structure score trends ( $\mathcal{T}_{structure} = \{(i,S_i) : i \in iterations\}$ ).

**Local History.** Each agent maintains personal history tracking including personal action records ( $A_i = \{(a, \text{outcome}) : \text{position } i\}$ ), success rates for different amino acid substitutions, context-specific performance patterns, and neighboring position interaction effects.

Memory Context Generation. For each agent decision, the system provides:

$$M_i = f(\mathcal{G}, \mathcal{L}_i, \mathcal{C}_i) \tag{3}$$

where  $\mathcal{G}$  is global memory,  $\mathcal{L}_i$  is local history for position i, and  $\mathcal{C}_i$  is current context.

Each agent  $A_i$  receives a structured input including the current amino acid state  $(s_i)$ , local sequence context  $(\mathcal{N}_i)$ , spatial structural context  $(\mathcal{S}_i)$ , solvent exposure  $(E_i)$ , memory insights  $(M_i)$ , design objective (G), and previous iteration outcomes. The agent outputs a structured proposal:

$$Proposal_{i} = \{reasoning, proposed\_value\}$$
 (4)

where proposed\_value is constrained to a single amino acid code from  $A^{20}$ .

**Energy Calculation:** Rosetta computes detailed energy terms including van der Waals interactions, hydrogen bonding, electrostatic interactions, and reference energies:

$$E_{\text{total}} = \sum_{i} E_{\text{vdw}}(i) + \sum_{i,j} E_{\text{hbond}}(i,j) + \sum_{i,j} E_{\text{elec}}(i,j) + E_{\text{reference}}$$
(5)

**Secondary Structure Analysis:** DSSP assigns secondary structure elements to each residue, classifying them as  $\alpha$ -helix (H),  $\beta$ -strand (E), or loop/coil (L):

$$SS(i) = \begin{cases} H & \alpha\text{-helix} \\ E & \beta\text{-strand} \\ L & \text{loop/coil} \end{cases}$$
 (6)

**Objective-Specific Evaluation:** Custom metrics assess design objective achievement based on secondary structure composition, spatial arrangement, and energy terms:

ObjectiveScore = 
$$f(SS\_composition, spatial\_arrangement, energy\_terms)$$
 (7)

Each objective is evaluated using domain-specific metrics that combine structural analysis, energy scoring, and sequence composition analysis to provide comprehensive feedback to the swarm framework.

The system uses a recursion guard to prevent structure computation during agent processing, ensuring clean phase separation. Memory is persisted between iterations to enable learning, and all structural data is cached to minimize redundant computations.

#### 4.2 Experimental Validation

Peptides were chemically synthesized and purchased from Genscript (Piscataway, NJ, U.S.A.) at 98% purity followed by high-performance liquid chromatography. CD spectra were acquired using a peptide concentration of 1 mg/mL (SDEEDAAAQAKETESSES) dissolved in 0.1 M phosphate buffer (PB) and 0.1 mg/mL (KTEKTQQKTN) dissolved in 0.01 M PB respectively. The peptide concentration was chosen to stabilize the formation of secondary structures and

enhance signal-to-noise, especially for detecting subtle helical content. CD measurements were performed in circular dichroism spectrophotometer (Jasco #J-1500) in a standard 1-mm pathlength quartz cuvette (Jasco #1103-0172) at room temperature, with wavelength scans from 260 nm to 190 nm at 0.5 nm interval. Each reported spectrum is the average of five (helix) and three (coil) scans, and baseline correction was performed by subtracting the 0.1 M or 0.01 M PB buffer spectrum. The spectra were analyzed on the BESTSEL online server to determine secondary structures [\[79\]](#page-18-20).

## 4.3 Sequence Space Analysis

To evaluate the novelty and quality of swarm-generated sequences, we performed comprehensive sequence space analysis comparing swarm trajectories with ProteinMPNN-generated sequences. We extracted unique sequences from ten swarm trajectory datasets:

- form alpha helices using alanine, leucine, and glutamate in repeating patterns
- design one helix with one side hydrophobic and one side polar
- design beta strands by placing alternating hydrophobic and polar residues
- stabilize alpha helices using N-cap with serine, threonine and C-cap with glycine, proline, asparagine, aspartate
- form loose, extended coils using polar and charged residues to reduce compaction
- form a beta hairpin with two beta strands and one turn, using at least two aromatic residues in the strands and one proline in the turn
- promote compact local packing using hydrophobic residues and turn-promoting motifs
- create a helix-turn-helix motif with one helix followed by one turn and a second helix
- choose residues that mirror their left and right neighbors to promote local symmetry
- design protein with vibrational frequency spectrum matching target=[0.1, 0.15, 0.5, 0.6, 0.7, 0.8] for optimal dynamical properties

For comparison, we used ProteinMPNN-generated sequences as a baseline representing state-of-the-art deep learning protein design methods. All sequences were de-duplicated to ensure unique analysis.

For each sequence S, we computed a total of 22 numerical features: 20 amino acid composition features (C(S) = [cA, cR, . . . , c<sup>Y</sup> ] where c<sup>X</sup> is the frequency of amino acid X, representing the proportion of each of the 20 standard amino acids in the sequence), and 2 physicochemical properties (molecular weight, calculated as the average molecular weight per residue in Daltons, and aromaticity, representing the proportion of aromatic amino acids F, W, and Y). This feature set ensures fair comparison across all sequence types, as all sequences (SCOPe, SWARM, and ProteinMPNN) are treated equally based solely on sequence composition and physicochemical properties, without any structural bias that might penalize designed sequences lacking structural classifications. Data preprocessing involved robust feature selection to ensure numerical stability, including removal of near-zero variance features (threshold: 10<sup>−</sup><sup>7</sup> ) and elimination of highly correlated features (correlation threshold: 0.95).

*t*-Distributed Stochastic Neighbor Embedding (*t*-SNE) was performed to visualize sequence relationships in a 2D embedding space. The algorithm calculates pairwise similarities in the high-dimensional feature space and creates a 2D map that preserves local structure, where similar sequences cluster together and dissimilar sequences are pushed apart. We used optimized parameters for protein sequence analysis: perplexity of 30 to balance local and global structure preservation, maximum iterations of 1000 with early exaggeration, and random seed initialization for reproducibility. The perplexity parameter was set as min(30, ⌊(N − 1)/3⌋) where N is the number of sequences. *t*-SNE minimizes the Kullback–Leibler divergence between pairwise similarities in high- and low-dimensional spaces:

$$KL(P \parallel Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$
(8)

where pij denotes the similarity between data points i and j in the high-dimensional space, and qij denotes their similarity in the low-dimensional embedding. The resulting 2D coordinates (*t*-SNE1, *t*-SNE2) for each sequence enable visualization where distance in 2D approximates sequence similarity, with clusters representing groups of sequences sharing similar properties and separation indicating different regions occupied by different methods (SCOPe, SWARM, ProteinMPNN).

Trees were constructed using the neighbor-joining (NJ) algorithm on Euclidean distance matrices computed from the feature vectors. The distance between sequences i and j was calculated as dij = pP k (xik − xjk) <sup>2</sup> for feature vectors x<sup>i</sup> and x<sup>j</sup> , where lower distance indicates more similar sequences. The NJ algorithm iteratively groups the nearest neighbors (most similar sequences) based on the minimum evolution principle, creating an unrooted tree that minimizes the total branch length while maintaining additivity of distances. For computational efficiency, datasets exceeding a size threshold were subsampled to 1000 sequences. The resulting tree was rooted at the first sequence (arbitrary rooting) to convert it to a rooted tree for visualization. It is important to note that this tree is based on feature similarity rather than evolutionary history, showing functional and structural relationships between sequences rather than true phylogenetic relationships.

## Acknowledgments

Compute: We acknowledge support from the MIT Office of Research Computing and Data for providing computational resources. Funding: We acknowledge support by the Bernard E. Proctor Memorial Fund and 2025 Mathworks Fellowship. Author contributions: M.J.B. and F.Y.W. conceived the idea. F.Y.W. and M.J.B. developed the framework, ran iterations, and analyzed the results. D.S.L. conducted CD spectroscopy experiments. M.J.B. and D.L.K. supervised the project. F.Y.W. and M.J.B. wrote the manuscript. Competing interests: The authors declare that they have no competing interests. Data and materials availability: All data needed to evaluate the conclusions in the paper are present in the paper and the Supplementary Materials. Additional data related to this paper may be requested from the corresponding author.

## References

- <span id="page-15-0"></span>[1] Albanese, K. I., Barbe, S., Tagami, S., Woolfson, D. N. & Schiex, T. Computational protein design. Nature Reviews Methods Primers 5, 13 (2025). URL <https://doi.org/10.1038/s43586-025-00383-1>.
- [2] Listov, D., Goverde, C. A., Correia, B. E. & Fleishman, S. J. Opportunities and challenges in design and optimization of protein function. Nature Reviews Molecular Cell Biology 25, 639–653 (2024). URL [https:](https://doi.org/10.1038/s41580-024-00718-y) [//doi.org/10.1038/s41580-024-00718-y](https://doi.org/10.1038/s41580-024-00718-y).
- [3] Korendovych, I. V. & DeGrado, W. F. De novo protein design, a retrospective. Quarterly Reviews of Biophysics 53, e3 (2020). URL <https://www.cambridge.org/core/product/FF37903868E1651D7E61A8495FB00B50>.
- [4] Woolfson, D. N. A brief history of de novo protein design: Minimal, rational, and computational. Journal of Molecular Biology 433, 167160 (2021). URL [https://www.sciencedirect.com/science/](https://www.sciencedirect.com/science/article/pii/S0022283621003892) [article/pii/S0022283621003892](https://www.sciencedirect.com/science/article/pii/S0022283621003892).
- [5] Khakzad, H. et al. A new age in protein design empowered by deep learning. Cell Systems 14, 925–939 (2023). URL <https://www.sciencedirect.com/science/article/pii/S2405471223002983>.
- <span id="page-15-1"></span>[6] Ranbhor, R., Venkatesan, R., Redkar, A. S. & Ramakrishnan, V. Computational protein design: Advancing biotechnology through in silico engineering. Progress in Biophysics and Molecular Biology 197, 75–83 (2025). URL <https://www.sciencedirect.com/science/article/pii/S0079610725000380>.
- <span id="page-15-2"></span>[7] Kalita, P., Tripathi, T. & Padhi, A. K. Computational protein design for covid-19 research and emerging therapeutics. ACS Central Science 9, 602–613 (2023). URL <https://doi.org/10.1021/acscentsci.2c01513>. Doi: 10.1021/acscentsci.2c01513.
- [8] Rodda, L. B. et al. Functional sars-cov-2-specific immune memory persists after mild covid-19. Cell 184, 169– 183.e17 (2021). URL <https://doi.org/10.1016/j.cell.2020.11.029>. Doi: 10.1016/j.cell.2020.11.029.
- <span id="page-15-3"></span>[9] Walls, A. C. et al. Elicitation of broadly protective sarbecovirus immunity by receptor-binding domain nanoparticle vaccines. Cell 184, 5432–5447.e16 (2021). URL <https://doi.org/10.1016/j.cell.2021.09.015>. Doi: 10.1016/j.cell.2021.09.015.
- <span id="page-15-4"></span>[10] King, N. P. et al. Computational design of self-assembling protein nanomaterials with atomic level accuracy. Science 336, 1171–1174 (2012). URL <https://doi.org/10.1126/science.1219364>. Doi: 10.1126/science.1219364.
- [11] Bethel, N. P. et al. Precisely patterned nanofibres made from extendable protein multiplexes. Nature Chemistry 15, 1664–1671 (2023). URL <https://doi.org/10.1038/s41557-023-01314-x>.
- <span id="page-15-5"></span>[12] Tunuhe, A. et al. Protein-based materials: Applications, modification and molecular design. BioDesign Research 7, 100004 (2025). URL <https://www.sciencedirect.com/science/article/pii/S2693125725000056>.
- <span id="page-15-6"></span>[13] Kortemme, T. <em>de novo</em> protein design—from new structures to programmable functions. Cell 187, 526–544 (2024). URL <https://doi.org/10.1016/j.cell.2023.12.028>. Doi: 10.1016/j.cell.2023.12.028.

- <span id="page-16-0"></span>[14] Kretschmer, S. & Kortemme, T. Advances in the computational design of small-molecule-controlled protein-based circuits for synthetic biology. Proceedings of the IEEE 110, 659–674 (2022).
- <span id="page-16-1"></span>[15] Pan, X. & Kortemme, T. Recent advances in de novo protein design: Principles, methods, and applications. Journal of Biological Chemistry 296, 100558 (2021). URL [https://www.sciencedirect.com/science/](https://www.sciencedirect.com/science/article/pii/S0021925821003367) [article/pii/S0021925821003367](https://www.sciencedirect.com/science/article/pii/S0021925821003367).
- <span id="page-16-2"></span>[16] Kuhlman, B. et al. Design of a novel globular protein fold with atomic-level accuracy. Science 302, 1364–1368 (2003). URL <https://doi.org/10.1126/science.1089427>. Doi: 10.1126/science.1089427.
- <span id="page-16-3"></span>[17] Chaudhury, S., Lyskov, S. & Gray, J. J. Pyrosetta: a script-based interface for implementing molecular modeling algorithms using rosetta. Bioinformatics 26, 689–691 (2010). URL [https://doi.org/10.1093/](https://doi.org/10.1093/bioinformatics/btq007) [bioinformatics/btq007](https://doi.org/10.1093/bioinformatics/btq007).
- <span id="page-16-4"></span>[18] Xu, J. Distance-based protein folding powered by deep learning. Proceedings of the National Academy of Sciences 116, 16856–16865 (2019). URL [https://doi.org/](https://doi.org/10.1073/pnas.1821309116) [10.1073/pnas.1821309116](https://doi.org/10.1073/pnas.1821309116). Doi: 10.1073/pnas.1821309116.
- [19] Anishchenko, I. et al. De novo protein design by deep network hallucination. Nature 600, 547–552 (2021). URL <https://doi.org/10.1038/s41586-021-04184-w>.
- [20] Baek, M. et al. Accurate prediction of protein structures and interactions using a three-track neural network. Science 373, 871–876 (2021). URL <https://doi.org/10.1126/science.abj8754>. Doi: 10.1126/science.abj8754.
- <span id="page-16-15"></span>[21] Dauparas, J. et al. Robust deep learning–based protein sequence design using proteinmpnn. Science 378, 49–56 (2022). URL <https://doi.org/10.1126/science.add2187>. Doi: 10.1126/science.add2187.
- <span id="page-16-5"></span>[22] Singer, J. M. et al. Large-scale design and refinement of stable proteins using sequence-only models. PLOS ONE 17, e0265020 (2022). URL <https://doi.org/10.1371/journal.pone.0265020>.
- <span id="page-16-6"></span>[23] Ferruz, N., Schmidt, S. & Höcker, B. Protgpt2 is a deep unsupervised language model for protein design. Nature Communications 13, 4348 (2022). URL <https://doi.org/10.1038/s41467-022-32007-7>.
- [24] Hayes, T. et al. Simulating 500 million years of evolution with a language model. Science 387, 850–858 (2025). URL <https://doi.org/10.1126/science.ads0018>. Doi: 10.1126/science.ads0018.
- [25] Nijkamp, E., Ruffolo, J. A., Weinstein, E. N., Naik, N. & Madani, A. Progen2: Exploring the boundaries of protein language models. Cell Systems 14, 968–978.e3 (2023). URL [https://doi.org/10.1016/j.cels.](https://doi.org/10.1016/j.cels.2023.10.002) [2023.10.002](https://doi.org/10.1016/j.cels.2023.10.002). Doi: 10.1016/j.cels.2023.10.002.
- <span id="page-16-7"></span>[26] Lv, L. et al. Prollama: A protein large language model for multi-task protein language processing. IEEE Transactions on Artificial Intelligence 1–12 (2025).
- <span id="page-16-8"></span>[27] Watson, J. L. et al. De novo design of protein structure and function with rfdiffusion. Nature 620, 1089–1100 (2023). URL <https://doi.org/10.1038/s41586-023-06415-8>.
- [28] Trippe, B. L. et al. Diffusion probabilistic modeling of protein backbones in 3d for the motif-scaffolding problem. arXiv preprint arXiv:2206.04119 (2022).
- <span id="page-16-9"></span>[29] Anand, N. & Achim, T. Protein structure and sequence generation with equivariant denoising diffusion probabilistic models. https://arxiv.org/abs/2205.15019 (2022).
- <span id="page-16-10"></span>[30] Vaswani, A. et al. Attention is all you need. In Guyon, I. et al. (eds.) Advances in Neural Information Processing Systems, vol. 30 (Curran Associates, Inc., 2017). URL [https://proceedings.neurips.cc/paper\\_files/paper/2017/file/](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) [3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).
- <span id="page-16-11"></span>[31] Brown, T. et al. Language models are few-shot learners. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M. & Lin, H. (eds.) Advances in Neural Information Processing Systems, vol. 33, 1877–1901 (Curran Associates, Inc., 2020). URL [https://proceedings.neurips.cc/paper\\_files/paper/2020/file/](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) [1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf).
- <span id="page-16-12"></span>[32] Ghafarollahi, A. & Buehler, M. J. Sciagents: Automating scientific discovery through bioinspired multi-agent intelligent graph reasoning. Advanced Materials 37, 2413523 (2025). URL [https://doi.org/10.1002/adma.](https://doi.org/10.1002/adma.202413523) [202413523](https://doi.org/10.1002/adma.202413523).
- <span id="page-16-13"></span>[33] Buehler, M. J. Preflexor: preference-based recursive language modeling for exploratory optimization of reasoning and agentic thinking. npj Artificial Intelligence 1, 4 (2025). URL [https://doi.org/10.1038/](https://doi.org/10.1038/s44387-025-00003-z) [s44387-025-00003-z](https://doi.org/10.1038/s44387-025-00003-z).
- <span id="page-16-14"></span>[34] Ghafarollahi, A. & Buehler, M. J. Sparks: Multi-agent artificial intelligence model discovers protein design principles (2025). URL <https://arxiv.org/abs/2504.19017>. <2504.19017>.

- <span id="page-17-0"></span>[35] Ghafarollahi, A. & Buehler, M. J. Protagents: protein discovery via large language model multi-agent collaborations combining physics and machine learning. Digital Discovery 3, 1389–1409 (2024). URL <http://dx.doi.org/10.1039/D4DD00013G>.
- <span id="page-17-1"></span>[36] Ni, B. & Buehler, M. J. Mechagents: Large language model multi-agent collaborations can solve mechanics problems, generate new data, and integrate knowledge. Extreme Mechanics Letters 67, 102131 (2024). URL <https://www.sciencedirect.com/science/article/pii/S2352431624000117>.
- <span id="page-17-2"></span>[37] Stewart, I. & Buehler, M. J. Molecular analysis and design using generative artificial intelligence via multi-agent modeling. Mol. Syst. Des. Eng. 10, 314–337 (2025). URL <http://dx.doi.org/10.1039/D4ME00174E>.
- <span id="page-17-3"></span>[38] Rahman, M. A. U. & Schranz, M. Llm-powered swarms: A new frontier or a conceptual stretch? (2025). URL <https://arxiv.org/abs/2506.14496>. <2506.14496>.
- <span id="page-17-4"></span>[39] Spek, E. J., Olson, C. A., Shi, Z. & Kallenbach, N. R. Alanine is an intrinsic α-helix stabilizing amino acid. Journal of the American Chemical Society 121, 5571–5572 (1999). URL [https://doi.org/10.1021/](https://doi.org/10.1021/ja990056x) [ja990056x](https://doi.org/10.1021/ja990056x). Doi: 10.1021/ja990056x.
- <span id="page-17-5"></span>[40] Lyu, P. C., Sherman, J. C., Chen, A. & Kallenbach, N. R. Alpha-helix stabilization by natural and unnatural amino acids with alkyl side chains. Proceedings of the National Academy of Sciences 88, 5317–5320 (1991). URL <https://doi.org/10.1073/pnas.88.12.5317>. Doi: 10.1073/pnas.88.12.5317.
- <span id="page-17-6"></span>[41] Mandel-Gutfreund, Y. & Gregoret, L. M. On the significance of alternating patterns of polar and non-polar residues in beta-strands. Journal of Molecular Biology 323, 453–461 (2002). URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/S0022283602009737) [science/article/pii/S0022283602009737](https://www.sciencedirect.com/science/article/pii/S0022283602009737).
- <span id="page-17-7"></span>[42] van der Lee, R. et al. Classification of intrinsically disordered regions and proteins. Chemical Reviews 114, 6589–6631 (2014). URL <https://doi.org/10.1021/cr400525m>. Doi: 10.1021/cr400525m.
- <span id="page-17-8"></span>[43] Radivojac, P. et al. Protein flexibility and intrinsic disorder. Protein Science 13, 71–80 (2004). URL [https:](https://doi.org/10.1110/ps.03128904) [//doi.org/10.1110/ps.03128904](https://doi.org/10.1110/ps.03128904).
- <span id="page-17-9"></span>[44] Greenfield, N. J. Using circular dichroism spectra to estimate protein secondary structure. Nature Protocols 1, 2876–2890 (2006). URL <https://doi.org/10.1038/nprot.2006.202>.
- <span id="page-17-10"></span>[45] Holzwarth, G. & Doty, P. The ultraviolet circular dichroism of polypeptides1. Journal of the American Chemical Society 87, 218–228 (1965). URL [https://doi.org/10.1021/](https://doi.org/10.1021/ja01080a015) [ja01080a015](https://doi.org/10.1021/ja01080a015). Doi: 10.1021/ja01080a015.
- <span id="page-17-11"></span>[46] Venyaminov, S. Y., Baikalov, I. A., Shen, Z. M., Wu, C. S. C. & Yang, J. T. Circular dichroic analysis of denatured proteins: Inclusion of denatured proteins in the reference set. Analytical Biochemistry 214, 17–24 (1993). URL <https://www.sciencedirect.com/science/article/pii/S0003269783714508>.
- <span id="page-17-13"></span>[47] Chaudhury, S., Lyskov, S. & Gray, J. J. Pyrosetta: a script-based interface for implementing molecular modeling algorithms using rosetta. Bioinformatics 26, 689–691 (2010). URL [https://doi.org/10.1093/](https://doi.org/10.1093/bioinformatics/btq007) [bioinformatics/btq007](https://doi.org/10.1093/bioinformatics/btq007).
- <span id="page-17-12"></span>[48] OpenAI. Chatgpt. <https://platform.openai.com/> (2024). Accessed via the ChatGPT API.
- <span id="page-17-14"></span>[49] Alford, R. F. et al. The rosetta all-atom energy function for macromolecular modeling and design. Journal of Chemical Theory and Computation 13, 3031–3048 (2017). URL [https://doi.org/10.1021/acs.](https://doi.org/10.1021/acs.jctc.7b00125) [jctc.7b00125](https://doi.org/10.1021/acs.jctc.7b00125). Doi: 10.1021/acs.jctc.7b00125.
- <span id="page-17-15"></span>[50] Alford, R. F. et al. A deep-dive into the rosetta energy function for biological macromolecules. Biophysical Journal 112, 194a (2017). URL <https://doi.org/10.1016/j.bpj.2016.11.1078>. Doi: 10.1016/j.bpj.2016.11.1078.
- <span id="page-17-16"></span>[51] Ma, R., Luijkx, J., Ajanovic, Z. & Kober, J. Explorllm: Guiding exploration in reinforcement learning with large language models (2025). URL <https://arxiv.org/abs/2403.09583>. <2403.09583>.
- <span id="page-17-17"></span>[52] Bakan, A., Meireles, L. M. & Bahar, I. Prody: Protein dynamics inferred from theory and experiments. Bioinformatics 27, 1575–1577 (2011). URL <https://doi.org/10.1093/bioinformatics/btr168>.
- <span id="page-17-18"></span>[53] Rosenzweig, A. C. Metallochaperones: Bind and deliver. Chemistry & Biology 9, 673–677 (2002). URL <https://www.sciencedirect.com/science/article/pii/S1074552102001564>.
- <span id="page-17-19"></span>[54] xAI. grok-3-mini. <https://docs.x.ai/docs/guides/reasoning> (2025). URL [https://docs.x.ai/](https://docs.x.ai/docs/guides/reasoning) [docs/guides/reasoning](https://docs.x.ai/docs/guides/reasoning). Artificial Intelligence Language Model.
- <span id="page-17-20"></span>[55] OpenAI. Gpt-o4-mini. <https://platform.openai.com/docs/models/o4-mini> (2025). URL [https:](https://platform.openai.com/docs/models/o4-mini) [//platform.openai.com/docs/models/o4-mini](https://platform.openai.com/docs/models/o4-mini). Artificial Intelligence Language Model.

- <span id="page-18-0"></span>[56] OpenAI. Introducing openai o3 and o4-mini. [https://openai.com/index/](https://openai.com/index/introducing-o3-and-o4-mini/) [introducing-o3-and-o4-mini/](https://openai.com/index/introducing-o3-and-o4-mini/) (2025). Accessed: 2025-07-27.
- <span id="page-18-1"></span>[57] AI, M. Models overview. [https://docs.mistral.ai/getting-started/models/models\\_overview/](https://docs.mistral.ai/getting-started/models/models_overview/) (2024). Accessed: 2025-07-27.
- <span id="page-18-2"></span>[58] AI, M. Un ministral, des ministraux. <https://mistral.ai/news/ministraux> (2024). Accessed: 2025-07-27.
- <span id="page-18-3"></span>[59] OpenAI. Introducing gpt-4.1 in the api. <https://openai.com/index/gpt-4-1/> (2025). Accessed: 2025-07- 27.
- <span id="page-18-4"></span>[60] OpenAI. Hello gpt-4o. <https://openai.com/index/hello-gpt-4o/> (2024). Accessed: 2025-07-27.
- <span id="page-18-5"></span>[61] OpenAI. Gpt-4o system card. <https://openai.com/index/gpt-4o-system-card/> (2024). Accessed: 2025-07-27.
- <span id="page-18-6"></span>[62] Grattafiori, A. et al. The llama 3 herd of models (2024). URL <https://arxiv.org/abs/2407.21783>. [2407.](2407.21783) [21783](2407.21783).
- [63] AI, M. Llama-3.2-3b (1b and 3b multilingual llm, instruction-tuned). [https://huggingface.co/meta-llama/](https://huggingface.co/meta-llama/Llama-3.2-3B) [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) (2024). Accessed: 2025-07-27.
- <span id="page-18-7"></span>[64] Song, F. et al. Well begun is half done: Low-resource preference alignment by weak-to-strong decoding (2025). URL <https://arxiv.org/abs/2506.07434>. <2506.07434>.
- <span id="page-18-8"></span>[65] Gallifant, J. et al. Peer review of gpt-4 technical report and systems card. PLOS Digital Health 3, e0000417 (2024). URL <https://doi.org/10.1371/journal.pdig.0000417>.
- [66] Wolfe, R. et al. Laboratory-scale ai: Open-weight models are competitive with chatgpt even in low-resource settings. In The 2024 ACM Conference on Fairness, Accountability, and Transparency, FAccT '24, 1199–1210 (ACM, 2024). URL <http://dx.doi.org/10.1145/3630106.3658966>.
- <span id="page-18-9"></span>[67] Lande, D. & Strashnoy, L. GPT Semantic Networking: A Dream of the Semantic Web––The Time Is Now (Engineering, Kyiv, 2023). Monograph.
- <span id="page-18-10"></span>[68] Jiang, A. Q. et al. Mistral 7b (2023). URL <https://arxiv.org/abs/2310.06825>. <2310.06825>.
- <span id="page-18-11"></span>[69] Touvron, H. et al. Llama 2: Open foundation and fine-tuned chat models (2023). URL [https://arxiv.org/](https://arxiv.org/abs/2307.09288) [abs/2307.09288](https://arxiv.org/abs/2307.09288). <2307.09288>.
- <span id="page-18-12"></span>[70] Sapkota, R., Raza, S. & Karkee, M. Comprehensive analysis of transparency and accessibility of chatgpt, deepseek, and other sota large language models (2025). URL <https://arxiv.org/abs/2502.18505>. <2502.18505>.
- <span id="page-18-13"></span>[71] Jumper, J. et al. Highly accurate protein structure prediction with alphafold. Nature 596, 583–589 (2021). URL <https://doi.org/10.1038/s41586-021-03819-2>.
- <span id="page-18-14"></span>[72] Abramson, J. et al. Accurate structure prediction of biomolecular interactions with alphafold 3. Nature 630, 493–500 (2024). URL <https://doi.org/10.1038/s41586-024-07487-w>.
- <span id="page-18-15"></span>[73] Chandonia, J.-M. et al. Scope: improvements to the structural classification of proteins – extended database to facilitate variant interpretation and machine learning. Nucleic Acids Research 50, D553–D559 (2022). URL <https://doi.org/10.1093/nar/gkab1054>.
- <span id="page-18-16"></span>[74] Rives, A. et al. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. Proceedings of the National Academy of Sciences 118, e2016239118 (2021). URL [https:](https://doi.org/10.1073/pnas.2016239118) [//doi.org/10.1073/pnas.2016239118](https://doi.org/10.1073/pnas.2016239118). Doi: 10.1073/pnas.2016239118.
- [75] Lin, Z. et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science 379, 1123–1130 (2023). URL <https://doi.org/10.1126/science.ade2574>. Doi: 10.1126/science.ade2574.
- <span id="page-18-17"></span>[76] Hayes, T. et al. Simulating 500 million years of evolution with a language model. Science 387, 850–858 (2025). URL <https://doi.org/10.1126/science.ads0018>. Doi: 10.1126/science.ads0018.
- <span id="page-18-18"></span>[77] Wu, R. et al. High-resolution de novo structure prediction from primary sequence. bioRxiv (2022). URL [https:](https://www.biorxiv.org/content/early/2022/07/22/2022.07.21.500999) [//www.biorxiv.org/content/early/2022/07/22/2022.07.21.500999](https://www.biorxiv.org/content/early/2022/07/22/2022.07.21.500999). [https://www.biorxiv.org/](https://www.biorxiv.org/content/early/2022/07/22/2022.07.21.500999.full.pdf) [content/early/2022/07/22/2022.07.21.500999.full.pdf](https://www.biorxiv.org/content/early/2022/07/22/2022.07.21.500999.full.pdf).
- <span id="page-18-19"></span>[78] Joosten, R. P. et al. A series of PDB-related databases for everyday needs. Nucleic Acids Research 39, D411–D419 (2011). URL <https://doi.org/10.1093/nar/gkq1105>. Journal Article.
- <span id="page-18-20"></span>[79] Micsonai, A. et al. Bestsel: a web server for accurate protein secondary structure prediction and fold recognition from the circular dichroism spectra. Nucleic Acids Research 46, W315–W322 (2018). URL [https://doi.org/](https://doi.org/10.1093/nar/gky497) [10.1093/nar/gky497](https://doi.org/10.1093/nar/gky497).

## Supplementary Materials

# Swarms of Large Language Model Agents for Protein Sequence Design with Experimental Validation

Fiona Y. Wang Di Sheng Lee David L. Kaplan Markus J. Buehler3,# Corresponding author: # mbuehler@mit.edu

The PDF file includes: Figures S1 to S5

#### PART 1: Your Role and Task

Design Goal: form local alpha helices using hydrophilic residues like serine, threonine, aspartate, glutamate, and others

Position: 12 Current residue: E

Current full sequence: EEEDEQEEQAAEAASEES

#### Decision Rules:

- 1. Most importantly, consider the design goal: form local alpha helices using hydrophilic residues like serine, threonine, aspartate, glutamate, and others
- 2. Learn from the memory history, context of the current position, sequence and spatial neighbors, and structure and energy feedback to inform decision on no mutation or mutation
- 3. Consider fundamental folding principles
- 4. Maintain sequence diversity to avoid repetitive residues but still consider residues mentioned in the design goal

#### Fundamental Folding Principles:

- Minimize disruption to secondary structures
- Favor compact, stable folding with low Rosetta energy
- Prefer conservative mutations unless necessary
- Consider hydrophobic core stability and surface accessibility

### Your Task:

Choose the best amino acid for position 12 to achieve the design goal: form local alpha helices using hydrophilic residues like serine, threonine, aspartate, glutamate, and others considering the above decision rules and fundamental folding principles

Figure S1. Example input prompt (Part 1: Role and Task) given to one LLM agent.

```
PART 2: Local Neighborhood Context
Secondary structure:
- Position 12 secondary structure: Turn/bend/coil (T/S/-)
- Analysis method: DSSP (high confidence)
- Turn/bend/coil/loop
Linear neighbors: 2 N-terminus neighbor(s): AA, 2 C-terminus neighbor(s): AA
Spatial neighbors from distance matrix within 8.0 Å: 8 total
Detailed spatial neighbors sorted by distance:
- Position 13: A (distance: 3.7Å)
- Position 11: A (distance: 3.8Å)
- Position 9: Q (distance: 5.2Å)
- Position 15: S (distance: 5.2Å)
- Position 14: A (distance: 5.3Å)
- Position 10: A (distance: 5.5Å)
- Position 8: E (distance: 6.1Å)
- Position 16: E (distance: 6.4Å)
Spatial context: Not so spatially connected (8 neighbors) - likely exposed surface
Mutate or keep the current residue according to the design goal; might favor polar, charged, or functional-
ly relevant residues.
Detailed structural context from distance matrix analysis:
Structural summary:
- Compactness measured by average distance to 5 residues to the N-terminus and 5 residues to the 
C-terminus: Very compact and tightly packed environment, average distance smaller than 6 Å.
- Contact density measured by number of neighbors within 8.0 Å: Moderate contact density (8 neighbors) 
- moderately constrained, 4-8 neighbors.
- Structural flexibility measured by standard deviation of local distances:: Moderately flexible position, 
standard deviation of local distances 1.0-2.0 Å.
- Secondary structure region predicted based on distance patterns: Local geometry potentially suggests 
helical environment.
```

Figure S2. Example input prompt (Part 2: Local Neighborhood Context) given to one LLM agent.

| PART 3: Memory History and Analysis<br>Global Patterns from All Agents from Memory:<br>- Total iterations completed: 40<br>- Overall acceptance rate: 40.0%<br>- Recent acceptance rate in the last 5 iterations:<br>40.0%<br>- Energy trend: improving<br>Previous iteration outcomes:<br>- Accepted iterations and energies: ['Iter35(238.8)',<br>'Iter38(476.2)']<br>- Rejected iterations and energies: ['Iter36(238.8)',<br>'Iter37(238.8)', 'Iter39(476.2)']<br>Personal Mutations at Position 12 and Local Neigh<br>borhood Mutations from Memory: | Local Neighborhood Mutations Analysis:<br>- Current local context: EEQAAEAASEE<br>- Pareto optimal sequences (score prioritized, then<br>energy):<br>1. Sequence: ATATEETEEEQEEESSEE<br>Score: 20.00, Energy: 238.8<br>- Worst rejected sequences to avoid:<br>• Sequence: EEEDEQEEQAAEAASEES<br>Energy: 476.2, Reason: energy and structure<br>worsened<br>• Sequence: ATATEETEEEQEEESSEE<br>Energy: 238.8, Reason: energy worsened<br>• Sequence: ATATEETEEEQEEESSEE<br>Energy: 238.8, Reason: energy and structure<br>worsened<br>- Local mutation patterns near position 12:<br>Personal Mutations Analysis at Position 12: |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Iter 39: E->T REJECTED for reason of energy and<br>structure worsened<br>Iteration details: Energy: 476.2, Score: 0.0, 14<br>mutations<br>Total energy: 476.2                                                                                                                                                                                                                                                                                                                                                                                             | - Acceptance trend (first half vs second half): stable<br>- Common rejection reasons: {'energy is too high,<br>consider protein folding principles': 3}<br>- Your rejected patterns: ['E->S (18.7%)']<br>- Recent trend in the last 10 iterations: declining                                                                                                                                                                                                                                                                                                                                                                    |
| Iter 36: E->S REJECTED for reason of energy<br>worsened<br>Iteration details: Energy: 238.8, Score: 20.0, 11<br>mutations<br>Total energy: 238.8                                                                                                                                                                                                                                                                                                                                                                                                          | - Residues that were successfully accepted at this<br>position and the number of times they were accept<br>ed: {'E': 2, 'T': 1}<br>- Residues that were rejected at this position and the<br>number of times they were rejected: {'T': 2, 'S': 3, 'E':<br>2}                                                                                                                                                                                                                                                                                                                                                                    |
| Iter 35: T->E ACCEPTED for reason of energy<br>worsened<br>Iteration details: Energy: 238.8, Score: 20.0, 15<br>mutations<br>Total energy: 238.8<br>Iter 34: T->E REJECTED for reason of unknown<br>reason<br>Total energy: N/A                                                                                                                                                                                                                                                                                                                           | - Current residue E performance: 2/4 accepted<br>- Average total energy with E: 585.85<br>Personal and Neighborhood Energy Analysis from<br>Memory:<br>- Energy worsening iterations to avoid:<br>• Iter 35: energy 238.8, 15 mutations, accepted<br>• Iter 36: energy 238.8, 11 mutations, rejected<br>• Iter 37: energy 238.8, 13 mutations, rejected                                                                                                                                                                                                                                                                         |
| Iter 33: T->E REJECTED for reason of unknown<br>reason<br>Total energy: N/A<br>Iter 32: E->T ACCEPTED for reason of accepted                                                                                                                                                                                                                                                                                                                                                                                                                              | Mutation Recommendations for E at Position 12 from<br>Memory:<br>- Highly recommended (and past success rate): ['L<br>(1.00 (1/1))']<br>- Recommended (and past success rate): ['Q (0.71                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Energy change: 648.03<br>Iter 28: E->S REJECTED for reason of unknown<br>reason<br>Total energy: N/A                                                                                                                                                                                                                                                                                                                                                                                                                                                      | (10/14))', 'A (0.56 (14/25))']<br>- Avoid (and past success rate): ['C (0.00)', 'D (0.00<br>(0/2))', 'F (0.00)']<br>One-line Summary from Memory:<br>Multiple energy rejections - use energy-favorable                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Iter 27: E->S REJECTED for reason of unknown<br>reason<br>Total energy: N/A                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | mutations (A->V, S->A, E->A) Structure quality<br>issues - avoid large sidechains in constrained<br>regions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Iter 26: S->E ACCEPTED for reason of accepted<br>Energy change: 932.94<br>Iter 25: S->T REJECTED for reason of unknown<br>reason<br>Total energy: N/A                                                                                                                                                                                                                                                                                                                                                                                                     | Your Relevant Past Experiences and Energy Chang<br>es at Position 12:<br>- Iter 39: E->T → rejected<br>- Iter 36: E->S → rejected<br>- Iter 32: E->T → accepted (ΔE:<br>648.0268157943258)                                                                                                                                                                                                                                                                                                                                                                                                                                      |

Figure S3. Example input prompt (Part 3: Memory History and Analysis) given to one LLM agent.

#### PART 4: Design Goal and Energy Analysis

#### Design Goal Evaluation:

- Current design goal score: 0.0/100, POOR: Limited match to design goal
- Key aspects:
- helix\_structure: Limited helix content: 0.0% structure, 72.2% helix-forming residues
- sequence\_diversity: Limited diversity: 4/20 amino acids
- score\_normalization: Score normalized based on triggered conditions: alpha\_helices
- Structure and Design Goal Score Recommendations for E at Position 12: - Increase helix-forming residues (A,L,E,M,K) for better alpha helix structure
- Increase amino acid diversity

Figure S4. Example output (Part 4: Design Goal and Energy Analysis) given to one LLM agent.

## reasoning:

From the design goal of favoring alpha-helix formation, glutamate (E), the current residue, already aligns with this intention due to its alpha-helix promoting nature. The local environment and memory history do not strongly suggest that a mutation would improve the energy or structure. Surrounding other energetic considerations, stability, and structural constraints, maintaining glutamate (E) is a reasonable choice.

proposed\_value: E

Figure S5. Example output with reasoning and proposed mutation from one LLM agent.