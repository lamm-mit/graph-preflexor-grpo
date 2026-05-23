#### REGULAR ARTICLE

# **Polymatic:** a generalized simulated polymerization algorithm for amorphous polymers

Lauren J. Abbott · Kyle E. Hart · Coray M. Colina

Received: 8 October 2012/Accepted: 5 January 2013/Published online: 29 January 2013 © Springer-Verlag Berlin Heidelberg 2013

**Abstract** This work presents a generalized structure generation methodology for amorphous polymers by a simulated polymerization technique and 21-step molecular dynamics equilibration, which is particularly effective for high- $T_g$  polymers. The essential framework and parameters of the techniques and algorithms are described in detail, and example input scripts are provided for use with the freely available *Polymatic* simulated polymerization code and LAMMPS molecular dynamics package. The capabilities of the methods are examined through application to six linear, glassy polymers ranging in functionality, polarity, and rigidity. Validation of the methodology is provided by comparison of the simulations and experiments for a variety of structural, adsorption, and thermal properties, all of which showed excellent agreement with available experimental data.

**Keywords** Molecular simulations · Glassy polymers · Simulated polymerization · Equilibration

Published as part of the special collection of articles derived from the conference: Foundations of Molecular Modeling and Simulation 2012.

**Electronic supplementary material** The online version of this article (doi:10.1007/s00214-013-1334-z) contains supplementary material, which is available to authorized users.

L. J. Abbott · K. E. Hart · C. M. Colina (⋈)
Department of Materials Science and Engineering,
The Pennsylvania State University, University Park,
PA 16802, USA

e-mail: colina@matse.psu.edu

#### 1 Introduction

In addition to excellent mechanical properties, glassy polymers possess superior permeation and separation properties due to high free volume, which make them desirable for a variety of applications including gas separation, liquid purification, and catalysis, among others [1, 2]. Glassy polymers, by definition, are amorphous and exhibit no long-range order, so their structures are not well understood. Furthermore, characterization techniques often provide indirect measures of properties of interest through models that may be at the limits of their applicability [3–5]. This may include, for example, application of Brunauer-Emmett-Teller (BET) theory [6] or non-local density functional theory (NLDFT) [7] to derive apparent surface areas or pore size distributions from gas adsorption isotherms, or the use of positron annihilation lifetime spectroscopy (PALS) [8] to estimate void sizes from the lifetimes of positronium within the material. Therefore, molecular simulations can provide necessary insight into molecular-level structure and phenomena complementary to experimental techniques in order to improve understanding of the structure and properties of glassy polymers [9, 10]. Moreover, through the use of appropriate methodologies, simulations can enable the prediction of performance of new polymeric materials to guide future design efforts [9].

The first task for simulations is obtaining an initial structure of the system. For well-ordered and crystalline materials, on the one hand, reference structures can be obtained in a rather straightforward manner by methods such as X-ray crystallography. For amorphous materials, on the other hand, this is a non-trivial problem, requiring generation of an ensemble of structures that provide the proper statistical description of material properties of

interest, which is discussed in more detail in a review by Gelb [\[5](#page-17-0)], as well as the references therein. Specifically, achieving polymer models near mechanical equilibrium can be difficult due to their amorphous nature, a problem that is exacerbated as chain lengths increase and structures become stiffer and bulkier. Many techniques for generation of amorphous polymers are based on the influential Monte Carlo method proposed by Theodorou and Suter [\[11](#page-17-0)]. By this approach, an initial guess at the structure is obtained by placing the polymer chain in a periodic box bond by bond with chain conformation probabilities described by rotational isomeric state theory [[12\]](#page-17-0). Although the original work was demonstrated for the rather simple polypropylene, variations of the method have since been applied successfully to a number of more complex polymers [\[13–17](#page-17-0)]. In addition, alternative structure generation methods have become more abundant and sophisticated with the increasing need to simulate a vast array of polymeric materials. These include a broad range of Monte Carlo [[18–22\]](#page-17-0) and coarse-grain techniques [[23–25\]](#page-17-0), which allow for fast and efficient generation of equilibrated polymeric systems.

With the continuous improvement of computational resources, molecular dynamics has become a more dominant component of structure generation techniques. Namely, of frequent use is a simulated polymerization approach, where polymeric systems are generally constructed from a periodic box of monomers or repeat units that are ''polymerized'' over the course of extensive molecular dynamics simulations [[26–34\]](#page-17-0). It should be noted that these methods do not actually simulate the formation or breaking of chemical bonds, unlike reactive force fields [[35–38\]](#page-17-0). Simulated polymerization techniques provide an interesting alternative to other methods because they have the potential to resolve a number of difficulties typically faced during structure generation of amorphous polymers. First, the algorithm can be applied in nearly the same fashion to polymers of any connectivity, whether they are linear or networked. Second, it alleviates many of the steric hindrance issues often experienced for bulky and rigid structures. Third, it is possible for these methods to make qualitative conclusions about important processing parameters as the approach can mimic, to some extent, the synthetic route. Therefore, further development of these methods to be more general and accessible is imperative for effective simulations of glassy polymers.

After obtaining an appropriate initial guess structure, an equally important step in the structure generation of amorphous polymers is equilibration to achieve a relaxed final structure at a given set of conditions (e.g., 300 K and 1 bar). High-T<sup>g</sup> polymers, in particular, pose several difficulties in this regard due to their slow dynamics. Moreover, since structure generation of high-T<sup>g</sup> polymers is often

Many methods have been published in the literature for the structure generation of amorphous materials, and some open source [[40–42\]](#page-17-0) and commercial [\[43](#page-17-0)] software products exist to aid in these tasks. However, much of this work is specialized for specific materials and cases, and detailed algorithms are not always described. Therefore, in this work, a general structure generation methodology is presented via a simulated polymerization and 21-step molecular dynamics equilibration, with explicit description of its use as applied with the Polymatic code available online [\[44](#page-17-0)]. The outline of the paper is as follows: First, the essential framework and parameters of the simulated polymerization and 21-step equilibration are discussed in detail. Then, the applicability of the techniques is illustrated for six linear, glassy polymers covering a wide range of functionality, polarity, and rigidity: polystyrene (PS), poly(methyl methacrylate) (PMMA), poly(ethylene terephthalate) (PET), polycarbonate (PC), polyetherimide (PEI), and a polymer of intrinsic microporosity (PIM-1) [[45\]](#page-17-0), the chemical structures of which are given in Fig. [1.](#page-2-0) Through application of the presented methodology, important parameters and conditions of the simulated polymerization and 21-step equilibration are examined as applied to the example systems. Finally, the methods are validated by comparison of the simulations with available experimental data.

## 2 Simulated polymerization

A primary benefit of simulated polymerization techniques is their broad applicability to amorphous polymers with a variety of structural connectivity, which is illustrated by the large number of systems to which it has been applied [\[26–34](#page-17-0)]. Generally, this approach consists of the ''polymerization'' of a periodic system of monomers or repeat units, but can also include, for example, the addition of crosslinks to fully formed chains [[28\]](#page-17-0). In much the same manner, the method can handle monomers with a

<span id="page-2-0"></span>Fig. 1 Chemical structures of the polymers in this work: poly(methyl methacrylate) (PMMA), poly(ethylene terephthalate) (PET), polystyrene (PS), polycarbonate (PC), a polymer of intrinsic microporosity (PIM-1), and polyetherimide (PEI)

functionality of two to obtain linear polymers, as well as monomers with functionalities of three or more in the formation of branched and networked systems. Also accessible are polymers with a simple single backbone or a more complex ladder backbone, such as PIM-1 studied here. Moreover, the generality of the approach lends well to simulations over a range of different size scales through application to either atomistic [26–28, 31–34] or coarsegrained models [29, 30].

Simulated polymerization methods offer a few important advantages in the structure generation of glassy polymers. In particular, with other approaches, steric overlap issues are often experienced for high- $T_{\rm g}$  polymers due to their bulky structures, which can be difficult to overcome with energy minimization and molecular dynamics simulations. As a result, these types of polymers are usually constructed at very low densities ( $\sim 0.1 \text{ g cm}^{-3}$ ) and achieve shorter chain lengths (20 or fewer repeat units) [14, 16, 17, 39, 46]. A simulated polymerization method, on the other hand, can alleviate these issues, since all repeat units are present in the simulation box from the start. In addition, simulated polymerization approaches can be tailored to mimic synthetic routes, to some extent, to provide insight into the effect of different processing conditions. Examples of this can be found for simulations varying temperature, monomer concentration, monomer reactivities, and crosslinking degrees [28, 31, 34, 47, 48]. Although the algorithm is applied typically in a step-growth fashion, it has also been developed to capture free radical polymerization [34].

Furthermore, multistep processes can be examined, such as the generation of crosslinked polymers by first forming linear polymer chains followed by a second reaction to form crosslinks. Other possibilities include polymerization in confined spaces, or around fillers as in nanocomposites or mixed matrix membranes. These examples illustrate the flexibility possible with simulated polymerization methods.

This work is focused on the development of a generalized simulated polymerization algorithm that can be implemented for many amorphous polymers. The algorithm is based on initial work performed on a hypercrosslinked polymer [31], but has been updated and extended for use with a variety of other systems. In this section, the general form of the algorithm is presented, a flowchart for which is given in Fig. 2. A number of parameters and constraints for precise control over the simulated polymerization algorithm are discussed thereafter. Furthermore, the implementation of the algorithm in this work is briefly described with example scripts given in the Supplemental Material for use with the Polymatic code available online [44]. The code and detailed descriptions should provide a sufficient base for use in simulations of a wide range of polymeric materials that can be tailored to the needs of the specific systems of interest.

#### 2.1 Algorithm

The basic structure of the simulated polymerization algorithm, as utilized in this work, is as follows:

<span id="page-3-0"></span>Fig. 2 Flowchart of the Polymatic simulated polymerization algorithm. The start and stop steps are displayed in ovals, processing steps in rectangles, and conditional steps in diamonds. A polymerization

step is enclosed in a dashed box, while a polymerization cycle is outlined in a dotted box. EM energy minimization, MD molecular dynamics

- 1. Initial structures are obtained by a random packing of repeat units into a box under periodic boundary conditions at an initial density, q0. The chemical structures of the repeat units are defined in the way they would exist in the polymer with reactive atoms identified as those to be bonded during polymerization steps. The system is initialized for polymerization, such as by the optional addition of artificial charges, qpolym, on reactive atoms.
- 2. A polymerization step is performed: (a) The pair of reactive atoms closest in proximity that meets all

bonding criteria is selected. (b) A bond is formed between the atoms such that the proper polymeric structure is obtained, and an energy minimization is performed to relax the newly formed bond. If artificial charges were added, they are removed from the bonded pair at this time. (c) If no pair meeting all bonding criteria is found, a molecular dynamics simulation is performed and a polymerization step attempted again. This is repeated up to Mmax times until a bond is accepted. (See the dashed box outlining a polymerization step in Fig. 2.)

- 3. Ncyc polymerization steps are performed according to step 2 to compose one cycle. At the end of the cycle, a short molecular dynamics step (NVT or NPT) is carried out to relax any remaining stresses in the system, as well as to allow structural rearrangement of the configuration. Multiple equilibration types can be defined and alternated throughout the process, such as by running NVT steps with the inclusion of an NPT step every Nnpt cycles. (See the dotted box outlining a polymerization cycle in Fig. [2](#page-3-0).)
- 4. Cycles of polymerization steps (step 3) are successively repeated until Btot bonds are formed or until no pair meeting the bonding criteria is identified within the Mmax molecular dynamics simulations (see step 2c). To finalize the system, all artificial charges are removed from any remaining reactive atoms.

## 2.2 Details

## 2.2.1 Connectivity

A key component of the simulated polymerization algorithm is the definition of a ''polymerization step'' (see step 2 in the algorithm and the dashed box in Fig. [2\)](#page-3-0). For ease of implementation, repeat units are defined as they are to appear in the polymer structure. Thus, no atoms are required to be added or removed from the system, while only new bonded terms (bonds, angles, dihedrals, etc.) are added to the connectivity definition. Reactive atoms are identified by unique atom types, between which the new bond is formed. It is important to note that the simulated polymerization step does not have to represent the actual synthesis of the polymer, but only provide the correct chemical structure of the polymerized system.

While only one pair of reactive atoms is identified for new bond formation, the simulated polymerization algorithm is easily tailored to work for more complex connectivity requiring the creation of two or more bonds within a single polymerization step. A two-bond polymerization has been previously utilized, for example, in simulations of a crosslinked benzocyclobutene/styrene copolymer [\[28](#page-17-0)], where chains were bridged by an eightmember ring formed from two crosslinks. Two bonds are also required for ladder polymers, such as PIM-1 studied here.

## 2.2.2 Bonding criteria

Several criteria can be set to determine if a bond is added between a reactive pair. Since the chemical reactions are not directly considered in these classical simulations, bonding criteria are imposed to prevent unreasonable or unrealistic structural configurations and high levels of stress upon bond formation, which are unlikely to be adequately relaxed during energy minimization and molecular dynamics simulations. These are applied in a way to ensure that the simulated ''reactions'' in these types of approaches result in reasonable structures. Test cases should be performed for different types of systems to determine the appropriateness of potential bonding situations and the effectiveness of energy minimization or molecular dynamics in relieving potentially unrealistic bonds or structures. This is particularly important for very rigid or networked systems, since there are fewer degrees of freedom as the polymerization progresses. Additionally, bonding criteria can be defined to provide some degree of bias in bond formations, such as the relative reactivity volume criterion implemented by Jang et al. [\[34](#page-17-0)] to model reactivity ratios in copolymers, or different reaction probabilities for primary and secondary reactions in epoxy resins as imposed by Liu et al. [\[30](#page-17-0)].

Several examples of bonding criteria are implemented in the code utilized in this work. The first is a cutoff radius, rcutoff, which defines the maximum distance two reactive atoms can be from one another to allow bond formation (cf. capture radius, reaction cutoff, reaction radius). These distances are typically set between 4 and 10 A˚ [\[26](#page-17-0), [28](#page-17-0), [31](#page-17-0)] , but are sometimes increased up to larger values near the end of polymerization to allow for a greater completion [\[27](#page-17-0), [33](#page-17-0)]. Care must be taken when choosing large cutoffs to prevent the formation of unreasonable bond lengths that would not be properly minimized. The second type of bonding criteria is orientational or spatial. For example, angle conditions were imposed during structure generation of a crosslinked benzocyclobutene/styrene copolymer [\[28](#page-17-0)], in which potentially bonded atoms were required to have angles within a range around the equilibrium value. In this work, optional angle checks between vectors and planes defined by atoms within the system are implemented in the provided code. Lastly, a third bonding criterion is imposed to prevent intramolecular bonds during formation of linear polymers. Without this restriction, the construction of networked and looped structures is allowed.

# 2.2.3 Structural relaxation

In order to maintain a relaxed structure throughout the simulated polymerization, energy minimization and molecular dynamics steps are included. While energy minimizations provide initial relaxation of the newly formed bond, molecular dynamics offers a more thorough relaxation (and possible compression/decompression) of the structure to minimize any large stresses introduced by bond formations, as well as to allow fluctuations of the structural configuration. Due to the very slow relaxation

times of high-T<sup>g</sup> polymers, high temperatures may be necessary to allow for sufficient movement of the chains during the short molecular dynamics steps. Karayiannis et al. [[16\]](#page-17-0), for example, noted that temperatures as high as 2,000 K were needed to obtain adequate fluctuations of glassy polymers like poly(ethylene terephthalate).

The types of relaxations implemented in approaches in the literature vary greatly in frequency, length, and ensemble/conditions (i.e., NVT or NPT, temperature, pressure). In the provided code, an energy minimization is carried out after every bond formation to immediately reduce the stretched bond to an appropriate length. Molecular dynamics steps are then introduced in cycles every Ncyc bonds. For greater flexibility, an optional second type of molecular dynamics step can also be performed, which could involve the inclusion of different ensembles, conditions (temperature or pressure), or lengths. For example, in this work, NVT simulations are performed with NPT simulations incorporated every Nnpt cycles.

## 2.2.4 Density during polymerization

The density of the system during the simulated polymerization is an important parameter to control. If densities are too low, polymer fragments will exist far away from each other and generation of longer chain lengths will be prevented. On the other hand, the formation of rigid polymers at densities too high is hindered by the inability of the chains to adequately fluctuate such that the system is frozen in a frustrated dense state. As high temperatures were found to be important for achieving adequate fluctuations of glassy polymers during molecular dynamics steps, Karayiannis et al. [[16\]](#page-17-0) also found that densities of \*0.7 g cm-<sup>3</sup> were necessary to provide enough free volume for sufficient mobility of the chains. Here, the density of the system during the simulated polymerization can be controlled by the initial packing density and inclusion of NPT molecular dynamics during cycle equilibrations. The temperature, pressure, length, and frequency of these NPT runs determine the density of the system throughout the simulated polymerization.

## 2.2.5 Artificial charges

The simulated polymerization can be biased to encourage quicker and more complete bonding by adding artificial charges to reactive atoms. Opposite charges on reactive atoms can provide a slight association between chain ends during the molecular dynamics simulations, instead of relying solely on their random passing. The charges should be large enough to induce some long-range interaction, but not so large that unrealistic structures can result. The magnitude of these charges depends on the material being studied and the other charges within the system; test simulations can determine appropriate values for artificial charges, as is discussed later. In this work, equal and opposite artificial charges (± qpolym) are added to the charges of the two reactive atom types to maintain a charge-neutral system. Lastly, the artificial charges are included only during the simulated polymerization and are removed from the pair immediately after bond formation to prevent unwanted influence on the structure of the polymer.

## 2.3 Code

In this work, the Polymatic simulated polymerization algorithm is implemented utilizing a collection of in-house scripts [\[44](#page-17-0)] and the Large-scale Atomic/Molecular Massively Parallel Simulator (LAMMPS) [\[49](#page-17-0), [50](#page-18-0)]. A bash script controls the main polymerization loop, from which polymerization steps, energy minimizations, and molecular dynamics simulations are called, as defined in the above algorithm and the flowchart in Fig. [2.](#page-3-0) A polymerization step is carried out with a perl script by searching the system for the closest reactive pair and checking the required bonding criteria. On bond formation, the necessary additions to the LAMMPS data file are made to include new bonds, angles, dihedrals, and impropers. Definition of the reactive atoms, bonding criteria, bonds to form, and artificial charges are given in an input file for the perl script. LAMMPS is also called from the bash loop script to perform energy minimization and molecular dynamics simulations, parameters for which are provided in LAMMPS input scripts.

Example input scripts for the Polymatic simulated polymerization code are provided in Appendix A of the Supplemental Material. Additionally, a perl script to randomly pack molecules into a periodic box, a perl script to perform a polymerization step, and a sample bash script for controlling the main polymerization loop are available online [[44\]](#page-17-0). Although the codes are written for use with LAMMPS, the algorithm is general and can be easily extended for use with other file types and software packages with minor modifications.

## 3 21-Step equilibration

For high-T<sup>g</sup> polymers, achieving an equilibrated system can be difficult due to very long relaxation times. Additionally, since they are polymerized at lower densities, the structures must also be compressed to experimental-like densities. Hofmann et al. [[14\]](#page-17-0) presented a protocol consisting of NVT and NPT molecular dynamics steps for use with rigid systems, which incorporated simulations at high temperatures (600 K) and pressures up to 5 9 10<sup>4</sup> bar to

effectively overcome large energy barriers. The final simulation finished at 300 K and 1 bar to obtain an equilibrated system; however, one iteration of this cycle was found to be insufficient for consistently reaching densities comparable with experimental values. Karayiannis et al. [\[16](#page-17-0)] made a key observation that the abrupt pressure jump of several orders of magnitude (104 to 10<sup>0</sup> bar) in the final step resulted in inconsistent stress tensor components and high levels of residual stress within the system. Likewise, Larsen et al. [\[39](#page-17-0)] noted large variations in the final densities obtained using the Hofmann scheme that depended heavily on the maximum pressure applied during the compression. To adequately alleviate the large stresses induced by the strong compressions, Karayiannis et al. suggested a ''milder'' decompression with small step changes instead of one large jump.

In this work, an updated version of the Hofmann scheme, as implemented by Larsen et al., has been examined and generalized. The 21-step molecular dynamics protocol incorporates NVT and NPT simulations at high temperatures, as well as a gradual compression and decompression to alleviate high levels of stress associated with large pressure jumps and corresponding large box size changes, as suggested by Karayiannis et al. In its general form, the 21-step equilibration is effective for glassy polymers to achieve efficient relaxation and compression of the structures to realistic and consistent final densities. The format and parameters of the 21-step procedure are described below.

## 3.1 Format

The 21-step equilibration consists of seven cycles of three molecular dynamics steps each: (1) NVT at a high temperature, Tmax, (2) NVT at a final temperature, Tfinal, and (3) NPT at Tfinal. The pressure imposed during the NPT steps is gradually ramped up to a maximum pressure, Pmax, over the first three cycles (steps 1–9) in longer simulations (50–100 ps each) to allow adequate fluctuation and relaxation of the structure. Then, the pressure is gradually stepped back down to a final pressure, Pfinal, over the last four cycles (steps 10–20) in shorter simulations (5–10 ps each) to control the large pressure jumps and prevent unwanted stress in the system. These decompression steps add only a short amount of simulation time to the equilibration, but greatly improve the consistency of the final densities achieved [\[39\]](#page-17-0). The ensemble, conditions, and length of each step are given in Table 1. Additionally, the stepwise compression and decompression is illustrated by a trace of the density during the equilibration of PC, shown in Fig. [3](#page-7-0). (A LAMMPS input script for the 21-step equilibration is provided in Appendix B of the Supplemental Material.)

Table 1 21-step molecular dynamics equilibration scheme

| Step | Ensemble | Conditions          | Length (ps) |
|------|----------|---------------------|-------------|
| 1    | NVT      | Tmax                | 50          |
| 2    | NVT      | Tfinal              | 50          |
| 3    | NPT      | Tfinal, 0.02 9 Pmax | 50          |
| 4    | NVT      | Tmax                | 50          |
| 5    | NVT      | Tfinal              | 100         |
| 6    | NPT      | Tfinal, 0.6 9 Pmax  | 50          |
| 7    | NVT      | Tmax                | 50          |
| 8    | NVT      | Tfinal              | 100         |
| 9    | NPT      | Tfinal, Pmax        | 50          |
| 10   | NVT      | Tmax                | 50          |
| 11   | NVT      | Tfinal              | 100         |
| 12   | NPT      | Tfinal, 0.5 9 Pmax  | 5           |
| 13   | NVT      | Tmax                | 5           |
| 14   | NVT      | Tfinal              | 10          |
| 15   | NPT      | Tfinal, 0.1 9 Pmax  | 5           |
| 16   | NVT      | Tmax                | 5           |
| 17   | NVT      | Tfinal              | 10          |
| 18   | NPT      | Tfinal, 0.01 9 Pmax | 5           |
| 19   | NVT      | Tmax                | 5           |
| 20   | NVT      | Tfinal              | 10          |
| 21   | NPT      | Tfinal, Pfinal      | 800         |

The scheme provided here is an updated and generalized form of that given in Ref. [\[39\]](#page-17-0)

The 21-step equilibration, as described, provides a general format for efficient relaxation and compression of glassy polymers. In particular, the high-temperature and high-pressure steps allow large energy barriers to be overcome to speed up the normally lengthy relaxation of glassy polymers (which may never truly reach equilibrium). This scheme is particularly important for effective simulations of high-T<sup>g</sup> polymers, as studied here. It is important to note that the gradual compression and decompression of the equilibration is critical, but does not necessitate the exact steps as presented in this work. However, the format as described has been used to produce realistic and consistent simulations of hypercrosslinked polymers [[31\]](#page-17-0), polymers of intrinsic microporosity [[39,](#page-17-0) [51](#page-18-0)], and organic molecules of intrinsic microporosity [\[52](#page-18-0)]. It is shown to be equally effective for a range of glassy polymers in this work.

## 3.2 Parameters

## 3.2.1 Final conditions

The thermodynamic state of the final polymer structure obtained after application of the 21-step molecular dynamics equilibration is determined by the temperature

<span id="page-7-0"></span>Fig. 3 The bulk density of a polycarbonate box throughout the 21-step equilibration. The density jumps are marked by the pressure imposed during the NPT simulations as a fraction of the maximum pressure, Pmax = 5 9 10<sup>4</sup> bar. The trace of the density illustrates the gradual compression and decompression performed during the 21-step equilibration before reaching the predicted final density at the final pressure, Pfinal = 1 bar

and pressure during the final step, Tfinal and Pfinal. It should be stressed that the density of the final system is not predefined, but is instead achieved intrinsically during the last NPT simulation at Tfinal and Pfinal. This provides predictive structure generation at any set of conditions desired.

## 3.2.2 Maximal conditions

The equilibration is also defined by its maximum temperature and pressure, Tmax and Pmax, which allow the structure to adequately overcome large energy barriers and obtain an equilibrated state. Tmax should be chosen well above the glass transition temperature of the polymer. High temperatures provide sufficient energy to the system to allow for adequate fluctuations of the structure during molecular dynamics simulations. If the temperature is too low, the motions of the polymer chains will be limited and equilibrium may not be reached. Similarly, Pmax must be sufficiently high to force the rigid structures to reach consistent and realistic final densities. Previous work with polymers of intrinsic microporosity, for example, found that the final densities obtained after application of the 21-step scheme were mostly independent of the value of Pmax used, but were most consistent at the highest pressure of 5 9 10<sup>4</sup> bar [\[39](#page-17-0)].

# 4 Simulation details

# 4.1 Models

Molecular models of all polymers except PIM-1 were described using the polymer consistent force field (PCFF)

## 4.2 Structure generation

In this work, polymeric structures were generated via the Polymatic simulated polymerization algorithm [[44\]](#page-17-0), as described above. Initial packing arrangements were obtained by random insertions of repeat units in a periodic box at an initial density of q<sup>0</sup> & 0.3–0.4 g cm-<sup>3</sup> . The number of bonds added per cycle was set to Ncyc = 10 for PS, PVAc and PMMA, and Ncyc = 5 for PET, PC, PEI, and PIM-1. The molecular dynamics steps during the cycle equilibrations were performed for 5 ps in the NVT ensemble, with NPT molecular dynamics steps every Nnpt = 3 or 5 cycles. The maximum number of bond attempts performed during a polymerization step was set to Mmax = 50, each of which was 2 ps in length. A few parameters of the simulated polymerization were varied in this work to determine their effects on the polymerization process. The temperatures of the molecular dynamics steps, TNVT and TNPT, were set between 300 and 3,000 K. Artificial charges were also added during the polymerization with values ranging from qpolym = 0.0 to 2.0 e.

The polymerized structures were subsequently subjected to the 21-step equilibration described above to allow for relaxation of the structures and compression to realistic final densities. The final parameters of the equilibration were chosen based on the final conditions of the simulation desired, which were typically Tfinal = 300 K and Pfinal = 1 bar. To allow adequate fluctuations of the system during equilibration, Tmax was chosen to be sufficiently above the glass transition temperatures of the polymers (Tmax = 600–1,000 K). For the calculation of glass transition temperatures, however, structures were generated at higher temperatures (Tfinal = 650–1,300 K and Tmax = 1,000–2,000 K) to provide a melt state for subsequent cooling. Based on previous work with high-T<sup>g</sup> polymers [\[39](#page-17-0)], Pmax was set to 5 9 10<sup>4</sup> bar in all cases to provide consistent final densities.

All energy minimizations and molecular dynamics simulations were performed with LAMMPS. A cascade of

the steepest descent and conjugate gradient algorithms was utilized during energy minimizations. Molecular dynamics simulations were carried out in the canonical (NVT) or isothermal-isobaric (NPT) ensemble with a Nose´–Hoover thermostat and barostat, a velocity Verlet integrator, and a 1 fs timestep. Coulombic terms were calculated with a cutoff of 15 A˚ , and long-range interactions were implemented using the particle–particle particle–mesh (PPPM) method [[62,](#page-18-0) [63](#page-18-0)].

## 5 Results and discussion

## 5.1 Applicability for various connectivities

To illustrate the generality of the structure generation techniques described above, six linear, glassy polymers (Fig. [1](#page-2-0)) spanning a wide range of functionality, polarity, and rigidity were studied. The Polymatic simulated polymerization code [\[44](#page-17-0)] was applied to each in the same manner with the repeat units specifically designed to provide the proper polymerized structures with the addition of one (or more) bonds and no deletion of atoms. The polymerization steps were specified by definition of the reactive atoms and bonding criteria for each polymer with only minor adjustments to the Polymatic input scripts, examples of which are provided in Appendix A of the Supplemental Material. Reactive atoms were identified by unique atom types initially, then reverted back to normal atoms types after bond formation to prevent further polymerization steps involving these atoms. A cutoff radius of 6 A˚ was defined for all systems and orientational bonding criteria were imposed for PIM-1, as discussed below. Additionally, a constraint was imposed in all polymers to prevent intramolecular bonding for the formation of loops in the linear polymers.

During the polymerization step of each polymer, the closest pair of reactive atoms meeting all bonding criteria was identified and a bond formed between them. This completed the one-bond step necessary for PS, PMMA, PET, PC, and PEI. PIM-1, however, required a two-bond polymerization step, so a second bond was added between the neighboring reactive atoms to produce the ladder backbone. Definition of this second bond was included in an additional line in the Polymatic input script. Snapshots illustrating the one- and two-bond polymerization steps of PS and PIM-1, respectively, are shown in Fig. 4 (the others are given in Fig. S2 of the Supplemental Material). From these images, it can be observed that the bonds formed have unrealistically long lengths initially, since the atoms are not able to get much closer during the simulations due to the strong repulsive van der Waals forces. For this reason, the newly formed bonds are immediately relaxed by an energy minimization to obtain a realistic structure before proceeding to the next step.

Due to the complex and rigid nature of PIM-1, additional orientational bonding criteria were implemented

Fig. 4 One- and two-bond polymerization steps illustrated for a polystyrene and b PIM-1, respectively. First, the closest pair of reactive atoms meeting all bonding criteria is identified, as shown by the dotted lines (top). Second, a bond is formed between the closest reactive pair, labeled 1 for each case. For PIM-1, a second bond is added between the adjacent pair of reactive atoms, labeled 2, to give the ladder backbone structure (middle). Third, an energy minimization is performed to relax the newly formed bonds (bottom)

during the simulated polymerization to ensure realistic structures were obtained. The first check was performed to make sure that the aromatic rings of the two chain ends to be bonded lay roughly in the same plane, as shown in Fig. 5a. As such, the angle between best-fit planes for each of the aromatic rings (atoms 1–4 and 5–8 in Fig. 5a) was required to satisfy ha\40 or ha[140. The second check made sure that the chain ends faced in approximately opposite directions, as shown in Fig. 5b. For this check, a vector was defined pointing outward from the chain ends (atoms 9–10 and atoms 12–11 in Fig. 5b), the angle between which had to satisfy hb[135. These orientational bonding criteria were chosen to ensure realistic

Fig. 5 Orientational bonding criteria imposed during polymerization of ladder polymer PIM-1: a The angle between two best-fit planes (atoms 1–4 and atoms 5–8) is checked so that the aromatic rings lie roughly in the same plane. b The angle between two vectors (atoms 9–10 and atoms 12–11) is checked so that the chains are facing in approximately opposite directions. Examples of c a properly aligned segment of PIM-1 where the backbone is planar and d an extreme bend in the ladder backbone possible if no orientational bonding criteria are imposed

The flexibility of the simulated polymerization algorithm presented here is illustrated by its implementation to a wide variety of systems through the use of simply defined polymerization steps and bonding criteria. Polymerization steps are controlled by straightforward input scripts in Polymatic, even for more complex connectivity like the ladder backbone of PIM-1. Bonding criteria have been chosen in this work to ensure the generation of realistic structures throughout the simulations. Just as a reaction during experimental polymerization would only occur under the proper conditions, imposed bonding criteria in the simulations ensure similar realistic structures are obtained, since no chemical reactions are directly considered. The extra orientational criteria utilized for PIM-1 were seen to be critical to control the realistic bonding of the more complicated ladder backbone structures. Despite the more complicated nature of these checks, they have been implemented into Polymatic in a straightforward fashion, requiring the addition of only simple commands to the input script (see Appendix A of the Supplemental Material).

While this work presented examples of some possible bonding criteria (e.g., cutoff radius, orientational constraints, and intramolecular-only bonding), countless other bonding criteria are possible to easily extend the methodology for use with other polymeric materials. For example, bonding criteria can also be designed to add probability or bias into bonding steps, such as to provide a more likely bonding between one atom pair over another [[30,](#page-17-0) [34](#page-17-0)]. It should be stressed here that while simulated polymerization approaches provide possibilities for mimicking polymer synthesis more directly, this was not the aim of the present work. Instead, the focus was the development and examination of a generalized simulated polymerization algorithm and some of the important parameters involved, specifically in obtaining longer chains of high-T<sup>g</sup> polymers. Hopefully, the provided details of this work will enable these types of approaches to be more accessible to the community to aid in future efforts to address more complicated problems, such as a more direct modeling of the synthesis process.

## 5.2 Control of the simulated polymerization

Atomistic simulations of polymers are limited by the length scales achievable with current computational resources.

<span id="page-10-0"></span>Given that polymer properties are substantially impacted by molecular weight, it is important for structure generation methods to strive for longer chain lengths to yield realistic representations of polymers. This is particularly true for high-T<sup>g</sup> polymers, which present steric hindrance issues and slow dynamics, typically resulting in shorter chain lengths. Moreover, simulated polymerization approaches can be more computationally demanding than alternative methods, since they are driven by brute-force molecular dynamics. Therefore, this work aimed at achieving longer chain lengths, while maintaining reasonable efficiency. Examination of the simulated polymerization on the polymers studied here suggested two parameters most influential for improving the completion (i.e., chain lengths) and efficiency: the temperature of molecular dynamics simulations and artificial charges on reactive atoms. These parameters were examined thoroughly, and the results are presented here for PET, PC, PEI, and PIM-1.

The completion of the simulated polymerization was improved when increasing the temperature of the molecular dynamics simulations during cycle equilibrations, as shown in Fig. 6. A significant improvement in the completion percentages occurred at temperatures significantly above the glass transition temperatures. Completion of the more flexible polymers (PET and PC), for example, increased 3–6 % when raising the temperature from 300 to 600 K. For the more rigid polymers (PEI and PIM-1), the increase in the molecular dynamics temperature was even more substantial due to the slower dynamics of these high-T<sup>g</sup> polymers. As such, the completion percentages jumped 12–17 % when the temperature was raised from 600 to 1,000 K. On further temperature increases, improvements to the completion percentages leveled off. The increased temperatures also boosted the efficiency of the algorithms,

Fig. 6 Percent completion of the simulated polymerization as a function of the temperature during NVT molecular dynamics simulations. Results are averaged over five independent simulations with error bars given by the standard deviation

Fig. 7 a Percent completion of the simulated polymerization as a function of the artificial charges, qpolym, added to the reactive atoms. All sets were polymerized with an NVT temperature of 2,000 K, except PIM-1 (b), which was polymerized at 1,000 K. Results are averaged over five independent simulations with error bars given by the standard deviation. b A snapshot of PIM-1 during polymerization illustrating the formation of loops and clusters of charged chain ends (qpolym = 1.0 e) due to the strong attractive forces. Positively and negatively charged end groups are labeled with a ''?'' and ''-,'' respectively. Note that the remainder of the chains are cutoff after the first repeat unit for clarity

as calculated by the average number of bond attempts required per polymerization step at the end of the polymerization (see step 2c in the algorithm), which is shown in Fig. S3 of the Supplemental Material. The larger structural fluctuations induced by higher temperatures led to faster growth of the polymer chains.

In addition to adjusting the temperature, simulated polymerizations were performed with artificial charges on reactive atoms, qpolym, ranging from 0.0 to 2.0 e. In Fig. 7a, the percent completion of the polymerization is plotted as a function of qpolym, from which a few observations can be made. First, the percent completion achieved with any charge decreased with increasing polymer rigidity; that is, the more flexible polymers (PET and PC) had the highest completion, and the more rigid

<span id="page-11-0"></span>polymers (PEI and PIM-1) the lowest. Second, nearly full completion was achieved for PET and PC in all cases, suggesting that the effect of the artificial charges was negligible for the more flexible polymers. The more rigid PIM-1, on the other hand, showed an improvement in the chain lengths when going from no artificial charges to 0.3 or 0.5 e. The small charges introduced slight long-range attractions between unreacted chain ends, which caused a rise in the completion percentage of \*6 %. Furthermore, the attraction between chain ends provided by the addition of artificial charges reduced the average bond length made during polymerization steps, as shown in Fig. S4 of the Supplemental Material.

When the artificial charges were increased further to larger magnitudes (1.0 and 2.0 e), a negative effect was observed during the polymerization of PEI and PIM-1. In these cases, the association between charged chain ends became too strong. Loops of chains were observed in many of the structures, for example, where the two chain ends were held together by strong Coulombic interactions. These were not bonded due to the constraints preventing intramolecular bonding, but were also not able to break free during the molecular dynamics simulations, even with temperatures as high as 1,000 and 2,000 K. For PIM-1, the impact of the larger charges was even more destructive because of the extra orientational bonding criteria imposed. In many of the simulations, clusters of four chain ends formed and hindered proper alignment. These groupings remained throughout the polymerization and prevented bonding from occurring, thus resulting in a decrease in the completion of 5–6 % when increasing the charge to 1.0 e, and an additional drop of 7 % with a charge of 2.0 e. An example of loops and clusters of chain ends formed during the polymerization of PIM-1 is shown in Fig. [7b](#page-10-0), and further examples for PEI and PIM-1 are given in Fig. S5 in the Supplemental Material.

It should be stressed that temperature increases and artificial charges were implemented into the simulated polymerization purely as a means to improve the completion percentages and efficiency, and not to affect the final state of the system. Particularly, high temperatures were used to speed up the dynamics of the high-T<sup>g</sup> polymers during polymerization. Likewise, artificial charges were included to increase the long-range interactions of reactive atom and bias the probability of two chain ends finding each other during the simulations. Proper relaxation of the structures was provided by subsequent application of the 21-step equilibration, bringing the system to the desired final conditions. Therefore, this work suggests that molecular dynamics temperatures above the glass transition temperature, 600–2,000 K, and artificial charges of qpolym B0.5 e be employed to provide the best completion and efficiency of the simulated polymerization.

Validation of the structure generation methodologies presented in this work is provided by comparison of the simulations to available experimental data, including densities, wide-angle X-ray scattering (WAXS), gas adsorption isotherms, porosity measurements, glass transition temperatures, and thermal expansion coefficients. Due to the approximations and assumptions involved, these comparisons must be made with care. As Rouquerol et al. [[3\]](#page-17-0) wrote: ''One must not look for a ''perfect agreement'' between parameters provided by different methods... Instead, one must be aware of the specific, limited and complementary significance of the information delivered by each method of characterization.'' Although this was written about porous solids, it certainly extends to the characterization of materials in general, whether it be in simulations or experiments. Therefore, in this work, comparison of the simulations to experimental data is made for a range of properties with consideration of the fundamental differences of each technique.

## 5.3.1 Structural properties

Experimentally determining the density of a polymer can be quite elusive and, depending on the method used, can result in a range of values [\[3](#page-17-0), [64,](#page-18-0) [65](#page-18-0)]. One distinction is made in the definition of the material's volume. If the pores are included in the volume, a measure of the ''bulk'' density is given:

$$\rho_{\text{bulk}} = \frac{m}{v_{\text{tot}}},\tag{1}$$

where m is the mass of the polymer and vtot is the total sample volume. Alternatively, the density measured excluding all (accessible) pores from the volume provides a ''skeletal'' or ''true'' density:

$$\rho_{\rm skel} = \frac{m}{v_{\rm skel}} = \frac{m}{v_{\rm tot} - v_{\rm pore}},\tag{2}$$

where vskel is the skeletal volume and vpore is the pore volume (vtot = vskel ? vpore). If the crystal structure of a material is known, the bulk density is readily given by the volume of the unit cell. For amorphous materials, experimental techniques usually provide a skeletal density that depends on the probe being used. Examples include helium pycnometry or hydrostatic weighing, where measures of the polymer volume are those inaccessible to the gas or fluid [\[3](#page-17-0), [64\]](#page-18-0).

Simulations provide an exact structure and therefore yield true bulk densities of amorphous materials from the total volume of the simulation unit cell. It should be noted that while this may be the exact bulk density of the

<span id="page-12-0"></span>Fig. 8 Schematic representation of geometric surface definitions, which can be thought of as being traced by a probe molecule rolled across the framework atoms. The Connolly surface is traced out by the edge of the probe, while the accessible surface is taken from the center of the probe

simulated structure, it carries no bearing on the accuracy of its representation of the experimental system. In addition to bulk densities, pore volumes can be obtained geometrically from the simulations to provide skeletal densities of the polymer. The geometric pore volume is measured from the ''Connolly'' surface mapped out by the edge of a probe molecule as if rolling it across the system atoms [[66–68\]](#page-18-0), a schematic representation of which is given in Fig. 8. In this work, pore volumes were obtained with the Atom Volumes and Surfaces tool in the Materials Studio software package [[43\]](#page-17-0). A He-size probe (dHe = 2.6 A˚ ) was used in order to provide skeletal densities consistent with those determined experimentally via helium pycnometry.

The simulated densities of the polymers at 300 K and 1 bar are given in Table 2 along with experimental data. In all cases, better agreement with the experimental values was achieved with the simulated skeletal densities, suggesting that this provides a more realistic comparison with densities calculated by experimental methods such as helium pycnometry. For all polymers, except PIM-1, the simulated skeletal densities are within 2 % of the experimental values, showing excellent agreement. Recall that the final densities of the simulations were not set to match experimental values, but were predicted after application of the 21-step equilibration. The skeletal densities obtained for PIM-1 showed greater deviation (\*8 %) from the experimental values. This is likely due to the variability of the larger pore volume of PIM-1, which affects the calculation of the skeletal densities, as discussed below.

Given the definitions in Eqs. [1](#page-11-0) and [2,](#page-11-0) the difference between the bulk and skeletal densities is determined by the amount of pore volume that exists in the material, as described by the equation:

Table 2 Polymer densities

| Polymer | q (g cm-3<br>) |               |               |  |  |  |
|---------|----------------|---------------|---------------|--|--|--|
|         | Expa           | Sim (skel)b   | Sim (bulk)b   |  |  |  |
| PS      | 1.04–1.065     | 1.04 (0.01)   | 0.97 (0.01)   |  |  |  |
| PMMA    | 1.17–1.20      | 1.18 (0.03)   | 1.12 (0.02)   |  |  |  |
| PET     | 1.333–1.335    | 1.315 (0.005) | 1.293 (0.009) |  |  |  |
| PC      | 1.19–1.2       | 1.206 (0.004) | 1.168 (0.004) |  |  |  |
| PEI     | 1.27–1.28      | 1.277 (0.004) | 1.203 (0.008) |  |  |  |
| PIM-1   | 1.06–1.4c      | 1.281 (0.002) | 0.94 (0.02)   |  |  |  |

<sup>a</sup> Experimental densities from Refs. [\[70,](#page-18-0) [71\]](#page-18-0) unless otherwise noted

$$\frac{1}{\rho_{\text{skel}}} = \frac{1}{\rho_{\text{bulk}}} - \frac{v_{\text{pore}}}{m}.$$
 (3)

Counter-intuitively, an increase in the pore volume results in an increased skeletal density, while the bulk density decreases. As such, greater deviations between the bulk and skeletal densities exist when the pore volume is large, explaining why PIM-1 shows such a large difference between the densities (0.94 and 1.281 g cm-<sup>3</sup> , respectively). These observations also provide some insight into the wide range of densities measured for PIM-1 experimentally. From helium pycnometry, a value of 1.4 g cm-<sup>3</sup> [\[46](#page-17-0)] was obtained. Densities determined by hydrostatic weighing in a fluorocarbon fluid, on the other hand, produced smaller values of 1.06–1.09 g cm-<sup>3</sup> [\[69](#page-18-0)], since less pore volume was accessible to the larger probe molecules. Moreover, comparison of simulated and experimental densities is more difficult for polymers with larger pore volumes, like PIM-1, since the geometric measurements in the simulations cannot compare directly with helium pycnometry experiments, the differences between which are exaggerated for larger pore volumes. Nonetheless, good agreement was still observed between the simulated skeletal and experimental helium pycnometry densities.

To illustrate the utility of the 21-step equilibration, simulations of PS were prepared for various final conditions (Tfinal and Pfinal) for comparison with experimental densities measured by Quach and Simha [[72\]](#page-18-0). In the experimental study, volume changes at different temperatures and pressures were calculated by length changes in the bellows of a dilatometer, which correspond to changes in bulk densities. The absolute volumes, however, were determined based on a reference volume measured by hydrostatic weighing in water, which yields a skeletal density. Therefore, to provide a fair comparison in the simulations, changes in the bulk density at different

<sup>b</sup> Simulation results averaged over five independent boxes with the standard deviation given in parentheses. Bulk and skeletal densities defined in Eqs. [1](#page-11-0) and [2](#page-11-0), respectively

<sup>c</sup> PIM-1 experimental densities from Refs. [[46](#page-17-0), [69](#page-18-0)]

conditions were applied to a reference skeletal density at 294 K and 1 bar,  $\rho_{\rm skel}^0$ , such that the adjusted density was given by:

$$\rho_c = \rho_{\text{skel}}^0 + \Delta \rho_{\text{bulk}} = \rho_{\text{skel}}^0 + \rho_{\text{bulk}}^c - \rho_{\text{bulk}}^0. \tag{4}$$

Here, the change in bulk density,  $\Delta \rho_{\rm bulk}$ , was measured as the difference between the bulk density of the reference state (294 K and 1 bar),  $\rho_{\rm bulk}^0$ , and the state of interest,  $\rho_{\rm bulk}^c$ . The predicted final densities of the simulations, adjusted by the above equation, are plotted against experimental data in Fig. 9 for two pressures ( $P_{\rm final}=1$  and 400 bar) and a range of temperatures below and above the glass transition ( $T_{\rm final}=294$ , 337, 393, 429, and 469 K). Overall, the agreement with experimental data is excellent, which further validates the 21-step equilibration and illustrates the potential of the presented methodology for predictive structure generation at any conditions.

Structural characterization of simulations is often provided by the radial distribution or pair correlation function, g(r), which expresses the probability of finding an atom a distance r away from another atom. The Fourier transform of g(r) provides the structure factor:

$$S(q) = 1 + 4\pi\rho \int_{0}^{\infty} r^2 \frac{\sin(qr)}{qr} g(r) dr, \qquad (5)$$

where  $q = 2\pi / d$  [73], which can be directly compared with experimental neutron or X-ray scattering data [74, 75]. This comparison yields, first and foremost, validation of the simulated structures, but the structural detail

**Fig. 9** The density of polystyrene simulations against experimental values, where experimental equivalence is represented by the *dashed line*. Simulations were prepared at 1 and 400 bar at temperatures of 294, 337, 393, 429, and 469 K (labeled below the 1 bar data, and above the 400 bar data). Simulation results were averaged over five independent simulations with *error bars* given by the standard deviation. Experimental data are from Ref. [72]

available from the simulations also can provide additional insight into understanding and interpreting scattering features. Examples can be found for PS [76, 77], PC [78], and PIM-1 [79, 80], among others. In this work, structure factors were calculated with the Interactive Structure Analysis of Amorphous and Crystalline Systems (ISAACS) software [81, 82], using X-ray scattering lengths for comparison with experimental WAXS data. Overall, the simulated structure factors showed excellent agreement with WAXS data in both the peak positions and relative intensities, as shown for PC, PEI, and PIM-1 in Fig. 10. Note that some

**Fig. 10** Simulated structure factors for **a** polycarbonate, **b** polyetherimide, and **c** PIM-1 shown in comparison with wide-angle X-ray scattering data. Simulation results were averaged over five independent simulations with *error bars* given by the standard deviation. Experimental data are from Refs. [79, 83–85]

variability in the peaks can be observed experimentally between polymer samples with different histories, as is shown for PIM-1.

#### 5.3.2 Adsorption properties

Glassy polymers often display desirable properties for gas adsorption because of their large free volumes. However, the porosity in polymers is difficult to characterize with the currently available experimental techniques, so many of the properties of interest are often obtained indirectly through models and theories that may not be appropriate. Several examples are derived from adsorption data, including apparent surface areas by application of BET theory [6], or pore size distributions from non-local and quenched solid density function theory (NLDFT and QSDFT) [7, 86] or the Horváth-Kawazoe method [87]. The questionable applicability of many of these techniques has been addressed in the literature [3, 4, 88], including direct comparisons to simulation results [51, 89, 90]. Simulations offer a unique perspective of the structures of porous solids because of the detail provided by the molecular models.

In this work, adsorption isotherms were obtained for PIM-1 using the Monte Carlo for Complex Chemical Systems (MCCCS) Towhee program [42] in the grand canonical ensemble [91] with the RANLUX pseudorandom number generator [92, 93]. Simulation parameters for methane (CH<sub>4</sub>) and carbon dioxide (CO<sub>2</sub>) were taken from the TraPPE force field [54, 94]. For CH<sub>4</sub>,  $3 \times 10^6$  equilibration and production steps were performed, while  $1.5 \times 10^7$  steps each were performed for CO<sub>2</sub>. During adsorption, the polymer atoms were held fixed to reduce the computational cost. In Fig. 11, CH<sub>4</sub> and CO<sub>2</sub> adsorption isotherms calculated for PIM-1 at 293 K are shown in comparison with experimental data. The simulated

**Fig. 11** Methane (CH<sub>4</sub>) and carbon dioxide (CO<sub>2</sub>) adsorption at 293 K in PIM-1. Simulation results were averaged over five independent simulations with *error bars* given by the standard deviation. Experimental data from Ref. [95]

isotherms agreed well with the experimental data, reproducing the shape and magnitude of the isotherms. Some of the discrepancies can be attributed to the kinetics of the experiment, such as accessibility of the pores and possible swelling of the polymer.

While BET surface areas can be calculated in simulations from a nitrogen adsorption isotherm at 77 K, like is done experimentally, this is a quite computationally expensive task. Therefore, a more common comparison with experimental BET surface areas is provided by geometric values calculated from the "accessible" surface [67, 68, 96], as shown in Fig. 8. It should be stressed that the geometric and BET surface areas are fundamentally different and should be compared with care. In this work, geometric surface areas were obtained for PIM-1 using an  $N_2$ -size probe ( $d_{N2} = 3.681 \text{ Å}$ ) in Materials Studio [43]. The average surface area was found to be 510  $\text{m}^2\text{ g}^{-1}$ . Although this value is much lower than experimental BET surface areas of PIM-1 (760–875  $m^2 g^{-1}$ ) [97, 98], it is consistent with previous simulations of PIM-1 that were also shown to exhibit simulated BET surface areas in the experimental range ( $SA_{geom} = 587$  and  $SA_{BET} = 830 \text{ m}^2 \text{ g}^{-1}$ ) [51]. The discrepancies, in this case, were shown to arise from the insufficiency of the BET model to estimate surface areas in PIMs, further discussion of which can be found there [51].

#### 5.3.3 Thermal properties

The glass transition temperature is an important property of polymers, as it marks the temperature at which the polymer transitions from a melt to glassy state, where the molecular motions are severely limited. Experimentally,  $T_{g}$ is typically measured by a transition in a property as a function of temperature, such as volume in dilatometry or enthalpy in calorimetry. In dilatometry, the specific volume of the polymer is tracked as it is cooled slowly at a constant rate from a high initial temperature in the melt state, and  $T_{\sigma}$ is determined as the temperature at which the rate of volume change slows. This procedure can be mimicked by molecular dynamics simulations, as has been shown in previous work [99, 100]. A few key differences exist in the simulations, which can affect the values of  $T_{\rm g}$  calculated: (a) Cooling rates are orders of magnitude faster ( $\sim 10^{12}$  vs. 10<sup>1</sup> K/min), (b) the molecular weights are much lower, and (c) the temperature during molecular dynamics is usually cooled stepwise instead of at a constant rate like is done experimentally.

In this work, initial structures in the melt state were achieved by application of the 21-step equilibration with  $P_{\rm final} = 1$  bar and  $T_{\rm final} > T_{\rm g}$  (600–1,200 K). Then, the structures were cooled stepwise during NPT molecular dynamics simulations of 500 ps every 50 K (a cooling rate

<span id="page-15-0"></span>of 6 9 10<sup>12</sup> K/min). The specific volumes at each temperature were calculated from the average over five independent boxes, each averaged over the last 400 ps of the simulation at that temperature. A linear regression was fit to the low- and high-temperature regions of the plot, as shown in Fig. 12a for PET and Fig. S6 in the Supplemental Material for the remaining polymers. Note that the regression was fit to the 3–5 data points in the temperature region showing linearity, such that generally the coefficient of determination, R<sup>2</sup> , was [0.998 in the low-temperature region and [0.996 in the high-temperature region. Although more variability was observed for PIM-1, as given by the standard deviations, reasonable fits were still obtained (R2[0.996).

The simulated Tg's were determined from the intersection of the low- and high-temperature regression lines and are given with experimental values for all polymers in Table [3](#page-16-0). Additionally, the simulated values are plotted

Fig. 12 a Specific volume of poly(ethylene terephthalate) plotted as a function of temperature, using a cooling rate of 50 K/500 ps. Simulation results were averaged over five independent simulations with error bars given by the standard deviation. b Simulated Tg's of the polymers plotted against the experimental values. The dashed line represents experimental equivalence

Earlier in this section, it was stressed that polymer simulations should strive for obtaining longer chain lengths since many polymer properties are dependent on molecular weight. Although the differences in structural properties are not as apparent, the effect of molecular weight in simulations is visible for kinetically influenced properties. For example, previous simulations have found that T<sup>g</sup> increases with molecular weight [[99\]](#page-18-0), as is observed experimentally. In this work, T<sup>g</sup> was calculated for shortchain PIM-1 simulations from simulated polymerizations with a lower completion percentage (\*80 %). Using the same procedures as described above, a T<sup>g</sup> of 903 K was measured in the short-chain PIM-1 simulations, as shown in Figure S6 of the Supplemental Material, which is 62 K lower than was found for long-chain PIM-1. This example illustrates the importance of chain lengths in atomistic simulations of amorphous polymers, particularly when kinetic factors are involved.

Another common thermal property is the thermal expansion coefficient, which measures the change in volume (or length) of a material in response to a change in temperature. The linear and volume thermal expansion coefficients are defined by:

$$\alpha_{\rm L} = \frac{1}{L} \left( \frac{\partial L}{\partial T} \right)_P \quad \text{and} \quad \alpha_{\rm V} = \frac{1}{V} \left( \frac{\partial V}{\partial T} \right)_P,$$
 (6)

respectively, where T is the temperature, L is the length, and V is the volume of the sample. In this work, a<sup>L</sup> and a<sup>V</sup> were calculated from the T<sup>g</sup> simulations described above, which were performed at constant pressure (1 bar) over a range of temperatures. Separate coefficients were obtained

<span id="page-16-0"></span>Theor Chem Acc (2013) 132:1334 Page 17 of 19

Table 3 Thermal properties of polymers

| Polymer | Tg<br>(K) |      |           | (9 10-5 K-1<br>)     | (9 10-4 K-1<br>aV<br>)  |                        |
|---------|-----------|------|-----------|----------------------|-------------------------|------------------------|
|         | Expa      | Simb | Expa      | Simb                 | Expa                    | Simb                   |
| PS      | 373       | 400  | 6–8c      | 8.3–8.5c<br>, 16d    | 1.7–2.48c<br>, 5.1–6.0d | 2.5–2.6c<br>, 4.6–4.9d |
| PMMA    | 379–387   | 408  | 7c        | 5.8–5.9c<br>, 10-11d | 1.7–3c<br>, 5.1–6d      | 1.7–1.8c<br>, 3.0–3.2d |
| PET     | 342–388   | 404  | 9.1d      | 4.8–4.9c<br>, 11d    | 1.7c<br>, 3.94d         | 1.4–1.5c<br>, 3.1–3.3d |
| PC      | 413–424   | 465  | 6.75c     | 6.7–6.8c<br>, 14-15d | 1.80–2.15c<br>, 6.04d   | 2.0c<br>, 4.2–4.5d     |
| PEI     | 488–490   | 574  | 5.6–5.95c | 5.1c<br>, 10d        |                         | 1.5c<br>, 2.9–3.1d     |
| PIM-1   | 709e      | 965  |           | 8.1–8.2c<br>, 29–30d |                         | 2.4–2.5c<br>, 8.6–9.4d |

Tg glass transition temperature, a<sup>L</sup> linear thermal expansion coefficient, a<sup>V</sup> volume thermal expansion coefficient

for temperatures below and above Tg, with the slopes taken from the regression line for the given temperature region of either the volume or length data (qV/qT and qL/qT). The coefficients were then normalized by the volume or length at the given temperature. The simulated linear and volume thermal expansion coefficients for all polymers are given in Table 3 along with available experimental data. Although there is some deviation from the experimental data, good agreement was observed overall. Like with the T<sup>g</sup> calculations, discrepancies can be attributed to the faster cooling rates in the simulations.

This section has provided validation of the structure generation methodology presented in this work, both the simulated polymerization and 21-step compression, through its application to an array of linear polymers. To illustrate the validity of the simulations, comparisons of structural, adsorption, and thermal properties were made to available experimental data. For example, the skeletal densities of all six polymers studied here (PS, PMMA, PET, PC, PEI, and PIM-1) were in excellent agreement with experimental results. Moreover, densities from simulations of PS compared well with experimental measures at a wide range of conditions, illustrating the vast potential of the methods for structure generation at any conditions. Good agreement was also observed for gas adsorption isotherms and porosity measurements in PIM-1. Lastly, the simulations were shown to provide thermal properties, including glass transition temperatures, consistent with experimental data. Although specific properties were examined in this work, the presented structure generation methodology is expected to extend equally well to study a range of other properties for any number of applications of amorphous polymers.

## 6 Conclusions

In this work, a computational methodology for structure generation of amorphous polymers was presented, consisting of a simulated polymerization algorithm and 21-step molecular dynamics equilibration scheme. The algorithm and parameters of the simulated polymerization were discussed in detail with a focus on its implementation in the Polymatic code. Two parameters of the simulated polymerization, the molecular dynamics temperature and artificial charges on reactive atoms, were determined to be the most important for improving its completion and efficiency. Similar consideration was provided to the role of the 21-step equilibration as a general scheme for effectively and efficiently relaxing and compressing polymer simulations to consistent and realistic final densities. Moreover, this scheme provides structures with predicted final densities shown to be consistent with experimental data. The applicability of the presented methods was illustrated for six linear, glassy polymers with varying functionality, polarity, and rigidity: polystyrene (PS), poly(methyl methacrylate) (PMMA), poly(ethylene terephthalate) (PET), polycarbonate (PC), polyetherimide (PEI), and a polymer of intrinsic microporosity (PIM-1).

Validation of the simulated polymerization and 21-step molecular dynamics equilibration was provided by comparison of the simulations and experiments for a range of properties. For example, structural properties, including skeletal densities and structure factors, were shown to be in excellent agreement with experimental data for all polymers. In addition, the porosity of PIM-1 was examined through gas adsorption isotherms and geometric measures of surface area, which were consistent with experimental

<sup>a</sup> Experimental data from Refs. [\[70,](#page-18-0) [71](#page-18-0), [103\]](#page-18-0) unless otherwise noted

<sup>b</sup> Simulation results averaged over five independent boxes. a<sup>L</sup> and a<sup>V</sup> defined in Eq. [6](#page-15-0)

<sup>c</sup> For T\T<sup>g</sup>

<sup>d</sup> For T[T<sup>g</sup>

<sup>e</sup> Experimental T<sup>g</sup> of PIM-1 from Ref. [[101](#page-18-0)]

<span id="page-17-0"></span>results. Lastly, thermal properties like glass transition temperatures and thermal expansion coefficients were examined for all polymers. Despite the fast cooling rates required in the simulations, the simulated values agreed well with the experimental data. Overall, the strong agreement observed between the simulations and available experimental results illustrated the validity of the presented structure generation techniques.

This work demonstrates the vast potential of the presented structure generation methodology for simulations of amorphous polymers. First and foremost, the generality of the techniques has been illustrated for a wide range of polymeric materials. It has been shown for structures with a variety of connectivity and is particularly effective for high-T<sup>g</sup> polymers, despite their bulky structures and slow relaxation times. Second, these methods allow for predictive simulations to be performed knowing only the chemical structure. The 21-step molecular dynamics equilibration can provide predicted final densities consistent with a wide range of experimental conditions. Third, this work provides sufficient details of the algorithms and techniques used, as well as example input scripts for use with the freely available Polymatic simulated polymerization code and LAMMPS software. Through the synergism of simulation and experiment, the predictive capability and generality of the presented simulation methodology will allow for improved understanding of the structure and properties of amorphous polymers and their potential for a variety of applications.

Acknowledgments The authors acknowledge the National Science Foundation (DMR-0908781) for funding. Computational resources were provided by the Materials Simulation Center of the Materials Research Institute, the Research Computing and Cyberinfrastructure unit of Penn State Information Technology Services, and the Penn State Center for Nanoscale Science. Additional computational resources were provided by instrumentation funded by the National Science Foundation (OCI-0821527).

## References

- 1. Haward RN, Young RJ (eds.) (1997) The physics of glassy polymers, 2nd edn. Chapman & Hall, London
- 2. Tant MR, Hill AJ (eds.) (1999) Structure and properties of glassy polymers. American Chemical Society, Washington, DC
- 3. Rouquerol J, Avnir D, Fairbridge CW, Everett DH, Haynes JM, Pernicone N, Ramsay JDF, Sing KSW, Unger KK (1994) Pure Appl Chem 66:1739–1758
- 4. Sing K (2001) Colloids Surf A 187–188:3–9
- 5. Gelb LD (2009) MRS Bull 34:592–601
- 6. Brunauer S, Emmett P, Teller E (1938) J Am Chem Soc 60:309–319
- 7. Lastoskie C, Gubbins KE, Quirke N (1993) J Phys Chem 97:4786–4796
- 8. Gidley DW, Peng HG, Vallery RS (2006) Annu Rev Mater Res 36:49–79

- 9. Theodorou DN (2007) Chem Eng Sci 62:5697–5714
- 10. Barrat JL, Baschnagel J, Lyulin A (2010) Soft Matter 6:3430– 3446
- 11. Theodorou DN, Suter UW (1985) Macromolecules 18:1467– 1478
- 12. Flory PJ (1974) Macromolecules 7:381–392
- 13. Charati SG, Stern SA (1998) Macromolecules 31:5529–5535
- 14. Hofmann D, Fritz L, Ulbrich J, Schepers C, Bo¨hning M (2000) Macromol Theory Simul 9:293–327
- 15. Hofmann D, Fritz L, Ulbrich J, Paul D (2000) Comput Theor Polym Sci 10:419–436
- 16. Karayiannis NC, Mavrantzas VG, Theodorou DN (2004) Macromolecules 37:2978–2995
- 17. Fang W, Zhang L, Jiang J (2010) Mol Sim 36:992–1003
- 18. Siepmann JI, Frenkel D (1992) Mol Phys 75:59–70
- 19. Dodd L, Boone T, Theodorou D (1993) Mol Phys 78:961–996
- 20. Mavrantzas VG, Boone TD, Zervopoulou E, Theodorou DN (1999) Macromolecules 32:5072–5096
- 21. Santos S, Suter UW, Muller M, Nievergelt J (2001) J Chem Phys 114:9772–9779
- 22. Peristeras LD, Rissanou AN, Economou IG, Theodorou DN (2007) Macromolecules 40:2904–2914
- 23. Mu¨ller-Plathe F (2002) Chem Phys Chem 3:754–769
- 24. Curco´ D, Alema´an C (2007) J Comput Chem 28:1929–1935
- 25. Peter C, Kremer K (2009) Soft Matter 5:4357–4366
- 26. Wu C, Xu W (2006) Polymer 47:6004–6009
- 27. Varshney V, Patnaik SS, Roy AK, Farmer BL (2008) Macromolecules 41:6837–6842
- 28. Liu JW, Mackay ME, Duxbury PM (2009) Macromolecules 42:8534–8542
- 29. Farah K, Karimi-Varzaneh HA, Mu¨ller-Plathe F, Bo¨hm MC (2010) J Phys Chem B 114:13656–13666
- 30. Liu H, Li M, Lu ZY, Zhang ZG, Sun CC, Cui T (2011) Macromolecules 44:8650–8660
- 31. Abbott LJ, Colina CM (2011) Macromolecules 44:4511–4519
- 32. Ikeno T, Tsubuku M, Katagiri M, Tsujimoto T (2011) Chem Lett 40:309–311
- 33. Khare KS, Khare R (2012) Macromol Theory Simul 21:322–327
- 34. Jang C, Lacy TE, Gwaltney SR, Toghiani H, Pittman CU (2012) Macromolecules 45:4876–4885
- 35. Brenner D (1990) Phys Rev B 42:9458–9471
- 36. Marks N (2000) Phys Rev B 63:1–7
- 37. Stuart SJ, Tutein AB, Harrison JA (2000) J Chem Phys 112:6472–6486
- 38. Van Duin ACT, Dasgupta S, Lorant F, Goddard WA (2001) J Phys Chem A 105:9396–9409
- 39. Larsen GS, Lin P, Hart KE, Colina CM (2011) Macromolecules 44:6944–6951
- 40. Martı´nez L, Andrade R, Birgin EG, Martı´nez JM (2009) J Comput Chem 30:2157–2164
- 41. Martı´nez L, Martı´nez JM, Birgin E (2011) Packmol. [http://](http://www.ime.unicamp.br/martinez/packmol) [www.ime.unicamp.br/martinez/packmol](http://www.ime.unicamp.br/martinez/packmol)
- 42. Martin MG (2010) Monte Carlo for complex chemical systems (MCCCS) towhee, version 6.2.12. <http://towhee.sourceforge.net>
- 43. Accelrys Software Inc (2007) Materials studio, release 6.0. Accelrys Software Inc, San Diego, CA
- 44. Abbott LJ (2012) Polymatic, Version 1.0. [http://www.matse.psu.](http://www.matse.psu.edu/colinagroup/polymatic) [edu/colinagroup/polymatic](http://www.matse.psu.edu/colinagroup/polymatic)
- 45. McKeown NB, Budd PM (2010) Macromolecules 43:5163– 5176
- 46. Heuchel M, Fritsch D, Budd PM, McKeown NB, Hofmann D (2008) J Memb Sci 318:84–99
- 47. Li C, Strachan A (2010) Polymer 51:6058–6070
- 48. Izumi A, Nakao T, Shibayama M (2012) Soft Matter 8:5283– 5292
- 49. Plimpton S (1995) J Comput Phys 117:1–19

- <span id="page-18-0"></span>50. Plimpton S, Thompson A, Crozier P (2012) Large-scale atomic/ molecular massively parallel simulator (LAMMPS). [http://](http://lammps.sandia.gov/) [lammps.sandia.gov/](http://lammps.sandia.gov/)
- 51. Hart KE, Abbott LJ, Colina CM (2013) Analysis of force fields and BET theory for polymers of intrinsic microporosity. Mol Sim. doi[:10.1080/08927022.2012.733945](http://dx.doi.org/10.1080/08927022.2012.733945)
- 52. Abbott LJ, McDermott AG, Del Regno A, Taylor RGD, Bezzu CG, Msayib KJ, McKeown NB, Siperstein FR, Runt J, Colina CM (2013) J Phys Chem B 117:355–364
- 53. Sun H (1994) J Comput Chem 15:752–768
- 54. Martin MG, Siepmann JI (1998) J Phys Chem B 102:2569–2577
- 55. Martin MG, Siepmann JI (1999) J Phys Chem B 103:4508–4517
- 56. Wick CD, Martin MG, Siepmann JI (2000) J Phys Chem B 104:8008–8016
- 57. Wick CD, Stubbs JM, Rai N, Siepmann JI (2005) J Phys Chem B 109:18974–18982
- 58. Rai N, Siepmann JI (2007) J Phys Chem B 111:10790–10799
- 59. Wang J, Wolf RM, Caldwell JW, Kollman PA, Case DA (2004) J Comput Chem 25:1157–1174
- 60. Bayly CI, Cieplak P, Cornell W, Kollman PA (1993) J Phys Chem 97:10269–10280
- 61. Frisch MJ, Trucks GW, Schlegel HB, Scuseria GE, Robb MA, Cheeseman JR, Montgomery JA, Vreven T, Kudin KN, Burant JC, Millam JM, Iyengar SS, Tomasi J, Barone V, Mennucci B, Cossi M, Scalmani G, Rega N, Petersson GA, Nakatsuji H, Hada M, Ehara M, Toyota K, Fukuda R, Hasegawa J, Ishida M, Nakajima T, Honda Y, Kitao O, Nakai H, Klene M, Li X, Knox JE, Hratchian HP, Cross JB, Bakken V, Adamo C, Jaramillo J, Gomperts R, Stratmann RE, Yazyev O, Austin AJ, Cammi R, Pomelli C, Ochterski JW, Ayala PY, Morokuma K, Voth GA, Salvador P, Dannenberg JJ, Zakrzewski VG, Dapprich S, Daniels AD, Strain MC, Farkas O, Malick DK, Rabuck AD, Raghavachari K, Foresman JB, Ortiz JV, Cui Q, Baboul AG, Clifford S, Cioslowski J, Stefanov BB, Liu G, Liashenko A, Piskorz P, Komaromi I, Martin RL, Fox DJ, Keith T, Al-Laham MA, Peng CY, Nanayakkara A, Challacombe M, Gill PMW, Johnson B, Chen W,Wong MW, Gonzalez C, Pople JA (2004) Gaussian 03, revision C.02. Gaussian Inc, Wallingford, CT
- 62. Hockney RW, Eastwood JW (1988) Computer simulation using particles. Taylor & Francis, New York
- 63. Plimpton S, Pollock R, Stevens M (1997) Particle-mesh Ewald and rRESPA for parallel molecular dynamics simulations. In: Proceedings of the eighth SIAM conference on parallel processing for scientific computing
- 64. Kholodovych V, Welsh WJ (2007) Densities of amorphous and crystalline polymers. In: Mark JE (ed.) Physical properties of polymers handbook, 2nd edn. Springer, New York
- 65. Tsyurupa M, Davankov V (2006) React Funct Polym 66:768– 779
- 66. Connolly ML (1983) J Appl Crystallogr 16:548–558
- 67. Gelb LD, Gubbins KE (1998) Langmuir 14:2097–2111
- 68. Sarkisov L, Harrison A (2011) Mol Sim 37:1248–1257
- 69. Budd PM, McKeown NB, Fritsch D (2006) Macromol Symp 245–246:403–405
- 70. Brandrup J, Immergut EH, Grulke EA, Abe A, Bloch DR (eds.) (2005) Polymer handbook, 4th edn. Wiley, New York

- 71. Mark JE (ed.) (2009) Polymer data handbook, 2nd edn. Oxford University Press, New York
- 72. Quach A, Simha R (1971) J Appl Phys 42:4592–4606
- 73. Allen MP, Tildesley DJ (2007) Computer simulation of liquids. Oxford University Press, New York
- 74. Paul W, Smith GD (2004) Rep Prog Phys 67:1117–1185
- 75. Arbe A, Alvarez F, Colmenero J (2012) Soft Matter 8:8257– 8270
- 76. Mondello M, Yang HJ, Furuya H, Roe RJ (1994) Macromolecules 27:3566–3574
- 77. Ayyagari C, Bedrov D, Smith GD (2000) Macromolecules 33:6194–6199
- 78. Eilhard J, Zirkel A, Tschop W, Hahn O, Kremer K, Scharpf O,
- Richter D, Buchenau U (1999) J Chem Phys 110:1819–1830 79. McDermott AG, Larsen GS, Budd PM, Colina CM, Runt J
- (2011) Macromolecules 44:14–16 80. Fang W, Zhang L, Jiang J (2011) J Phys Chem C 115:14123– 14130
- 81. Le Roux S, Petkov V (2010) J Appl Crystallogr 43:181–185
- 82. Le Roux S, Petkov V (2012) Interactive structure analysis of amorphous and crystalline systems (ISAACS), version 2.5. <http://isaacs.sourceforge.net>
- 83. LeGrand DG, Bendler JT (eds.) (2000) Handbook of polycarbonate science and technology. Marcel Dekker, New York
- 84. Wang Y, Jiang L, Matsuura T, Chung TS, Goh SH (2008) J Membr Sci 318:217–226
- 85. Du N, Robertson GP, Song J, Pinnau I, Thomas S, Guiver MD (2008) Macromolecules 41:9656–9662
- 86. Neimark AV, Lin Y, Ravikovitch PI, Thommes M (2009) Carbon 47:1617–1628
- 87. Horva´th G, Kawazoe K (1983) J Chem Eng Jpn 16:470–475
- 88. Rouquerol J, Llewellyn P (2007) Stud Surf Sci Catal 160:49–56
- 89. Walton KS, Snurr RQ (2007) J Am Chem Soc 129:8552–8556
- 90. Bae YS, Yazaydin aO, Snurr RQ (2010) Langmuir 26:5475– 5483
- 91. Norman GE, Filinov VS (1969) High Temp 7:216–222
- 92. Lu¨scher M (1994) Comput Phys Commun 79:100–110
- 93. James F (1994) Comput Phys Commun 79:111–114
- 94. Potoff JJ, Siepmann JI (2001) AIChE J 47:1676–1682
- 95. Larsen GS (2011) Simulations and experiments on gas adsorption in novel microporous polymers. Ph.D Dissertation, The Pennsylvania State University
- 96. Du¨ren T, Millange F, Ferey G, Walton K, Snurr R (2007) J Phys Chem C 111:15350–15356
- 97. McKeown NB, Budd PM (2006) Chem Soc Rev 35:675–683
- 98. Weber J, Su Q, Antonietti M, Thomas A (2007) Macromol Rapid Commun 28:1871–1876
- 99. Lyulin AV, Balabaev NK, Michels MAJ (2003) Macromolecules 36:8574–8575
- 100. Soldera A, Metatla N (2006) Phys Rev E 74:1–6
- 101. Staiger CL, Pas SJ, Hill AJ, Cornelius CJ (2008) Chem Mater 20:2606–2608
- 102. Ahn J, Chung WJ, Pinnau I, Song J, Du N, Robertson GP, Guiver MD (2010) J Membr Sci 346:280–287
- 103. Greiner R, Schwarzl FR (1984) Rheol Acta 23:378–395