ELSEVIER

#### Contents lists available at ScienceDirect

# Polymer

journal homepage: www.elsevier.com/locate/polymer

# Evolution of network topology of bifunctional epoxy thermosets during cure and its relationship to thermo-mechanical properties: A molecular dynamics study

Chunyu Li, Alejandro Strachan\*

School of Materials Engineering and Birck Nanotechnology Center, Purdue University, West Lafayette, IN 47906, USA

#### ARTICLE INFO

Article history:
Received 21 June 2015
Received in revised form
13 August 2015
Accepted 19 August 2015
Available online 21 August 2015

Keywords:
Thermoset polymer
Polymer network
Crosslink density
Molecular dynamics
Glass transition temperature
Structure property relationship

#### ABSTRACT

Thermoset polymers are used for a wide range of application from large airframes to microelectronics and fundamental understanding of the development of their 3D network during cure and its relationship to thermo-mechanical properties is of fundamental and applied interest. Experimental characterization of network properties, such as the density of crosslinked chains, involves the use of approximate models with uncertain parameters leading, consequently, to uncertainties in resulting properties. We use molecular dynamics to simulate the cure of two thermoset epoxy systems, characterize the evolution of its network topology and establish relationships to the stiffness, yield stress and glass transition temperature. Relating the predicted crosslink density with experimental measurements of rubber elasticity indicates that these polymers behave somewhere between an affine and phantom networks; the proportionality constant extracted between rubber modulus and crosslink density can be used for similar systems to experimentally extract network characteristics.

© 2015 Elsevier Ltd. All rights reserved.

#### 1. Introduction

Epoxy thermoset polymers have tremendously wide applications in industries ranging from aerospace and automotive to energy and electronics [1]. They are either used as a constituent matrix of advanced composites or as an independent functional material. The wide application of epoxy thermosets comes from desirable properties for their processing (including resins with relatively low viscosity, excellent surface wetting and low chemical shrinkage upon curing) and excellent thermo-mechanical properties of the cured networks including high stiffness, strength glass transition temperature. The performance of thermoset polymers depends strongly on the 3D molecular network structure that forms during curing. Normally, epoxy thermoset polymers are cured using protocols designed to result in highly crosslinked networks in order to get the desired thermal and mechanical properties. However, very high cure degrees and highly crosslinked network can result in significant strain at molecular levels and result in increased brittleness [2]. Thus, formulations need to balance strength and toughness and the development of commercial products require optimizing multi-component and, often, multi-phase systems. Thus, establishing chemistry-processingstructure-property relationship for epoxy thermosets is important from an applied science point of view; it also remains a basic science grand challenge. Developing such knowledge would provide design engineers with tunable knobs to optimize ultimate properties reducing the number of trial-and-error experiments. Achieving this bold goal requires a clear understanding of polymer network development during the curing process for this class of materials and to relate network topology to properties; in this paper we simulate the curing of thermoset polymers using molecular dynamics (MD) simulations and analyze the evolution of network; the combination of simulation results with experimental measurements provide important insight into the properties of epoxy thermosets.

The importance of the network structure on properties has long been recognized. The classical Flory-Stockmayer theory on gelation and crosslinking was developed decades ago [3–6]. Many experimental characterization techniques designed to characterize polymer networks have also been developed in the past decades. Commonly used ones include equilibrium swelling [7], mechanical measurements [8] and nuclear magnetic resonance spectroscopy

Corresponding author.

E-mail address: strachan@purdue.edu (A. Strachan).

[\[9\]](#page-8-0). Modeling has also been applied to the growth of network structure, such as the tree-like model developed by Dusek [\[10,11\]](#page-8-0) and the lattice model developed by Topolkaraev et al. [\[12,13\].](#page-8-0) However, these experiments and theoretical efforts have limitations. The bulk of the experimental work concentrated on the network structure of rubbers and the techniques described above only provide an average description of network structure such as crosslink density. More importantly, extracting network characteristics from experimental measurements involves models with unknown parameters, leading to unquantified uncertainties in the predictions. In this paper we show that combining molecular dynamics-based predictions of network structure with experimental properties enables the calibration of the unknown parameters in the models.

In recent years, with the help of supercomputers, MD simulations have become a powerful tool to simulate the curing process of epoxy thermosets [\[14\]](#page-8-0) starting from the uncured resin and making relatively straightforward assumptions about chemical reactions. Hamerton et al. [\[15\]](#page-8-0) carried out the first fully atomistic MD simulation on trifunctional polycyanurate. Doherty et al. [\[16\]](#page-8-0) implemented a progressive polymerization during a MD simulation and provided a foundation for most current using algorithms. Yarovsky and Evans [\[17\]](#page-8-0) developed a computational procedure for constructing molecular models of crosslinked polymer networks based on epoxy resins. Heine et al. [\[18\]](#page-8-0) simulated the structure of endcrosslinked poly(dimethyl-siloxane) networks. Wu and Xu [\[19\]](#page-8-0) developed a similar method to construct polymer networks for an epoxy resin system but with a different molecular force field. Komarov et al. [\[20\]](#page-8-0) reported a computational method where the polymer network is polymerized at a coarse grain level and then mapped into a fully atomistic model. Varshney et al. [\[21\]](#page-8-0) studied molecular modeling of thermosetting polymers with special emphasis on comparison of crosslinking procedures. Lin and Khare [\[22\]](#page-8-0) presented a single-step polymerization method for the creation of atomistic model structures of crosslinked polymers. Bermejo and Ugarte [\[23\]](#page-8-0) introduced a method for building fully atomistic models of chemically crosslinked similar to the approach of Yarovsky and Evans [\[17\]](#page-8-0). Bandyopadhyay et al. [\[24\]](#page-8-0) proposed an efficient method of creating united-atom molecular models of a crosslinked epoxy system. More recently, Li and Strachan [\[25,26\]](#page-8-0) systematically studied the network formation of epoxy resins and developed a thermoset simulator, named MDPoS, which enables large-scale polymerization and crosslinking simulations by an efficient charge updating approach.

All-atom MD simulations provide the temporal evolution of every atom in the polymer and, thus, can be used to obtain a complete description of network topology and its evolution during network formation. So far, however, there has been no report on this aspect of polymer network studies. The objective of this paper is to fill this gap by presenting an algorithm for the characterization of thermoset network and applying it to characterize the network development of epoxy/amine systems. We focus on bifunctional epoxy resins and tetrafunctional crosslinkers. Two systems, namely diglycidyl-ether of bisphenol A and 3,3'-diaminodiphenyl sulphone (DGEBA/33DDS) and diglycidyl-ether of bisphenol F and diethylenetoluenediamine (DGEBF/DETDA) are the focus of this study. These systems were chosen since they are key components in commercial resin formulations and have been extensively characterized both experimentally and theoretically. By obtaining additional insight regarding their detailed network structure and how the network forms, we should be able to establish the linkage from curing process, to network structure, then to material properties.

The remainder of this paper is organized as follows. Section 2 describes the systems of interest and the molecular models used to simulate the cure process and Section [3](#page-2-0) describes the algorithms used to characterize network topology. Sections [4, 5 and 6](#page-2-0) describe the evolution of the network during cure and its relationships with experimental rubber elasticity measurements. Section [7](#page-6-0) correlated network topology with thermo-mechanical properties and conclusions are drawn in Section [8.](#page-8-0)

# 2. Epoxy systems and curing process

# 2.1. MDPoS

We characterize the curing of two bifunctional epoxy resins with tetrafunctional crosslinkers: DGEBA/33DDS and DGEBF/ DETDA. The molecular structures of these momoners are shown in [Fig. 1.](#page-2-0) The crosslinking simulations are performed with perfect stoichiometries, i.e. there are twice as many epoxy molecules as crosslinkers. The initial model system is built packing both activated epoxy and crosslinker monomers (1024 epoxies and 512 crosslinkers) into a simulation cell followed by a structural relaxation using the commercial software MAPS [\[27\]](#page-8-0). The number of total atoms is 64,000 for DGEBF/DETDA and 69,120 for DGEBA/ 33DDS. The crosslinking procedure MDPoS was detailed in the authors' previous articles [\[25,26\];](#page-8-0) here we provide a brief description. MDPoS mimics the polymerization by the periodic creation of bonds between pairs of reactive atoms using a distance criterion (with a cutoff equal to four times of the equilibrium NeC bond length of 1.41 Å). The new chemical bonds are turned on slowly using a 50 ps long multi-step relaxation procedure to avoid large atomic forces. After the new set of bonds is fully relaxed the system is thermalized for an additional 50 ps and a new round of bond creation is started. This procedure is carried out at 600 K with a desire to increase molecular mobility and produce well-relaxed crosslinked structures. The conversion limit is set to be 85% as a compromise between simulation time and achieving a realistic degree of cure; this is because the conversion rate decreases with conversion as finding nearby reactive atom pairs becomes more difficult.

# 2.2. Reaction kinetics

Primary and secondary amine reactions have different activation energies and, thus, different reaction rates. At relatively low temperatures primary reactions are favored over secondary reactions while at higher temperatures both reactions can occur with similar rates. MDPoS can incorporate such information and be used to study the effect of reaction path on the development of the network. As in Ref. [\[25\]](#page-8-0) we study the two extremes of the possible reactions: i) The equal reaction rate model assumes both primary and secondary reactions are equally likely and occur whenever reactive atoms are within the cutoff distance. ii) The primary reaction first model enables only reaction of primary amines during the initial stage of the reaction (taken as 2.0 ns, long enough for most possible reactions to occur) and only enables secondary amine reactions in the second stage). The former represents an idealization of a two-stage cure process with a low temperature cure followed by higher temperatures.

#### 2.3. Simulation details

All MD simulations are performed by using the open source package LAMMPS [\[28\]](#page-8-0), a massively parallel MD simulator from Sandia National Laboratories. In simulating the crosslinking process, 3D periodic boundary conditions are imposed to remove possible surface effects and a NoseeHoover thermostat [\[29\]](#page-8-0) with 100 fs coupling constant and a NoseeHoover barostat [\[30\]](#page-9-0) with

<span id="page-2-0"></span>
$$\begin{array}{c} O \\ CH_2 \\ CH_3 \\ CH_3 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2 \\ CH_2$$

Fig. 1. Two epoxy systems: DGEBA/33DDS and DGEBF/DETDA.

1000 fs coupling constant are respectively used for temperature and pressure control. Atomic interactions are described using the DREIDING force field [\[31\]](#page-9-0) with partial atomic charges obtained using the electrochemical potential equalization method [\[32\].](#page-9-0)

### 3. Algorithm for network topology characterization

The above-mentioned epoxy and crosslinker monomers are low molecular weight molecules and, as the crosslinking chemical reactions proceed, they gradually form a 3D network structure. At the various stages of the crosslinking process the network consists of a complex distribution of molecular fragments of varying architectures and molecular weights. It is important to develop a clear picture of network growth and characterize the network topology. Interesting structural parameters include: the molecular weight of chains between crosslink sites (tertiary amines in our case), the monomer composition and chain length of chains between crosslinks, the amount of monomers not attached to the network, the amount of chains attached to the network in one end but having a free end (dangling chains) at another end and so on.

The chains formed by crosslinking chemical reactions can be classified into three types: crosslinked (both ends are connected to crosslink points in the network), dangling (one end attached to the network but another end is free) and free chains (both end are free). Correlations between the location of these chains are also potentially important for understanding mechanical properties of thermoset polymers. Clearly the architectural characterization of a network polymer is more challenging than in the case of linear chain polymers.

For computational characterization of a thermoset network, we define a fully converted amine as a crosslink point and a fully unconverted amine or epoxide group as a free end. Any linear chain between two crosslink points is a crosslinked chain. Any linear chain between a crosslink point and a free end is defined as a dangling chain. A linear chain between two free ends is seen as a free chain. A linear chain is a segment between crosslinks or chain ends.

We developed an algorithm to identify all these three types of polymer chains for a given crosslinked system. The main steps shown in the flowchart are explained in the following:

Step 1. Identifying reacted sites: Reactive carbon atoms on the epoxy monomers and reactive nitrogen atoms are assigned with different atom types before crosslinking in MPPoS. When a reaction occurs between a carbon and a nitrogen atom, the corresponding atom types are changed. Thus by tracking the atom type changes, the reacted sites either carbon or nitrogen as well as partially converted or fully converted nitrogen are identified. Step 2. Establishing linkage map: Once reacted sites are established establish a linkage map between all molecules in the system by tracking all NeC bonds.

Step 3. Identifying linear chain segments: The next step in the process is to identify linear chain segments, each of these start and end either at a crosslink node (i.e. a tertiary amine) or a chain end (i.e. an unreacted epoxy C or primary amine N). Thus we identify connected molecules belonging to each chain.

Step 4. Characterizing network structure: Linear chain segments have two ends (a free end or a crosslink point) leading to three possible categories. If a segment two free ends, it is a free chain. If a chain has one free end and one crosslink point as an end, it is classified as a dangling chain. If both ends of a linear chain are crosslink points, it is classified as a crosslinked chain.

Step 5. Collecting statistic information: After network characterization, we perform a statistical analysis about the network structure. Information such as chain length, molecular weight, gyration radius, mass center, as well as the percentage of crosslinked chains, dangling chains and free chains in the network is carefully calculated and collected.

To verify our implementation of the algorithm described above, two small systems with 24 monomers (16 epoxy monomers and 8 crosslinker monomers) were crosslinked up to 80% conversion degree. For a few selected conversion degrees, topology maps were manually created based on the structure generated from our MDPoS simulator. The topology predicted by our analysis code coincided to the manual ones for all these curing stages; this verifies our implementation.

# 4. Network structure evolution during curing

As crosslinking proceeds, the conversion degree increases and network structure evolves. We tracked the changes of network structure by characterizing all three types of chains at various conversion degrees. [Fig. 2](#page-3-0) shows the evolution of the network structure during the curing process for the DGEBA/33DDS system. As expected, the total molecular weight of crosslinked chains increases with conversion degree while the number of free chains decreases. It is interesting to note that dangling chains initially outweigh the crosslinked chains but their molecular weight peaks at about 60% conversion degree and then decreases. Interestingly, even at 85% conversion there are still a non-negligible percentage of free chains in the system and over 20% of the molecular weight is on dangling chains. Similar trend is observed for the DGEBF/DETDA

<span id="page-3-0"></span>Fig. 2. Percent molecular weight in free, dangling and crosslinked chains as a function of cure degree for DGEBA/33DDS.

# system.

Fig. 3 displays the distribution of molecular compositions of the crosslinked (a), dangling (b) and free chains (c) that constitute the polymer network at a conversion degree of ~85% for both systems. For comparison, the number of a specific type of chains is normalized by the total number of monomers in the initial resin. We find a wide distribution of chain sizes. Most of crosslinked chains, dangling chains and free chains consist of a single molecule, either an epoxy resin or a crosslinker. But there are indeed a small percentage of longer chains, consisting of multiple monomers. The longest with 5 monomers (3 DGEBF plus 2 DETDA) is seen as a dangling chain. Even a free chain is shown to have 4 monomers (2 DGEBF plus 2 DETDA) connected together. Our simulations represent the first detailed characterization of polymer network structure of epoxies.

Possible correlations between the location of the various types of chain could provide additional insight into the polymerization process and the resulting properties of the polymer. [Fig. 4](#page-4-0) shows snapshots of molecular structures of the DGEBA/33DDS system at ~85% conversion where we separate crosslinked, dangling and free chains. While visual inspection is difficult for condensed systems, the results show that free chains are rather uniformly distributed. These locations of free chains could become nucleation sites for defects like voids or cracks when the polymer is subjected to mechanical loading. We are currently investigating the effect of these potential defects on mechanical properties.

# 5. Crosslink density, rubber modulus and crosslink molecular weight

It is not possible to experimentally characterize the network structure of thermosets directly [\[33\].](#page-9-0) Models are used to relate the experimental observable with network characteristics and this process introduces uncertainties that are difficult to quantify. On the other hand, as described in Section [4](#page-2-0), MD simulations can provide detailed information about the network; but MD simulations are not without limitations, especially in the characterization of low-frequency rubber properties [\[34\].](#page-9-0) We now show that combining MD results and experimental results from Ref. [\[35\]](#page-9-0) provides interesting insight into the physics of epoxy polymers

Fig. 3. Chain compositions in cured polymers: (a) crosslinked; (b) dangling; (c) free chains.

<span id="page-4-0"></span>Fig. 4. Chain locations (DGEBA/33DDS, conversion degree ~85%).

and relationships that can aid in experimental efforts.

A widely used approach to gain insight into the network from experiments involves measuring the equilibrium modulus of the polymer in the rubber state (T > Tg+40 K) and applying rubber elasticity theory to relate the modulus to the density of effective crosslinked strands ( $\chi$ ). The challenge is that the relationship between these two quantities is not known with a high level of accuracy. The simplest model of rubber elasticity, the affine network, predicts that each crosslinked chain contributes kT to the shear modulus [33]:

$$G = \frac{nkT}{V} = \chi kT \tag{1}$$

where n stands for the number of effective elastic chains, V the volume, k the Boltzmann's constant and T the absolute temperature. The phantom network model is less restrictive in terms of the deformation of the network and predicts lower modulus for the same crosslink density:

$$G = \frac{f - 2}{f} \chi kT \tag{2}$$

where f in the front factor represents the functionality of the polymer. In our case f=3 and the phantom and affine models differ by a factor of 1/3. In addition, both these expressions assume perfect networks and defects such as dangling and free chains introduce additional uncertainties. It is clear that the experimental determination of crosslink densities suffers from the uncertainties associated with the proportionality constant between the density of crosslinks and modulus.

We assume that the rubber modulus is proportional to the density of crosslinked chains:

$$G = \eta \chi kT \tag{3}$$

and determine the proportionality constant  $\eta$  combining our MD simulations for the crosslink density and experimental modulus. Fig. 5(a) shows the evolution of crosslink density as a function of conversion degree obtained from our simulations. We find that both polymer systems exhibit similar trends of crosslink density increasing with conversion degree. The horizontal dashed lines represent ideal values assuming all possible reactions occur. Fig. 5(b) compares the MD predictions (both for the equal chemical rates and primary first models) with experimental results from Ref. [35] for DGEBA/33DDS. The authors used the affine network model to extract crosslink density (as is common practice) and the

results are displayed as open triangles; these results are clearly lower than the MD predictions. Using the phantom model together with the experimental moduli leads to better agreement with the simulations but the value for 100% conversion is over the ideal value. As discussed above, the simulation results for crosslink density can be combined with the experimental modulus to predict the proportionality constant  $\eta$  in Eq. (3). This leads to a value  $\eta=0.625$ , intermediate between the affine model with  $\eta=1$  and

Fig. 5. Crosslink density increasing with conversion degree.

<span id="page-5-0"></span>the phantom model with <sup>h</sup> <sup>¼</sup> 1/3. Interestingly, the primary first model is closer to the experiments than the equal rates one.

The resulting value of h in Eq. [\(3\)](#page-4-0) from the combined MD and experiments results can be discussed in terms of the constrainedjunction model [\[36,37\].](#page-9-0) The affine and phantom network models can be considered as two extreme cases of the constrained-junction network model when the confining potential acting on the crosslink points is extremely large or small, respectively. The formulation of the constrained-junction model results in a network stiffness that depends on both elastic strain and confining potential. Following the approximate solution of the constrained-junction model given by Rubinstein and Panyukov [\[38\]](#page-9-0), we calculated the Mooney ratio and obtain a diagram showing Mooney ratio as a function of elongation. The diagram is included in the supplemental information. The combined MD crosslink information and experimental modulus results the relative strength of confining potential of N/m0 <sup>¼</sup> 6.0 for the constrained-junction model.

Another important and widely used characteristic of polymer networks is average molecular weight of crosslinked chains (Mc) [\[39,40\].](#page-9-0) For an ideal case in which all molecules contribute to the network Mc and the crosslink density are related by the mass density: c ¼ r/Mc. However, this relationship is extensively used in non-ideal networks [\[35,41\]](#page-9-0) due to its simplicity and the lack of more appropriate relationships. Our results in Section [4](#page-2-0) show that thermosets deviate from this simple relationship between crosslink density and molecular weight even for relatively high conversion degrees as a non-negligible fraction of the polymer mass remains in dangling and free chains. In general, the proportionality constant between c and 1/Mc is the mass density of the crosslinked network (r<sup>c</sup> the ratio of the total mass between crosslink points and the total volume).

In the following we explore this relationship for epoxy thermosets based on our MD simulations that provide the number of crosslinked chains and the exact molecular weight of every chain between crosslink points. We define a crosslink point as a tertiary amine nitrogen connected with at least one crosslink chain, which has both tertiary nitrogen ends. Fig. 6(a) shows the average molecular weight of crosslinked chains with conversion degree. During the early stages of the process Mc increases with cure degree as linear chains become crosslinked; after approximately 60% conversion the molecular weight decreases with increasing conversion degree as crosslinks become more numerous and chains become shorter. The error bars represent the standard deviation over all crosslinked chains in the system and not the uncertainty in the mean value; large values indicate a very heterogeneous system with significant variability in the molecular weight between crosslinks, as revealed in [Fig. 3](#page-3-0). Experimental Mc results from Ref. [\[35\]](#page-9-0) for DGEBA/33DDS system lead to significantly larger values; as can be see in Fig. 6(b). Recall that two assumptions are made to extract Mc, first a rubber elasticity model is used to obtain c from the rubber modulus, followed by Mc ¼ r/c. This last relationship implies that the entire mass of the system is part of crosslinked chains; as clearly shown in [Fig. 2](#page-3-0) this is not the case except at very high conversion degrees. Fig. 6(b) shows the results reported in Ref. [\[35\]](#page-9-0) using the affine model (pentagons) and the values obtained from the experimental rubber modulus and the phantom model (stars). In addition, we show MD predictions of r/c and the actual prediction Mc. It is clear that using the total density leads to a significant overestimation of the crosslinked molecular weight.

At room temperature, the total mass density r of epoxy thermosets is always around 1.2 g/cm<sup>3</sup> . It slightly increases with the conversion degree because of the volume shrinkage during the curing process but the change is usually within 10%. The mass

Fig. 6. (a) Average molecular weight of crosslinked chains Mc obtained from the MD simulations; (b) Traditional empirical estimation of Mc as r/c.

density of crosslinked chains is totally different. Fig. 7 shows the mass density of crosslinked chains r<sup>c</sup> based on the network structure shown in [Fig. 2](#page-3-0). As expected, this mass density increases significantly with increasing conversion degree. If we estimate Mc

Fig. 7. Mass density (rc) of crosslinked chains.

<span id="page-6-0"></span>using the mass density of crosslinked chains, i.e.  $\rho_c/\chi$ , the results (shown in the inset of Fig. 7) would be exactly the same as we presented before in Fig. 6(a), though the estimation based on mass density can never give the variation range of  $M_c$ . It is clear that estimating  $M_c$  from the mass density has to be done with care except for very high conversion degrees where  $\rho_c$  approaches  $\rho$ .

## 6. Radius of gyration of individual chains

Another interesting feature for a polymer network is the spatial extension of each chain and the average size of all chains. The radius of gyration is a useful measurement for the dimensions of polymer chains. The analysis presented above indicates a range of compositions for the crosslinked chains and we now focus on characterizing their shape. Once each chain is identified as described above we use LAMMPS to compute the radius of gyration between crosslink points. The radius of gyration is measured relative to the mass center of a chain and taken from a MD simulation at room temperature and averaged over 100 simulation timesteps (0.1 ps). Fig. 8(a) shows the resulting distribution of radius of gyration for the DGEBA/33DDS system at ~85% conversion degree. The majority of crosslinked chains (~95%) have a radius of gyration less than 1.0 nm but the distribution is bi-modal with few chains (less than 5%) exhibiting radii of gyration between 3 nm and 7 nm. The

**Fig. 8.** Radius of gyration: (a) number percentage of chains with certain radius of gyration at ~85% conversion (DGEBA/33DDS); (b) weighted average of gyration radius of all chains at different conversion degrees.

DGEBF/DETDA system shows a similar distribution but is not presented here. Further analysis of the data shows that the first peak in the distribution corresponds to DGEBA and 33DDS molecules and the second one corresponds to cases where crosslinks are separated by two or three molecules. Fig. 8(b) shows the evolution of the average radius of gyration for all chains in a system as a function of conversion degrees. The average is calculated by using the molecular weight of each chain as the relative weight, i.e.  $\overline{R_g} = \sum (m_i R_{gi}) / \sum m_i$ . The standard deviation is equally weighted and represents the variability in radii over the ensemble of chains and not the uncertainty in the mean value. We find that the average radius of gyration initially increases at lower conversion degrees and then gradually decreases after ~60% conversion degree, which is close to the gel point. This trend reflects both the volume shrinkage and the chain size reduction over crosslinking during the curing process.

#### 7. Structure-property relationship

Previous studies have shown that material properties such as the glass transition temperature ( $T_g$ ), stiffness and yield strength of thermosets increase with increasing conversion degree. Such trends have been captured via empirical equations that relate conversion degree or crosslink density, which is usually substituted by  $1/M_c$ , to the materials response. One of the well-known empirical equations is that of DiBenedetto that relates the glass transition temperature and conversion degree [42]:

$$\frac{T_g - T_{g0}}{T_{g1} - T_{g0}} = \frac{\lambda \alpha}{1 - (1 - \lambda)\alpha} \tag{4}$$

where  $\alpha$  stands for the conversion degree,  $T_{g0}$  and  $T_{g1}$  are the glass transition temperatures at conversion degree 0 and 1, respectively, and  $\lambda$  is an adjustable parameter that describes non-linearity in the  $T_g \propto \alpha$  curve. The relationship was reevaluated by Pascault and Williams [43] and the adjustable parameter  $\lambda$  was theoretically related to the change in the isobaric heat capacity between liquid and glassy states. Another widely used empirical equation ascribes a linear relationship between  $T_g$  and 1/Mc [44,45] with a following form

$$T_g = T_{g0} + \frac{\zeta}{M_c} \tag{5}$$

where  $T_{g0}$  is the glass transition temperature of uncured resin and  $\zeta$  is a linear correlation coefficient and depends on resin and cross-linker system ( $\zeta = \sim 40\,\mathrm{K}$  kg/mol for tetrafunctional networks). Crawford and Lesser [45] proposed that the yield strength follows a similar linear relationship with 1/Mc. Qualitatively, the network structure is related to the conversion degree and to Mc. However, as shown in Section 5, the crosslink density has a more direct relationship with the network structure and experimental observables and in the following subsections we explore how key thermomechanical properties depend on it. We stress that no single parameter can describe the complexity and variability.

# 7.1. Glass transition temperature

The first property we study is the glass transition temperature  $T_g$ . During the simulation of crosslinking process, model systems cured up to different conversion degrees are cooled down from the curing temperature (600 K) to room temperature with a cooling rate 10 K/60 ps under atmospheric pressure. The glass transition temperature  $T_g$  can be determined from the slope change in the density—temperature curve as detailed in our previous work

Fig. 9. Crosslink density effect on the glass transition temperature.

[25,26]. Fig. 9 shows the resulting  $T_g$  versus the crosslink density. As expected,  $T_g$  increases with crosslink density but the simulations reveal interesting physics. For low cure degrees there is a nonlinear relationship between  $T_g$  and crosslink density and when the gel point is reached (vertical line in the figure) the slope of the  $T_g$ - $\chi$  curve decreases and our simulation predicts a linear behavior for higher crosslink densities. As discussed above, the crosslink density can only be estimated experimentally (up to a multiplicative constant) when a resin system is cured above the gel point and, thus, there have been numerous reports of a linear relationship between  $T_g$  and crosslink density (or  $M_c$ ). However, Fig. 9 shows that the linear relationship cannot be extended to low crosslink densities below the gel point. Here we propose a new equation similar to Eq. (5) but with a modified parameter:

$$T_g = T_g^{gel} + \zeta \left( \chi - \chi^{gel} \right) \tag{6}$$

where  $T_g^{gel}$  represents the  $T_g$  at the gel point, the parameter  $\zeta$  depends on the resin system. In this study, we have  $\zeta=26.3$  K kg/mol for DGEBF/DETDA,  $\zeta=30.4$  K kg/mol for DGEBA/33DDS by fitting the data shown in Fig. 9.

## 7.2. Elastic properties of networks

Previous experimental studies reported that the Young's modulus of glassy polymer networks exhibits only a weak dependence on the crosslink density [45–47]. As shown below our simulations indicate that there is a considerable effect of crosslink density on the elastic modulus.

The Young's moduli of our samples are obtained from engineering stress-strain curves obtained by deforming the simulation cells with various degrees of cure under uniaxial tension conditions. The simulation cells are stretched along one direction with a deformation rate of  $2.0 \times 10^9~\text{s}^{-1}$  and a stress of one atmosphere is maintained in the two lateral directions with a Parrinello and Rahman barostat [48,49]. Examples of the stress-strain curves are shown in Fig. 10. Young's modulus is obtained by fitting the stress-strain curves up to strain of 4% and averaging results from deformations along the x, y and z directions respectively. The reported errors bars represent the standard deviation among the three tests for each sample. Note that MD codes (including LAMMPS) often report true-stress and we convert it into engineering stress to compare better with experimental results.

**Fig. 10.** Stress—strain relationships of two systems under uniaxial tension in room temperature.

Fig. 11 shows the room temperature Young's modulus for the two epoxy systems as a function of crosslink density. The simulations predict linear relationship between these properties after the gel point with both materials exhibiting similar slopes. The values of the Young's modulus predicted by the MD simulations are comparable to the available experimental data (2.0–3.0 GPa) for similar systems in spite of the high strain rate used in the MD simulations. This is consistent with the weak dependence of glassy modulus on strain rate.

## 7.3. Yield behavior of networks

Unlike the elastic properties, all previous experimental studies reported that the yield behavior has a strong dependence on the crosslink density of polymer networks [45–47]. Our MD results are in good agreement with these observations. The yield stress is taken as the maximum stress in the stress-strain response of uniaxial tension and the yield strain is the strain corresponding with that yield stress. Similar to the case of Young's modulus, the standard deviations for the yield stress and yield strain are determined by averaging the values obtained for three directions.

**Fig. 11.** Crosslink density effect on the elastic stiffness of epoxy thermosets at room temperature.

<span id="page-8-0"></span>Fig. 12. Crosslink density effect on the yield behavior of epoxy thermosets at room temperature.

Fig. 12 shows the dependence of yield stress and strain on crosslink density for the two epoxy systems computed at room temperature. The yield stresses for both materials display a clear linear increase with the crosslink density after the gel point. But the slopes are different with DGEBF/DETDA showing higher sensitivity of yield stress to the crosslink density. Due to the high strain rate and small sample size, the values of the yield stress and strain are higher than the experimental values available in the literature, which is in the range of 50e100 MPa, for similar systems [\[50,51\].](#page-9-0) Below the gel point, the yield stress shows a nonlinear dependence on the crosslink density. It is understandable because the polymer networks have not been extensively formed below the gel point. Quite interestingly, the yield strains of both systems show no apparent dependence on the crosslink density and have an average value about 10.5% above gel point. The large value of yield strain compared with available experimental data, which is usually around 4e5% [\[52\],](#page-9-0) is possibly because the size effects [\[53\]](#page-9-0) (our simulation cells are around 10 nm by 10 nm by 10 nm). This result is consistent with the weak dependence of yield strain on loading conditions in PMMA [\[54\]](#page-9-0).

# 8. Conclusions

We used molecular dynamics-based simulations of the curing of thermoset polymers to characterize the evolution of their molecular structure. The algorithm developed enables us to quantify the molecular weight in crosslinked, dangling and free chains. Interestingly even at 85% conversion a non-negligible fraction of the polymer is not part of the network, indicating that the traditional relationship between crosslinked density and molecular weight cannot be used and our simulations provide more appropriate relationships. Further, combining simulation and experimental results we establish relationships between crosslinked density and rubber modulus that indicate that these systems behave somewhere between an affine and phantom networks. This relationship should be useful to experimentally extract network information for similar systems.

#### Acknowledgments

The authors acknowledge support by the Boeing Company and computational resources from [nanoHUB.org](http://nanoHUB.org) and Research Computing at Purdue University.

# Appendix A. Supplementary data

Supplementary data related to this article can be found at [http://](http://dx.doi.org/10.1016/j.polymer.2015.08.037) [dx.doi.org/10.1016/j.polymer.2015.08.037.](http://dx.doi.org/10.1016/j.polymer.2015.08.037)

# References

- [1] [H.Q. Pham, M.J. Marks, Epoxy resins, in: Encyclopedia Polymer Science](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref1) & [Technology, John Wiley](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref1) & [Sons, New York, NY, 2004.](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref1)
- [2] [A.S. Argon, R.E. Cohen, Polymer 44 \(2003\) 6013](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref2)e[6032](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref2).
- [3] [P.J. Flory, J. Am. Chem. Soc. 63 \(1941\) 3083.](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref3)
- [4] [P.J. Flory, J. Am. Chem. Soc. 63 \(1941\) 3091.](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref4)
- [5] [P.J. Flory, J. Am. Chem. Soc. 63 \(1941\) 3096.](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref5)
- [6] [W.H. Stockmayer, J. Chem. Phys. 12 \(1944\) 125](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref6).
- [7] [P.J. Flory, Principles of Polymer Chemistry, Cornell University Press, Ithaca, NY,](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref7) [1953.](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref7)
- [8] [J.D. Ferry, Viscoeslastic Properties of Polymers, third ed., John Wiley](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref8) & [Sons,](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref8) [New York, 1980.](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref8)
- [9] [W. Kuhn, P. Barth, P. Denner, R. Müller, Solid State Nucl. Magn. reson. 6 \(1996\)](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref9) [295](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref9)e[308](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref9).
- [10] [K. Dusek, Rubber Chem. Technol. 55 \(1982\) 1](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref10)e[22](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref10).
- [11] [K. Dusek, Macromolecules 17 \(1984\) 716](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref11)e[722.](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref11)
- [12] [V.A. Topolkaraev, V.G. Oshmyan, A.A. Berlin, A.N. Zelenetskii, E.V. Prut,](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref12) [N.S. Enikolopyan, Dokl. Akad. Nauk. USSR 225 \(1975\) 1124](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref12)e[1127](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref12).
- [13] [E.F. Oleinik, Epoxy Resins and Composites IV, in: Advances in Polymer Science,](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref13) [80, 1986, pp. 49](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref13)e[99.](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref13)
- [14] [C.Y. Li, A. Strachan, J. Polym. Sci. Part B Polym. Phys. 53 \(2015\) 103](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref14)e[122](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref14).
- [15] [I. Hamerton, C.R. Heald, B.J. Howlin, J. Mater. Chem. 6 \(1996\) 311](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref15)e[314](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref15).
- [16] [D.C. Doherty, B.N. Holmes, P. Leung, R.B. Ross, Comp. Theor. Polym. Sci. 8](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref16) [\(1998\) 169](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref16)e[178.](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref16)
- [17] [E.E. Yarovsky, Polymer 43 \(2002\) 963](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref17)e[969](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref17).
- [18] [D.R. Heine, G.S. Grest, C.D. Lorenz, M. Tsige, M. Stevens, Macromolecules 37](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref18) [\(2004\) 3857](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref18)e[3864](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref18).
- [19] [C.F. Wu, W.J. Xu, Polymer 47 \(2006\) 6004](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref19)e[6009.](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref19)
- [20] [P.V. Komarov, Y.T. Chiu, S.M. Chen, P.G. Khalatur, P. Reineker, Macromolecules](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref20) [40 \(2007\) 8104](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref20)e[8113](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref20).
- [21] [V. Varshney, S.S. Patnaik, A.K. Roy, B.L. Farmer, Macromolecules 41 \(2008\)](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref21) [6837](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref21)e[6842](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref21).
- [22] [P.H. Lin, R. Khare, Macromolecules 42 \(2009\) 4319](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref22)e[4327](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref22).
- [23] [J.S. Bermejo, C.M. Ugarte, Macromol. Theory Simul. 18 \(2009\) 259](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref23)e[267](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref23).
- [24] [A. Bandyopadhyaya, P.K. Valaala, T.C. Clancy, K.E. Wise, G.M. Odegard, Poly](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref24)[mer 52 \(2011\) 2445](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref24)e[2452.](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref24)
- [25] [C.Y. Li, A. Strachan, Polymer 51 \(2010\) 6058](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref25)e[6070.](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref25)
- [26] [C.Y. Li, A. Strachan, Polymer 52 \(2011\) 2920](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref26)e[2928.](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref26)
- [27] MAPS (The Materials And Processes Simulations platform), Scienomics Inc. [http://scienomics.com.](http://scienomics.com)
- [28] LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator), [http://](http://lammps.sandia.gov/) [lammps.sandia.gov/.](http://lammps.sandia.gov/)
- [29] [W.G. Hoover, Phys Rev. A 31 \(1985\) 1695](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref29).

- <span id="page-9-0"></span>[30] [W.G. Hoover, Phys. Rev. A 34 \(1986\) 2499](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref30).
- [31] [S.L. Mayo, B.D. Olafson, W.A. Goddard III, J. Phys. Chem. 94 \(1990\) 8897](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref31)e[8909](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref31).
- [32] [W.J. Mortier, K.V. Genechten, J. Gasteiger, J. Am. Chem. Soc. 107 \(1985\)](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref32) [829](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref32)e[835.](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref32)
- [33] [M. Rubinstein, R.H. Colby, Polymer Physics, Oxford University Press Inc., New](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref33) [York, 2003](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref33).
- [34] [C.Y. Li, G.A. Medvedev, E.W. Lee, J. Kim, J.M. Caruthers, A. Strachan, Polymer 53](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref34) [\(2012\) 4222](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref34)e[4230.](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref34)
- [35] [M. Pramanik, E.W. Fowler, J.W. Rawlins, Polym. Eng. Sci. 54 \(2014\)](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref35) [1990](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref35)e[2004](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref35).
- [36] [G. Ronca, G. Allegra, J. Chem. Phys. 63 \(1975\) 4990.](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref36)
- [37] [P.J. Flory, J. Chem. Phys. 66 \(1977\) 5120](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref37).
- [38] [M. Rubinstein, S. Panyukov, Macromolecules 35 \(2002\) 6670](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref38)e[6686.](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref38)
- [39] [J.A. Schroeder, P.A. Madsen, R.T. Foister, Polymer 28 \(1987\) 929.](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref39)
- [40] [G. Levita, S. DePetris, A. Marchetti, A. Lazzeri, J. Mater. Sci. 26 \(1991\)](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref40) [2348](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref40)e[2352](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref40).
- [41] [L.W. Hill, Prog. Org. Coatings 31 \(1997\) 235](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref41)e[243.](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref41)
- [42] [A.T. DiBenedetto, J. Polym. Sci. Part B Polym. Phys. 25 \(1987\) 1949.](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref42)

- [43] [J.P. Pascault, R.J.J. Williams, J. Polym. Sci. Part B Polym. Phys. 28 \(1990\) 85.](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref43)
- [44] [L. Banks, B. Ellis, Polymer 23 \(1982\) 1466](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref44).
- [45] [E. Crawford, A.J. Lesser, J. Polym. Sci. Part B Polym. Phys. 26 \(1998\)](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref45) [1371](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref45)e[1382](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref45).
- [46] [E. Crawford, A.J. Lesser, Polym. Eng. Sci. 39 \(1999\) 385](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref46)e[392](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref46).
- [47] [U.M. Vakil, G.C. Martin, J. Appl. Polym. Sci. 46 \(1992\) 2089](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref47)e[2099](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref47).
- [48] [M. Parrinello, A. Rahman, J. Appl. Phys. 52 \(1981\) 7182.](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref48)
- [49] [W. Shinoda, M. Shiga, M. Mikami, Phys. Rev. B 69 \(2004\) 134103](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref49).
- [50] [L. Sun, G.L. Warren, J.Y. O'Reilly, W.N. Everett, S.M. Lee, D. Davis, D. Lagoudas,](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref50) [H.J. Sue, Carbon 46 \(2008\) 320](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref50)e[328](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref50).
- [51] [Y.X. Zhou, F. Pervin, L. Lewis, S. Jeelani, Mater. Sci. Eng. A 452](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref51)e[453 \(2007\)](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref51) [657](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref51)e[664.](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref51)
- [52] [J.N. Gao, J.T. Li, B.C. Benicewicz, S. Zhao, H. Hillborg, L.S. Schadler, Polymers 4](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref52) [\(2012\) 187](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref52)e[210](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref52).
- [53] [S. Wang, Y. Yang, L.M. Zhou, Y.-W. Mai, J. Mater. Sci. 47 \(2012\) 6047](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref53)e[6055](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref53).
- [54] [E. Jaramillo, N. Wilson, S. Christensen, J. Gosse, A. Strachan, Phys. Rev. B 85](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref54) [\(2012\) 024114](http://refhub.elsevier.com/S0032-3861(15)30176-2/sref54).