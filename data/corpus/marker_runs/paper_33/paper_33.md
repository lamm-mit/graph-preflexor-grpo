# Integration of Atomistic Simulation with Experiment Using Time—Temperature Superposition for a Cross-Linked Epoxy Network

Ketan S. Khare\* and Frederick R. Phelan Jr.\*

For glass-forming polymers, direct quantitative comparison of atomistically detailed molecular dynamics simulations with thermomechanical experiments is hindered by the vast mismatch between the accessible timescales. Recently, the authors demonstrated the successful application of the timetemperature superposition (TTS) principle to perform such a comparison for the volumetric properties of an epoxy network. Here, the local translational dynamics of the same network is computationally followed-up and studied. The mean squared displacement (MSD) and time-scaling exponent trends of select atoms of the network are calculated over a temperature range that spans the glass transition. Using TTS, both trends collapse onto master curves that relate the reduced MSD and time-scaling exponent to the reduced time at a reference temperature. Because the reduced time of these computational master curves extends to 109 s, they can be directly compared with experimental creep compliance for the same material from the literature. A quantitative comparison of the three master curves is performed to provide an integrated view that relates atomic-level dynamics with macroscopic thermomechanics. The time-shift factors needed for TTS in simulation show excellent agreement with experiment in the literature, further establishing the veracity of the approach.

#### 1. Introduction

There is a sustained interest in the use of molecular dynamics (MD) simulations of cross-linked epoxy and other network polymers for materials research. [1] However, the timescales accessible by MD simulations and thermomechanical experiments are vastly mismatched. This mismatch is exacerbated for viscoelastic polymers where the rate-dependence of their properties is an essential aspect of their physics. This rate effect

K. S. Khare, F. R. Phelan Jr.
Materials Science and Engineering Division
National Institute of Standards and Technology
Gaithersburg, MD 20899, USA
E-mail: ketan.khare@nist.gov; frederick.phelan@nist.gov
K. S. Khare
Department of Physics
Georgetown University
Washington, DC 20057, USA

The ORCID identification number(s) for the author(s) of this article can be found under https://doi.org/10.1002/mats.201900032.

DOI: 10.1002/mats.201900032

further increases the computational workload by necessitating simulations not only at multiple rates but also at slower rates. The integration of the computational and experimental approaches of research is thus hindered.<sup>[2]</sup>

Here, we are specifically interested in atomistically detailed molecular dynamics (atomistic MD) simulations of all-atom models of cross-linked epoxy. Such all-atom models explicitly account for specific chemical interactions such as hydrogen bonding, that have a significant impact on the thermomechanical properties of such systems.<sup>[3,4]</sup> The creation of realistic all-atom model structures of cross-linked epoxy is non-trivial,<sup>[5]</sup> and two successful strategies have been developed to address the associated challenges: 1) the simulated annealing method (SA)<sup>[6–9]</sup> and 2) the directed diffusion (DD) method.<sup>[5,9]</sup>

After the creation of the model structures, the most common method of characterizing such systems is to obtain the glass transition temperature ( $T_g$ ) by cooling the model from the rubbery to

the glassy state and then by analyzing the resulting specific volume–temperature ( $\nu_{\rm sp}$ -T) trend for the characteristic kink of the glass transition. However, because of the timescale limitations of atomistic MD simulations, the computational cooling rates ( $q_{\rm cool}$ 's) are vastly higher than the experimental rates. The use of such high rates in simulations has the effect of drastically elevating the  $T_{\rm g}$ 's and significantly reducing the densities of the glass compared to experiments.<sup>[10]</sup> Furthermore, such high  $q_{\rm cool}$  values broaden the glass transition and make it difficult to determine the  $T_{\rm g}$  values computationally.<sup>[11]</sup>

Recently, [9] we studied an atomistic model for a cross-linked epoxy network formed by the epoxy monomer Epon 1001F<sup>[12,13]</sup> and the cross-linker 4,4'-diaminodiphenyl sulfone (4,4'-DDS) using MD simulations. We refer to this network as the Epon 1001F/4,4'-DDS system. We obtained five computational  $\nu_{\rm sp}$ - T trends by varying the  $q_{\rm cool}$  value that was used to cool the model from the rubbery to the glassy state. We then showed that a comparison of these trends enabled a more objective identification of the rubbery and the glassy states. Because the  $\nu_{\rm sp}$ -T trend in the rubbery state is cooling rate-independent, the computationally obtained rubbery trend could be extrapolated to lower temperatures and then directly compared with the

**Figure 1.** Chemical structure of a) Epon 1001F and b) 4,4′-DDS units in the network. We selected two sets of atoms for analysis: CC and XN atoms, which are highlighted by red and blue circles, respectively.

experimental values in the literature. This protocol, which is called the specific volume-cooling rate (*v*sp-*q*cool) analysis, also enabled a determination of the computational *T*g-*q*cool trend.

Using time-temperature superposition (TTS), the computational time-shift factors (*a*T's) can be calculated for comparison with experiment. Plazek and co-workers[14–16] have provided the experimental *a*T-*T* trend from their study of volume-rate dependent processes and creep compliance measurements for the same system. Thus, we could successfully compare the *a*T-*T* trends from the experiment, simulation, and the William– Landel–Ferry (WLF) equation using the material-specific WLF parameters. Since the use of TTS hinges on the assumption of thermorheological simplicity, this assumption appears to be empirically valid for this network in the context of bridging the mismatch between the timescales of the computational and experimental approaches.[17]

The TTS principle was also invoked by Sirk et al.[18] to compare the Young's modulus obtained using atomistic MD simulations of cross-linked epoxy with experimental data. Specifically, the Young's modulus was obtained by simulating the deformation of the network using four different strain rates at temperatures spanning the *T*g. It was found that the Young's modulus could be collapsed onto a master curve. The resulting computational master curve shows reasonable agreement with the experimental master curve of the storage modulus.[18] Further, the computational and experimental *a*T-*T* trends show excellent agreement. However, Young's modulus and other quantities that rely on the virial stress tensor are subject to large statistical fluctuations, and hence, a relatively high degree of uncertainty.[1]

Conversely, microscopic quantities, such as the mean squared displacement (MSD) of the atoms, can be accurately calculated using atomistic MD simulations with far less uncertainty. Nevertheless, in the available literature on atomistic MD simulations, reports of such MSD trends in a cross-linked epoxy network that extend beyond one nanosecond are rare.[3,4,19] Specifically, Lin and Khare[19] have provided a detailed analysis of the local translational dynamics of select atoms across the *T*g.

Using their approach as a base, in this work, we have characterized the dynamics of our model[9] for the Epon 1001F/4,4′-DDS system. Our model is more than 25 times larger than theirs[19] because of advances in the available computational power over the last decade. Additionally, our system has been extensively characterized experimentally.[14–16,20] These two factors have enabled us to characterize the local translational dynamics of the system with less uncertainty and at significantly longer timescales that proves advantageous for integrating our findings with experiment.

The remainder of the article is organized into four sections. In Section 2, we describe the relevant details of the methods that were used here. Specifically, we present details about the chemistry of the polymer network, the molecular model for the epoxy network, and the simulation and analysis techniques. While all the necessary details are included here, the context provided by our recent work[9] would likely be useful to the reader. The results and discussion in this work involve two principal findings. For their effective presentation, the two findings are presented and discussed separately in Sections 3 and 4.

In Section 3, we focus on the results and discussion of the local translational dynamics from the atomistic MD simulations. We discuss the temperature trends of the MSD and the time-scaling exponent. These trends are then comparatively analyzed with the volumetric behavior from our recent simulation work.[9] In Section 4, we present the superposition of the MSD and the exponent trends, and then make a quantitative comparison of the resulting computational master curves with the experimental creep compliance from the literature. Finally, we summarize our findings in Section 5.

#### **2. Computational Section**

#### **2.1. Network Chemistry**

The Epon 1001F/4,4′-DDS system studied here was formed by the polycondensation of epoxy monomers and aromatic diamine cross-linkers in the stoichiometric ratio of 2:1. The monomer was Epon 1001F,[12,13] and the cross-linker was 4,4′- DDS (see **Figure 1**). This network chemistry was chosen due to the availability of experimental data in the literature.[14–16,20] The experimental values of *T*g range from 399.3 to 407.3 K,[15,16] while the computational values range from 489 to 556 K.[9] Both experimental and computational values show a strong dependence on *q*cool. [9]

#### **2.2. Force-Field and Simulation Parameters**

All simulations were performed using the Large-Scale Atomic/ Molecular Massively Parallel Simulator (LAMMPS) simulation package.[21] The molecular models were described by the all-atom general AMBER force field (gAff).[22–24] The partial charges on atoms were calculated by the Austin Model 1 with bond charge correction (AM1-BCC) method.[25–27] The van der Waals (vdW) interactions were truncated at 9 Å, and the residual was accounted for using tail corrections.[28] The shortrange pairwise coulombic interactions were directly calculated up to 9 Å, and the long-range interactions were calculated using the particle–particle particle–mesh (pppm) method.[29] Isothermal–isobaric conditions were maintained using the Nosé–Hoover[30–32] thermostat and barostat. The pressure was maintained at 5 MPa. This value of pressure was used during

the *v*sp-*q*cool analysis[9] to match experimental conditions,[15] and we have maintained the same pressure here. Bonds and angles containing hydrogen atoms were constrained by the RATTLE algorithm.[33,34] A time step of 1 fs was used throughout the simulations. These details are identical to recent work[9] and are also similar to previous works on cross-linked epoxy networks in the literature.[3–5,8,9,18,19,35–38]

#### **2.3. Model Structures of the Network**

We used the model structures that were prepared in our recent work.[9] We briefly recapitulate some of the relevant details for the sake of completeness. Five independent replicas, each containing about 212139 atoms, were prepared.[9] The preparation strategy started with a model of an equilibrated reaction mixture that contained a stoichiometric mixture of the epoxy monomer (1458 molecules) and the cross-linker (729 molecules) in a simulation box. The typical experimental timescale[14,39] of network formation (curing) by the condensation polymerization of the reactants is inaccessible by simulations. Hence, the network formation for the models was emulated by combining the SA[6–8,40] method and the DD[5] method.[9]

The SA method was reported in 1983 for solving optimization problems such as the traveling salesman problem (TSP).[6,40] The connectivity sequence of the reactants can be framed as a TSP, and then the SA method can be used to obtain a reasonable solution for the sequence, such that the sum of the distances between the reactants is minimized. This strategy was first used[7] to obtain model structures of polystyrene and was later extended[8] for creating models of epoxy networks. Subsequently, it has been used for the study of various epoxy network chemistries.[18,36,38]

After obtaining a reasonable solution for the connectivity sequence of the reactants, the distances between the neighbors still exceeded the gAff-prescribed bond length. The DD[5] method can be used to accelerate the process of the relaxation of these distances while retaining the topological integrity of the reactants. In the method, weak harmonic springs were used to couple the reacting atoms, and the parameters of the spring were gently adapted to the appropriate gAff bond parameters in a series of relatively short MD simulations at temperatures drastically higher than the *T*g of the network.[5] Thus, the diffusion of the reactants in the simulation box was directed in response to the applied spring forces. Unlike Brownian diffusion, DD is extremely rapid, and the high value of the simulation temperature rapidly relaxes the overall topology of the model. The DD method can be conceptualized as a type of steered molecular dynamics (SMD) simulation.[41] The topological parameters of the model containing these diffused reactants were then altered according to that prescribed by the gAff, which completed the network formation process. The distribution of the bonds, angles, and the dihedrals within the network can then be assessed to validate the topological integrity of the model.

After employing the SA and the DD method in our recent work,[9] the five replicas of the model were equilibrated at a high temperature of 820 K. These model structures were then cooled from that temperature in the rubbery state to a low temperature of 140 K in the glassy state using temperature steps of 5 K. At each temperature step, a constant number of particles, pressure, and temperature (NPT) simulation was performed. The duration (*t*cool) of the simulation at each temperature step determined the *q*cool. At the end of each temperature step, the snapshot of the model was saved. This procedure was repeated for each of the five replicas at each of the five *q*cool values.

#### **2.4. Simulation and Visualization**

We began all the simulations here using snapshots from the slowest cooling rate (*q*cool = 5.556 × 109 K s<sup>−</sup><sup>1</sup> ) temperature series. The value of *T*g at this cooling rate is 489 ± 1.0 K. Constant NPT atomistic MD simulations were performed for a duration of 55 ns at 13 different values of temperature (*T*), specifically (350, 400, 450, 475, 500, 525, 550, 575, 600, 650, 700, 750, and 800 K). The corresponding *T*/*T*g values are 0.72, 0.82, 0.92, 0.97, 1.02, 1.07, 1.12, 1.18, 1.23, 1.33, 1.43, 1.53, and 1.64, respectively. All five replicas were used for simulations, altogether, resulting in 65 simulation trajectories. Coordinates for the heavy atoms were saved at intervals of 25 ps.

#### **2.5. Trajectory Analysis**

For analysis, we selected two sets of atoms: 1) the three central carbon atoms of the bisphenol A moieties in the Epon 1001F units; and 2) the nitrogen atoms of the 4,4′-DDS units in the network (see Figure 1). Throughout the remainder of the text, these sets of atoms are correspondingly referred to as the CC atoms and the XN atoms. As described in prior work,[19] it is reasonable to expect the CC and XN atoms to show relatively fast and slow dynamics in the network, respectively. Of the two molecular units in the network, the CC atoms are on the longer and more flexible epoxy monomer. In contrast, the XN atoms are the cross-linking sites in the network, which are bridged by the rigid 4,4′-DDS units. For qualitative comparison, the dynamics of the CC and XN atoms in this work can also be mapped to the dynamics of the middle monomers and crosslinks of model polymer networks studied using coarse-grained MD simulations by Duering et al.[42]

Quantitatively, we characterized the length of the molecular units by calculating the root-mean-square end-to-end distance (*R*e), while their flexibility was characterized by calculating the root-mean-square deviation (δ*R*e) from the corresponding *R*e. Each value and its corresponding distribution are very weakly dependent on the temperature and are identical in both the uncross-linked melt and the polymer network. At 800 K, the *R*<sup>e</sup> ± <sup>δ</sup>*R*e values for the Epon 1001F and the 4,4′-DDS units are 26 ± 8 Å and 9.3 ± 0.7 Å, respectively. Thus, our characterization of the monomer as "longer" and more "flexible" than the cross-linker in this text is only valid on a relative basis.

We used the rerun command in LAMMPS[21,43] to calculate the MSD trends of the atoms. Calculations were repeated for each trajectory using 101 windows,[44,45] each moved by 50 ps from the previous window. The average MSD trend for the two sets of atoms at each temperature was calculated for a total duration of 40 ns. Finally, the overall average trends and their

**Figure 2.** MSD (〈*r* <sup>2</sup>〉) versus *t* of a) CC atoms and b) XN atoms, respectively (log-log axis). *T*g is 489 ± 1 K. Two lines with slopes of 0.17 and 0.5 are shown as a guide for the eye. Uncertainty is less than the thickness of the line. Time-scaling exponent (*m*) of c) CC atoms and d) XN atoms (linear-log axis). The appropriate axes of the four parts are common to facilitate comparison.

standard deviations were calculated using the trends for each of the five replicas. Thus, a given data point in the MSD trend comprises an average of 2208870 and 736290 individual values for the CC and the XN atoms, respectively.

The time-scaling exponent (*m*) of the dynamics is equal to the local slope of the log-log MSD trends. Since the differentiation of data sets amplifies the noise and introduces artifacts, the exponent was calculated using three different approaches: 1) analytical differentiation of quintic smoothing spline approximation of the MSD trends; 2) analytical differentiation of polynomial curve fits of the MSD trends; and 3) the numerical differentiation using the finite difference approximation of the moving block averaged MSD trends.

At each temperature, the three approaches were used for the trends of each replica, and average and uncertainty values were obtained. The *m*-*t* trends for the three approaches were then compared with each other and successfully validated between 0.05 and 20 ns. Beyond this range, the three approaches introduced different artifacts, and the uncertainty of the slopes was too high for any productive analysis. Here, we have only presented the results of the first approach. The findings of our work are neither qualitatively nor quantitatively affected by the use of the other two approaches.

"Error bars" in the figures or uncertainty associated with quantities in the text reflect the "standard error of the mean" for the five model structures. If not shown, the size of the error bars is smaller than the size of the symbols or lines. Finally, the WebPlotDigitizer[46] software was used to extract the necessary data from the literature.[14–16]

#### **3. Results and Discussion of Local Translational Dynamics from Simulation**

#### **3.1. Temperature Trends of Dynamics**

The 13 temperature trends of the MSD and the time-scaling exponent are shown in the four parts of **Figure 2**. The MSD versus time trends for the CC and XN atoms are shown in Figure 2a,b (log-log scale), and the time-scaling exponent trends for the sets of atoms are shown in Figure 2c,d (linear-log scale). The range of time is about 3.2 orders of magnitude, and the *T* values range from 350 to 800 K (0.72*–*1.64*T*g).

The two sets of atoms show broad qualitative similarity in dynamical behavior as can be seen in Figure 2. This qualitative similarity can be attributed to the fact that the overall dynamics

of the cross-linked network is tightly coupled. The MSD values increase monotonically with both increasing time and temperature. The values vary by roughly 2.5 orders of magnitude from about 0.5  $\mbox{Å}^2$  to more than 100  $\mbox{Å}^2$ . As expected, the CC atoms are consistently more mobile than the XN atoms.

The time-scaling exponent ranges from about 0.04 to 0.5. Such sub-diffusive behavior is expected for polymer networks in all dynamical regimes. Unlike the MSD trends, the exponents show non-monotonic behavior with time/temperature. While the trends at the lowest T values show a monotonic increase, the trends at the highest T show a monotonic decrease. The trends for intermediate temperatures show an increase to a peak, followed by a decrease.

Based on their overall characteristics, the trends can be distinguished into three types. These characteristics vary with the  $T/T_{\rm g}$  value of the trends. The time-scale of the step-wise cooling is less than that of the dynamics, and trends cannot be trivially identified as corresponding to the glassy or rubbery states/regimes based solely on  $T/T_{\rm g}$  values. Nevertheless, the evolution of the characteristics is closely related to the glass transition. For each type, we discuss the MSD values, the corresponding length-scale ( $\sqrt{\rm MSD}$ ), and the exponent trends.

1.  $T/T_{\rm g}$  < 0.95: For the three trends below 0.95 $T_{\rm g}$  (350, 400, and 450 K), the MSD values range from 0.4 to 4 Å<sup>2</sup>. The associated length scale is less than the vdW diameter of the carbon and nitrogen atoms, [47] as can be expected for temperatures below the  $T_{\rm g}$ . For each trend, the MSD values increase by a factor less than about 4 in the time range. Furthermore, the MSD trends show relatively low dependence on time, with the exponents increasing from about 0.04 to about 0.2 in a roughly linear fashion.

The trends at 475 K ( $0.97\,T_g$ ) are at the cusp between types and based on the slopes have different characteristics for the two sets of atoms. For the CC atoms, the slope is more clearly self-similar to the next set of higher temperature trends, while for the XN, it is more clearly similar to the lower temperature trends.

2.  $T/T_g$  < 1.2: For the three trends in this range (500, 525, and 550 K), the MSD values range from 1 to 35 Ų. The associated length scale increases to molecular values but is less than the  $R_e$  of the 4,4′-DDS units (9.3 Å). Each trend is characterized by a characteristic upturn in the slope, indicating a crossover to a sharper time-dependence. In this range, the exponents increase from about 0.2 to a peak value of about 0.5. The crossover points occur at increasingly earlier values of time with increasing T. For each trend, the MSD values increase by a factor between about 5 and 20 over the time range, and the factor increases consistently with increasing T.

Again, the trends at 575 K lie at the cusp between the two types and show different behavior for the two sets of atoms. The slope trends for the CC atoms exhibit a long-time point of inflection (but without a maximum), indicating the start of a transition to a declining time-dependence. The trends for the XN have only a slight hint of a long-time inflection and remain more self-similar to the lower temperature trends.

3.  $T/T_g > 1.2$ : For the five trends in this range ( $T \ge 600$  K), the MSD values range from about 4 to 160 Å<sup>2</sup>. The associated length scale greatly exceeds the  $R_e$  values of both the monomer and the cross-linker. For each trend, the MSD values increase

by a factor between about 10 and 20. However, in contrast to the previous two types, this factor consistently decreases with increasing temperature. Also, in contrast to the previous two cases, the exponent shows non-monotonic time-dependence and decreases after reaching a peak value of about 0.5. The exponent trends attain a peak earlier with increasing values of T. The five MSD trends begin to plateau subsequently at longer times. The onset of the plateau also appears to be earlier with increasing values of T.

Altogether, the following picture emerges: At temperatures below  $T_{\rm g}$ , the atoms of the network are trapped in molecular cages and vibrate about their mean position. With increasing temperatures, the atoms break free from the molecular cages and explore the available conformational space with increasing vigor and the values of the MSD and the exponent increase sharply.

As the length scale of the dynamics begins to exceed the size of the cross-linker, the atoms become sensitive to the cross-linked nature of the matrix and trigger the constraints imposed the covalent cross-links. Hence, the time-dependence of the MSD decreases with both increasing time and increasing temperatures. At this point, the atoms continue to remain highly mobile. However, having exhausted the available conformational space, which is limited by the topology, this mobility does not lead to an increase in the values of the MSD, which attains a plateau. This phenomenon is referred to as topological localization.

Having discussed the similar qualitative behavior of the MSD and exponent trends of the two sets of atoms as a function of temperature, we now focus on the quantitative differences in the dynamical behavior between the two sets of atoms. The differences have two aspects. First, the CC atoms are more mobile than the XN atoms at all temperatures by an average factor of about 1.6. This finding can be explained by the fact that the XN atoms are the cross-linking sites, while the CC atoms are situated along the backbone of the somewhat more flexible monomer. If the relative difference in the mobility of the two sets was uniform at all temperatures, the ratio of their MSDs would be constant, and their time-scaling exponents essentially identical.

However, as can be seen for the time- and temperaturetrends in the ratio of the MSD trends of the CC to the XN atoms shown in Figure 3, this is not the case. Similarly, a comparison of Figure 2c,d shows important differences between the time-scaling exponent trends for the CC and XN atoms.

At two lowest temperatures, the ratio is roughly constant throughout the range in time, and the exponent trends for the two atoms are very similar. As the T approaches  $T_{\rm g}$ , the ratio shows a sharp increase. Correspondingly, while the exponent trends for both the atoms begin to show an upturn, the exponent for CC atoms is consistently higher than the XN atoms. Specifically, the difference in the exponent trends of the two atoms at temperatures of 475, 500, and 525 K (0.97, 1.02, and  $1.07T_{\rm g}$ ) is especially striking. At these temperatures, the exponent trends for the CC atoms show sharper upturns than those for the XN atoms. The MSD ratio reaches a peak value of about 1.85 at 525 K. While more subtle, at higher temperatures, the exponent trends of the CC atoms begin to decline slightly in advance than that of the XN atoms, and at the highest

**Figure 3.** MSD ratio of the CC atoms relative to the XN atoms versus time (*t*) at 13 temperatures. Selected temperatures [K] are labeled. All values are greater than unity showing that the CC atoms have greater mobility than the XN atoms. However, the enhancement is not uniform.

temperatures, the trends for the two sets are essentially indistinguishable. Accordingly, the MSD ratio declines to a constant value of about 1.5, just as the MSD trends in Figure 2 show evidence for topological localization.

Altogether, these differences suggest that 1) CC atoms are consistently more mobile than the XN atoms and that 2) during the glass transition, CC atoms have a higher propensity to be mobile than the XN atoms. This difference in the MSD ratio and the time-scaling exponent in the vicinity of *T*g can be attributed to the difference in the topology for the two sets of atoms. We refer to this difference as the topology-induced asynchronous (TIA) dynamics, which will be discussed in greater detail later.

#### **3.2. Comparison with the Literature**

Our observations are consistent with the available literature. First, we focus on the atomistic MD simulations of an epoxy network by Lin and Khare,[19] which appears to be the only work in the literature that has systematically studied the dynamics of the network in the vicinity of the *T*g using atomistic MD simulations. Despite differences in network chemistry, we see the following points of agreement:

- 1. The MSD trends for the CC and XN atoms show qualitative resemblance.
- 2. CC atoms are more mobile than XN atoms by a factor of 1.3, compared to an average of 1.6 seen by us. This difference can be attributed to the difference in the network chemistry since their monomer is significantly shorter and more rigid than that in our system.
- 3. The time-scaling exponent shows two sub-diffusive regimes that vary from about 0.2 to a peak value of 0.5.
- 4. The crossover between the two regimes occurs at an earlier time with an increase in the temperature.

However, they did not observe the expected plateauing due to topological localization that was seen by us. We believe that this lack of observation can be attributed to the limited range in temperatures above *T*g investigated by them since they studied a smaller temperature span (0.88*–*1.3 *T*g) compared to ours (0.72*–*1.6*T*g). Furthermore, since our model structures were larger by a factor of 25 than theirs, we were able to notice distinct trends in the time-scaling exponents and the ratio of the MSDs due to the drastically reduced uncertainty.

Duering et al.[42] studied the local translational dynamics of a coarse-grained polymer network with four different monomer lengths at a single temperature above *T*g. The MSD trends for the monomer and the cross-linking point showed qualitatively similarity, with the former having greater mobility than the latter by a factor of about 1.6, similar to that seen here. In agreement with our work, they observed a time-scaling exponent of about 0.5 (Rouse-like scaling) for both sets, beyond which topological localization occurred. From their work, it can also be seen that for the highest monomer chain length, the exponent of 0.5 was seen for more than one decade in time before the onset of localization.

A somewhat longer sub-diffusive region with an exponent of 0.5 was seen by Kenkare et al.[48] for networks with even longer chains, and thus, the evolution toward localization even was slower than that seen by Duering et al.[42] Indeed, the limiting case for longer chains is an unentangled polymer melt that shows Rouse-like scaling (*t* 0.5) indefinitely.[49] All these coarsegrained simulation results agree with theoretical predictions of Vilgis and Heinrich[50] that network atoms would show Rouselike scaling at short time-scales and topological localization at long time-scales.

The interpretation of the time-scaling exponent of 0.5 as Rouse-like is based on the assumption that the dynamics of the monomer between the covalent cross-links is conceptually equivalent to the entanglement of polymer chains. Thus, Rouse-like dynamics would be seen in the rubbery state before the onset of topological localization (or in the case of entangled chains, reptation dynamics), provided the monomer behaves sufficiently like a Gaussian chain. For the Epon 1001F/4,4′-DDS system, this is clearly not the case. The root-mean-square radius of gyration (*R*g) of the Epon 1001F unit is 9.7 Å, while the *R*e value is 26 Å. The ratio of *R*e/*R*g for Epon 1001F is significantly higher than that for Gaussian chains (2.68 instead of √6 or 2.45). Comparing the *R*e value and *R*e/*R*g ratio of Epon 1001F with *n*-alkanes, the monomers behave roughly like hexatriacontane (*n* = 36) molecules.[51] Despite being somewhat flexible, such short monomers chains are distinctly non-Gaussian. For *n*-alkanes, the crossover to Gaussian behavior has been estimated to be for *n* greater than 100; roughly three times the size of the Epon 1001 molecule.

While the Rouse model is not applicable for the network in the present study, the coarse-grained networks studied by Duering et al.[42] are based on a flexible bead-spring model,[52] which lacks both torsional and bending forces, and shows Rouse-like behavior even for the shortest monomer chain of 12 beads. While our findings show many points of agreement with the results of the coarse-grained models and theory, performing a direct mapping of the results is problematic.

Our models have chemical details that include the formation of hydrogen bonds, realistic potentials for the conformations of the angles and dihedrals, and cohesive interactions

**Table 1.** Comparison of the volumetric and dynamic properties at three cooling rates (also presented in Figure 4).

| Name      | Cooling rate<br>[GK s−1<br>] | tcool [ns] | Tg [K]    | Exponent<br>at Tg | T1 [K] | Exponent<br>at T1 |
|-----------|------------------------------|------------|-----------|-------------------|--------|-------------------|
| Slow—1×   | 5.556                        | 0.9        | 489 ± 1.0 | 0.2               | 620    | 0.47              |
| Medium—3× | 16.67                        | 0.3        | 501 ± 1.8 | 0.2               | 660    | 0.47              |
| Fast—9×   | 50.00                        | 0.1        | 516 ± 3.1 | 0.21              | 710    | 0.47              |

due to partial charges on the network atoms. These details impose severe restrictions on the local translational dynamics of the atoms. Furthermore, since our models undergo glass transition in the studied temperature span, the comparison of the time-scaling exponents with theory becomes even more complicated.

#### **3.3. Comparative Analysis of Dynamic and Volumetric Properties**

Based on the observations of Figure 2, the possibility of superposing these trends to form master curves is visually evident. However, the superposition of MSD trends at atomic length scales appears to be uncommonly performed, and in the literature, we have only found one recent experimental result for an entangled linear polymer in the rubbery state.[53] Furthermore, the application of the TTS principle is associated with many pitfalls that can mask poor superposition and yield results that are erroneous by multiple decades.[17] Hence, we have subjected the trends in Figure 2 to a comparative analysis with the volumetric behavior from our recent work.[9]

As discussed in Section 2.3, the duration (*t*cool) of the simulation at each temperature step of 5 K was varied to study the *v*sp-*T* trend at five *q*cool values (see **Table 1**). In **Figure 4**a, we plot the trends for the slowest three cooling rates. The *t*cool values for the two faster ones are insufficient for comparison. In the same figure, we also show the rubbery equation of state (EoS) from simulation and the experimental trend from the literature.[16] Assuming a constant coefficient of volumetric expansion in the rubbery state, the EoS was obtained by fitting the *v*sp-*T* at *T* > 700 for the slowest cooling rate trend.[9] The excellent agreement between the experimental trend[16] and the prediction by the EoS can be seen.

Using the EoS as a reference, we calculate Δ*v*sp for the three trends as a difference between each simulation trend and the EoS trend. These trends are shown in Figure 4b. At high temperatures, the values of Δ*v*sp for the three trends are negligible. This behavior is characteristic of the rubbery state, where the *v*sp-*T* trend is cooling rate-independent and corresponds to thermodynamic state points.[9] Above a cooling rate-dependent temperature of *T*1, the material is in the rubbery state. Below *T*1, the Δ*v*sp values become finite, and the models are out of equilibrium. At and below the *T*1, the duration of the simulation at each temperature step for the corresponding cooling rate is insufficient to relax the model to equilibrium. The Δ*v*sp values gradually increase as the values of temperature decrease toward *T*g, below which the increase is significantly sharper. The values of both *T*1 and *T*g were obtained from volumetric properties in our recent work (see Table 1).[9]

Since at *T*1, a simulation duration of *t*1 is exactly sufficient for the models to reach equilibrium in response to a temperature perturbation, it can be reasonably expected to find a dynamical signature for rubbery behavior at a temperature of *T*1 and at time *t*1 using Figure 4. The temperature trend in the

**Figure 4.** a) Specific volume (*v*sp) versus temperature (*T*) trends for three simulation cooling rates (Khare and Phelan[9]), rubbery equation of state (EoS), and experiment (Bero and Plazek[16]). b) Deviation of simulation trend from EoS can be used to find *T* at onset of departure from equilibrium (*T*1). c) Trend of time-scaling exponent (*m*) at time corresponding to the cooling rate. Uncertainty is about or less than the thickness of the lines or markers.

time-scaling exponent at time  $t_1$  is shown in Figure 4c. Here, we use the average exponent for the two sets of atoms, and the difference between the two is shown as the thickness of the lines.

In all three figures, dashed lines for three values of  $T_1$  and markers for the ordinate values of the figures at  $T_g$  are shown. The results are summarized in Table 1. The time-scaling exponent of the MSD trends is about 0.47 at the epoch of the rubbery state for all of the three cooling rates.

Below  $T_1$ , the exponent sharply drops as can be seen in Figure 4c. The exponent at  $T_{\rm g}$  is 0.2 and appears to be cooling-rate independent, the theoretical basis for which is unclear. Below the  $T_{\rm g}$ , the dependence of the exponent on temperature is somewhat reduced. At  $T_2$ , we had previously seen that the slope of the  $\nu_{\rm sp}$ -T trends becomes temperature independent in agreement with experiment. The values of the exponent at these temperatures vary between about 0.1 and 0.13, and show  $q_{\rm cool}$  dependence.

From a comparative analysis of the volumetric properties of our recent work  $^{[9]}$  and the dynamic properties of the present work, the values of the exponent at  $T_1$  and  $T_g$  are thus ascertained. It should be noted that unlike first-order phase transitions, the glass transition is substantially gradual, broad, and smooth, especially at simulation time-scales.  $^{[9]}$  We take a utilitarian approach for the identification of specific values of the exponent as dynamic signatures for states or regimes. Such an analysis provides us with a firmer footing for the subsequent superposition of the MSDs and the integration of atomistic simulation and experiment. Since the current work is the first such effort of its kind, considerable further work on other systems is essential to generalize our findings, particularly for the exponent at  $T_g$ .

### 4. Integration of Simulation and Experiment Using Superposition

#### 4.1. Time-Temperature Superposition

As discussed in Section 1, the TTS principle has already been invoked for this system both computationally<sup>[9]</sup> and experimentally for volume-rate dependent processes.<sup>[14–16]</sup> From the discussion in the previous section, the prospect of superposing the MSD and the exponent trends of the CC and XN atoms shown in Figure 2 are evident. Now, we discuss the application of the TTS principle to both the MSD and the time-scaling exponent trends shown in Figure 2.

The superposition of atomic MSD trends appears to be an uncommon application of the principle. We have found one recent work that superposed the MSD trends of polymer segments in a highly entangled melt of a linear polymer.<sup>[53]</sup> While superposition of the time-scaling exponent of the local dynamics does not appear to have been discussed before, the extension of the principle can be anticipated based on the superposition of the MSDs. First, in a thermorheological simple material, the dynamical regime (which is quantified by the time-scaling exponent) at an equivalent reduced time should be the same regardless of the temperature of the component trend. Second, collapsing two or more trends onto a

smooth master curve requires both the coincidence of trends and their first derivative.

A practical guide for TTS is available in the literature.<sup>[17]</sup> While this guide does not directly cover the superposition of MSDs trends, significant guidance can be inferred. We use the following relationships to perform the superposition of the MSD trends<sup>[17,54]</sup>

$$\langle r^2(T_{\text{ref}}, t/a_T) \rangle = b_T \langle r^2(T, t) \rangle$$
 (1)

$$\langle m(T_{\text{ref}}, t/a_{\text{T}}) \rangle = \langle m(T, t) \rangle$$
 (2)

where  $\langle r^2(T,t) \rangle$  is the MSD trend,  $\langle m(T,t) \rangle$  is the time-scaling exponent trend,  $a_{\rm T}$  and  $b_{\rm T}$  are the time-shift and vertical-shift factors, respectively, and  $T_{\rm ref}$  is a convenient reference temperature. The quantity  $t/a_{\rm T}$  is commonly referred to as the reduced time, which accounts for the effect of the acceleration (or deceleration) of the viscoelastic processes at temperatures higher (or lower) than  $T_{\rm ref}$ . With the appropriate  $a_{\rm T}$ -T and  $b_{\rm T}$ -T trends, Equations (1) and (2) transform the MSD and the exponent trends at a spectrum of temperatures to master curves that relate the reduced MSD and the time-scaling exponent with the reduced time at the reference temperature, respectively. The significance of the  $a_{\rm T}$ -T and  $b_{\rm T}$ -T trends is discussed in Sections 4.7 and 4.8, respectively.

Empirically, we started with the MSD and the exponent trends at 500 K. At this temperature, the value of  $a_{\rm T}$  was estimated from the cooling rate dependence of simulation  $T_{\rm g}^{[9]}$  by interpolation. We used a reference temperature of 403.2 K, which is consistent with our recent work<sup>[9]</sup> and the literature.<sup>[14–16]</sup> Analytically calculated<sup>[17]</sup> values of  $b_{\rm T}$  were used to obtain the vertical-shifts of the MSD trends, as is described in greater detail later. The time-scaling exponent trends do not require vertical shifting, which enables an independent validation of successful superposition. Each MSD and exponent trend at a higher temperature was then shifted by an additional factor to form two continuous curves. The time-shift factor used for both the MSD and the exponent trends were identical at all temperatures.

This process was repeated till all the MSD and exponent trends for temperatures above  $T_{\rm g}$  (500, 525, 550, 575, 600, 650, 700, 750, and 800 K) formed two smooth master curves. In order to unmask any thermorheological complexity, first, we visually checked for successful superposition using both the log-log scales and linear-log scale. Second, while the time-scaling exponent trends have somewhat higher uncertainty, these exponents do not require any vertical-shifting. Hence, their superposition is a more stringent test for complexity, especially since the exponent curves have extensive features, such as points of inflection and maximum.

For the four temperature trends below the  $T_{\rm g}$  (350, 400, 450, and 475 K), both shift factors were determined empirically. First, the time-scaling exponent trends were shifted to obtain the  $a_{\rm T}$  trend, since they did not require vertical-shifting. The  $b_{\rm T}$  trend was then obtained by superposing the MSD trends. The progressive shifting of the curves can be seen in the Video, Supporting Information. This protocol was performed for both sets of atoms using the same  $a_{\rm T}$ -T and  $b_{\rm T}$ -T trends to form two master curves each for the MSD and the exponent trends.

15213919, 2020, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.1002/mats.201900032 by Massachusetts Institute of Technolo, Wiley Online Library on [16/02/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License

**Figure 5.** a) Vertically offset MSD and exponent trends versus reduced time from simulations to show quality of superposition. Temperatures [K] are labeled. b) MSD master curves from simulation. c) Time-scaling exponents from simulation. d) Reduced creep compliance from experiment. Error bars are thickness of line, if not shown. Uncertainty estimates are not available for experimental data in panel (d).

In **Figure 5**, these results are presented as a montage where we compare the computational MSD master curves, time-scaling exponent master curve, and experimental creep compliance for the same system. The figure is divided into four parts. In Figure 5a, we show the individual component MSD and exponent trends that compose the master curves for the CC atoms—the trends are vertically offset to illustrate the quality of the superposition effort. The component trends for the XN atoms display a similar high quality. Figure 5b displays the full master curves for the shifted MSD versus reduced time for both sets of atoms, which is the second significant finding. In Figure 5c, we plot the time-scaling exponent (*m*) versus reduced time and use this time-dependence to define transitions in material behavior (Section 4.3). In the same figure, we show the ratio of the reduced MSD values of the CC to the XN atoms. Finally, in Figure 5d for the comparison (Section 4.4), we show the experimental creep compliance master curve for the same system reported by Bero and Plazek.[16] All four of these panels are vertically stacked with a common time axis (the reduced time) and at the same value of *T*ref. We break this montage down pointby-point in the subsequent sections to form an integrated understanding of the relationship between the atomic-scale molecular motions and the macroscopic material behavior in the reduced time-space as it transitions from the glassy to the rubbery regime.

## **www.advancedsciencenews.com www.mts-journal.de**

#### **4.2. MSD Master Curves**

As can be seen in Figure 5a,b and the Video, Supporting Information, excellent superposition of the MSD trends is seen for both the sets of atoms. Each master curve in Figure 5b is composed of the MSD trends from Figure 2a,b; slopes of 0.17 and 0.5 are also shown in the figure as guides for the eye. While the MSD trends in the entire range appear to superpose successfully, the time-scaling exponent trend unmasks some complexity at very short times (*t* < 0.25 ns) in the transition regime, as will be discussed in the next section. We observe that large sections of the 13 trends become indistinguishable from their neighbors as a result of the superposition. Both the CC and XN atoms were superposed with identical shift factor trends. For seven out of the ten decades in the range (10<sup>2</sup> –109 s), three or more MSD trends can be seen to overlap on the master curves at each point. The collapsed master curve is smooth and continuous in both the log-log and the linear-log scale, which is a recommended test for detecting complexity.[17] Furthermore, it can be observed that the exponents of the individual MSD trends change appreciably.[55] All these factors lend confidence to our superposition effort.

It is also observed that the master curves for the two sets of atoms bear a strong qualitative resemblance. This resemblance is expected, since the units form a highly cross-linked network and the translational dynamics of all the atoms are tightly coupled. However, there are some essential quantitative differences that we mentioned in Section 3.1.1, and we discuss further in Section 4.6. Quantitatively, averaged over the two curves, the reduced MSD values range from about 0.7 to 213 Å2. The *t*/*a*<sup>T</sup> values of the master curves range from about 10−<sup>1</sup> s to more than 10<sup>9</sup> s. Thus, the superposition enables us to extend our data to a macroscopic timescale.

#### **4.3. Time-Dependence of the Master Curves**

The time-scaling exponents previously shown in Figure 3c,d also formed smooth and continuous master curves with the same *a*T-*T* trend as that for the two MSD master curves. However, we detected that the eight of the trends at intermediate temperatures (450 K ≤ *T* ≤ 650 K) superpose less satisfactorily in the first decade (*t* ≤ 0.25 ns), but still show excellent superposition in the subsequent 2.2 decades. Accordingly, the trends for the MSDs and the exponents were truncated for creating the master curves. This ability to detect complexity demonstrates the value of using the exponent trends for validation of the superposition. For the CC atoms, the quality can be visualized in Figure 5a, where the untruncated component trends have been vertically offset. TTS of the exponents is a stringent test for successful superposition for three reasons: 1) the exponents do not require vertical shifting; 2) the first derivative of the MSDs is a far more sensitive test for smoothness rather than visually inspecting the MSD master curve; and finally 3) the exponent trends have far more features.[55]

Since the MSD master curves for the two sets are now available, we could reduce the uncertainty in the exponent master curves by calculating the first derivative of the MSD master curves, as was described for the MSD trends in Section 2. The

**Table 2.** log10 (*a*T/*t*) values when the time-scaling trends for the CC and XN atoms show features of interest.

| Feature                      | Trend | log10 (aT/t) |
|------------------------------|-------|--------------|
| Inflection points (m = 0.17) | CC    | 2.3          |
|                              | XN    | 2.8          |
| Maximums (m = 0.5)           | CC    | 6.5          |
|                              | XN    | 6.7          |

These values are identified as the soft boundaries between the glassy, transition, and rubbery dynamical regimes (dashed lines in Figure 5b–d).

exponent master curves derived from the MSD master curves were compared and found to be identical to the exponent master curves constructed from the individual components.

A plot of the resulting time-scaling exponent (*m*) versus *t*/*a*<sup>T</sup> for both sets of atoms is shown in Figure 5c. The trend for either set shows broadly similar behavior: 1) an initial region of gradual increase terminating at an inflection point; 2) a central region of first rapid increase and then decay terminating at a maximum; and 3) a final region of rapid decrease reflecting the onset of the localization plateau in the MSD trends.

As seen in Section 3.1.3 for the temperature trends, while the trends for the two sets of atoms are similar, it is also evident that they are slightly out of phase. Both the inflection and the maximum points for the CC atoms finitely precede those of the XN atoms. The values of the reduced time at the inflection points and the maximums for the two sets of atoms are shown in **Table 2**. The significance of this phase difference is further discussed in Section 4.6. The values of *m* at the inflection points (0.17) and the maximums (0.5) for the two trends are about the same.

The inflections points and the maximums seen in the simulations define transitions in the material behavior from the glassy to the rubbery dynamical regime. To illustrate this, in Figure 5b,c (also extending into Figure 5d), we overlay two dashed vertical lines at the values of *t*/*a*T corresponding to the inflection points and the maximums for each of the two trends. The lines for the CC atoms are shown in red, while those for the XN atoms are shown in blue. As can be seen in the figures, the two inflection point lines and the two maximum lines are paired, and these pairs demarcate the boundaries between three distinct regimes of material behavior. Regimes 1 and 2 are separated by the inflection points of the two *m* trends, while regimes 2 and 3 are separated by the two maximums.

However, the separation between the two inflections points and two maximums for the different types of atoms shows that the boundaries between these regimes are soft and that the transition is heterogeneous. It is evident that these regimes successively correspond to the glassy, the transition, and the rubbery dynamical behavior. These softer criteria are more consistent with the behavior of glass-forming materials than the sharp criteria observed in first-order phase transitions. We have first presented a comparison of the parts of Figure 5. In the subsequent two sections, we have discussed the lengthscales associated with the regimes and the differences in the time-scaling exponent trends for the two sets of atoms. Such discussion provides additional validation for the existence of these soft boundaries.

#### **4.4. Comparison of Atomic Dynamics from Simulation and Creep Compliance from Experiment**

TTS has enabled us to extend our simulation data out to 10<sup>9</sup> s, which matches the timescale of the creep compliance master curve of Bero and Plazek.[16] Thus, a quantitative comparison between the simulation and experimental data sets is now possible, which is depicted in Figure 5b,d. Such comparison gives us an integrated view relating the atomic-scale motions and the macroscopic material behavior. We discuss this for the three regimes marked out in the previous section.

#### *4.4.1. Glassy Regime*

During the first few decades (from 10<sup>−</sup><sup>1</sup> to about 102.3 s) in the reduced time, the signature of the glassy regime is observed in Figure 5b,d. Specifically, the average MSD values increase from about 0.7 to 2 Å2 (see Figure 5b). These relatively low values suggest molecular caging. In Figure 5c, the time-scaling exponent trends increase slightly from 0.1 to 0.17. Interestingly, the exponents for the two atoms are essentially the same, suggesting strong coupling in the dynamics. Correspondingly, the experimental creep compliance master curve in Figure 5d shows low values typical of the glassy behavior and a similarly modest increase in its time-dependence.

#### *4.4.2. Transition Regime*

Over the next roughly four decades (102.6–106.5 s) in reduced time, a significant transition in the behavior of the network is observed in Figure 5b,d. 1) The average MSD values show a significant increase from about 2 to 40 Å2; this is marked by a perceptible upturn in the MSD that is noticeable in Figure 5b inset at about 10<sup>5</sup> s. 2) The time-scaling exponent (*m*) increases sharply from 0.17 to 0.5. Interestingly, the exponents of the two atoms diverge at the beginning of this regime and somewhat converge toward the end. 3) The experimental creep compliance increases dramatically by a factor of about 25 and attains the rubbery plateau. This transition is especially appreciable in the inset to Figure 5d, which uses a linear-log scale.

#### *4.4.3. Rubbery Regime*

Finally, after about 106.7 s of reduced time, we see clear indications of rubbery behavior. The average MSD values increase sharply to molecular length-scales from about 40 to 200 Å2, as can be seen in the inset of Figure 5b. Subsequently, the curves show plateauing due to the topological localization. In Figure 5c, the time-dependence is seen to peak and then decline due to the localization. Finally, in Figure 5d, we see that the experimental creep compliance trend remains steady at the rubbery plateau.

Comparison of Figure 5c,d indicates that the molecular origin of the onset of the rubbery plateau corresponds to the point where diffusive behavior maximizes and then localizes at high terminal values of the MSD. At these elevated temperatures/long times, the molecular units of the network explore all available conformational space (see Video, Supporting Information). This view is also clearly consistent with the comparison to the volumetric data shown in Figure 4c.

#### *4.4.4. Topological Localization Plateau*

About a decade in the reduced time after the epoch of the rubbery behavior, the MSD master curves begin to noticeably plateau due to topological localization. The peak and decline in the values of the exponent in Figure 5c at the epoch of the rubbery regime indicate that to some degree, localization starts at this epoch, even though the effect on the reduced MSD trends is only gradually seen. Because of the highly cross-linked nature of the matrix, the atoms of the network quickly exhaust the limited conformational space available for sampling. Thus, even though the atoms are still highly mobile, the MSD trends plateau.

For networks composed of Gaussian monomer chains, the localization plateau in the MSDs of the monomer is related to the rubbery plateau in the compliance. In the case of the present system composed of non-Gaussian Epon 1001F monomers, the rubbery plateau in the compliance begins with the peak in the MSDs and extends through the localization plateau; no corresponding change in the creep compliance is seen with the advent of the topological localization. It is interesting to compare our findings about the topological localization with the coarse-grained simulation of rubbery polymer networks by Duering et al.[42] For those networks, which were composed of flexible Gaussian chains, it can be seen that the topological plateaus were associated with plateaus in the modulus, as expected. While this relationship was seen to be strong for the longest monomer chain, it appeared to weaken gradually as the length of the chain decreased.

For Gaussian chains, any deformation-induced stress generates entropic restoring forces, that are dissipated through conformational sampling. Since the translational aspect of the conformational space is quantified by the MSDs, the rubbery plateau would be associated with the localization plateau of the MSDs of the monomers, but not necessarily the corresponding plateau for the cross-linkers. As the monomer chain length increases, the onset of the MSD localization plateaus of the monomers and the cross-linkers can be expected to diverge. For the limiting case of an infinitely long monomer, it has been suggested that even though the MSD of the cross-linkers would still show a plateau, no corresponding plateau would be seen for the MSD of the monomers.[42]

On the other hand, as the monomer length reduces, the dynamics of the monomers and the cross-linkers become more strongly coupled, which was observed even for Gaussian chains. The assumption of affine network model fails, and the phantom network model, which considers the fluctuations of network junction, becomes more appropriate instead.[42] The non-Gaussian monomer in our study (Epon 1001F) is a far more extreme case of this coupling. The length-scales of the monomer and the cross-linker are comparable, and their dynamics in the rubbery regime are highly correlated. For

epoxy networks, front-factors are used to account for the constrained mobility of the monomers during the application of the theory of rubbery elasticity.[56] It is, thus, reasonable to expect that the ability of the network to dissipate stress is more directly related to the correlation in the dynamics of the monomer and the cross-linker, and less related to the conformational sampling of the monomer. In the case of our network model, this correlation is achieved when diffusive behavior maximizes, which is discussed even further in Sections 4.5 and 4.6.

#### *4.4.5. Empirical Time-Dependence versus Theory*

The designation of the *t* 0.17 and *t* 0.5 dependence as soft boundaries for the three regimes is based on empirical observations that emerge from our simulations (Figure 5b–d and Table 2). For a different epoxy network, Lin and Khare[19] also noticed the existence of an extended subdiffusive regime with an exponent of less than 0.2 at temperatures below the *T*g, followed by an increase to a value of 0.5 above the *T*g. Preliminarily, this suggests that the MSD trends of other epoxy networks have similar features, which provides a new direction for future investigations. However, since TTS was not considered in that work, they did not make any further connection of these time-scaling exponents with the glass transition.[19]

As discussed earlier in Section 3.2, there is not any clear connection of these results to existing theory. Neither the Rouse model nor the Reptation model is applicable to the cross-linked epoxy in the present study due to the underlying assumption of Gaussian chain statistics in those models.[57] As a frame of reference, the *t* 0.25 dependence was predicted by de Gennes[58] for Rouse-like diffusion within tube constraints,[57] and the *t* 0.5 dependence was predicted by primitive chain dynamics for entangled polymer melts.[57] Our results provide strong motivation for the further use of atomistic MD simulations of model polymer network systems to enhance the theoretical and conceptual understanding of the evolution of the local dynamics during the glass transition.

#### **4.5. Length-Scales across Reduced Time**

It is interesting to examine the length-scales of the local translational dynamics in the three regimes in comparison with other length-scales in the system. Here, the length-scale of the dynamics is quantified by the square root of the MSD (√MSD).[59] We calculated √MSD values at 1) 10−<sup>1</sup> s; 2) the inflection points of the exponent trends; and 3) the maximums of the exponent trends; and finally 4) the plateau values at 109 s. We also looked at the following length-scales for context: 1) the average covalent radius of carbon and nitrogen atoms;[60] 2) the average vdW radius of both the atoms;[47] and 3) the *R*e values of the Epon 1001F and 4,4′-DDS units in the network at a temperature of 800 K.[61]

In **Table 3**, all these values have been tabulated in ascending order. We find that the √MSD value at 10−<sup>1</sup> s in reduced time (the glassy regime) is comparable to the average covalent radius of the atoms. The average covalent radius of the atoms (i.e., the

**Table 3.** Relevant length-scales. The dynamics track length-scales associated with the glassy, the transition, and the rubbery regimes.

| Length-scale              | Value [Å] |  |  |
|---------------------------|-----------|--|--|
| Covalenta) radius         | 0.74      |  |  |
| √MSD at 10−1 s (glass)    | 0.8       |  |  |
| √MSD at inflection points | 1.4       |  |  |
| vdWb) radius              | 1.60      |  |  |
| √MSD at maximums          | 5.8       |  |  |
| Re of 4,4′-DDS            | 9.3       |  |  |
| √MSD at 109 s (rubber)    | 14.6      |  |  |
| Re of Epon 1001F          | 26        |  |  |

a)Cordero et al.[60]; b)Bondi.[47]

typical bond length) is a lower bound for MSD values that are relevant to network dynamics. Length-scales below the covalent radius correspond to the ballistic motion of the atoms and are only weakly related to the viscoelastic regime of the network. At this low value of the reduced time (10−<sup>1</sup> s), the atoms vibrate within their molecular cages, which are smaller than the vdW radius of the atoms. As the reduced time increases, the lengthscale of the local translational dynamics slowly increases. At the epoch of the transition regime (the inflection points), the √MSD value is still somewhat less than the average vdW radius of the atoms.

In the transition regime, the atoms begin to break free from their cages, and at the epoch of the rubbery regime, the √MSD value is about 5.8 Å. To justify this value, we consider the case of an average Epon 1001F unit being stretched by a strain of 20% along its end-to-end vector. Thus, the *R*e value would increase from 26 Å by about 5.2 Å, which is similar to the √MSD value at the maximums. Thus, the length-scale of the translational dynamics at the epoch of the rubbery regime would be sufficient to relax this unit back to 26 Å, even after assuming a somewhat higher value than typical for the maximum strain at the linear viscoelastic limit for epoxy networks. The similarity of these length-scales strongly validates the correspondence of peak diffusivity of the atoms seen here with the onset of the rubbery plateau in the experiment.

In the rubbery regime, the atoms and the molecular units have significantly higher mobility. After the onset of localization, the plateau √MSD value is somewhat higher than the *R*<sup>e</sup> distance of the rigid and short 4,4′-DDS units but much smaller than that of the Epon 1001F units. This trend suggests that the maximum extent of the local translational dynamics overall is limited by the stretching of the Epon 1001F molecular units.

Thus, the trend of the length-scales associated with the local translational dynamics of the atoms with reduced time is entirely consistent with the glassy, transition, and rubbery regimes of the network, which we have discussed earlier. This internal consistency is independent of the experimental creep compliance. We believe that the consideration of these lengthscales is a useful reasonableness "check" during the integration of molecular and macroscopic perspectives across these mismatched length-scales.

## Macromolecular Theory and Simulations www.mts-journal.de

#### 4.6. Topology-Induced Asynchronous Dynamics

The focus of the discussion is now returned to the two time-scaling exponent trends (m) with reduced time ( $t/a_T$ ) shown in Figure 5. As discussed in Section 4.3, each trend has an inflection point and a maximum point, which are separated by more than three decades in the reduced time. For both trends, the values of m at the two inflection points and the two maximums are about the same. However, the two trends are clearly out of phase with respect to the  $t/a_T$  values at which the inflections and maximums occur. This phase difference is shown by the pairs of dashed lines in Figure 5c. We have identified these pairs of lines as soft boundaries for the glassy, transition, and rubbery regimes.

This out-of-phase behavior can be attributed to the differing topological constraints of the two sets of atoms. In the glassy regime, all of the atoms experience caging, and hence the dynamics of the atoms in the network is tightly coupled with that of their neighbors. Hence, as can be seen in Figure 5c, the exponent trends for the two networks are essentially the same, and the MSD ratio trend is roughly constant.

In the transition regime, the atoms begin to break free and to explore the conformational space. Since the CC atoms have higher mobility than the XN atoms, as the network leaves the glassy regime, the CC atoms break free from the molecular cages before the XN atoms. Thus, the exponent trends diverge, and the MSD ratio shows a sharp increase, as can be seen in Figure 5c.

The dynamics of each rigid cross-linker requires correlated motion of the four covalently attached epoxy monomers. In the transition state, this need for this correlation suppresses the dynamics of the XN atoms compared to the CC atoms, which explains the delay in the crossover of the time-scaling exponent of the XN atoms compared to that for the CC atoms. When the exponent trends of the XN and the CC atoms intersect, the dynamics of the two become more correlated, and thus the MSD ratio was seen to decrease just before the onset of the rubbery state. As the XN atoms become increasingly mobile, the CC atoms achieve peak diffusivity, quickly followed by the XN atoms. In this highly correlated state, the molecular units now become sensitive to topological constraints and hence trigger topological localization.

The need for such correlated motion of the cross-linker has significant repercussions for the thermomechanical behavior of networks. For example, suppressing the need for this dynamical correlation between the cross-linker and the monomer in a similar epoxy network by substituting the 4,4′-DDS cross-linker with the corresponding meta analog (3,3′-DDS) was observed to drastically decrease the  $T_{\rm g}$  by nearly 50 K and to significantly enhance toughness. [62] These changes occurred even though the cross-link density and other aspects of the chemistry were unchanged.

As mentioned earlier, we term this temporal difference of the CC and the XN atoms TIA dynamics. The TIA dynamics is distinct from the dynamic heterogeneity phenomenon,<sup>[3,19,63,64]</sup> which is also implicated in the glass transition. TIA dynamics is a temporal difference in the dynamics of atoms that vary in their topological constraints and associated dynamical correlation, whereas the dynamic heterogeneity refers to a spatial feature

**Figure 6.** a) Time-shift factor  $(a_{\rm T})$  versus T. At T below the simulation  $T_{\rm g}$ , the simulation  $a_{\rm T}$ -T trend diverges from the other trends, as expected. Four master curves needed identical  $a_{\rm T}$ -T trend. b) Vertical-shift factor  $(b_{\rm T})$  versus temperature (T) using Equation (3) at  $T < T_{\rm g}$ . Unlike the  $a_{\rm T}$ 's, the  $b_{\rm T}$ 's were not needed for the exponent trends, and this was used to obtain trend below  $T_{\rm g}$ . Note the logarithmic axis for  $a_{\rm T}$  compared to the linear one for  $b_{\rm T}$ . Experimental data set is from the literature.

of the local dynamics in a system near the glass transition. We believe that the TIA dynamics is a characteristic of highly cross-linked polymers, where non-Gaussian molecular units of disparate lengths and flexibilities are covalently bound to form a network. In the future, it would be interesting to explore this phenomenon in the context of ongoing research<sup>[38,65,66]</sup> on the use of mixed networks for ballistics applications.

#### 4.7. Time-Shift Factors $(a_T)$

In Section 4.1, we discussed the superposition of the MSD and the exponent trends to determine the  $a_{\rm T}$ -T trend empirically. This  $a_{\rm T}$ -T trend is shown in **Figure 6a**. The values of  $a_{\rm T}$  for both the MSD and the exponent master curves for both the two sets of atoms are identical, as would be expected in the case of thermorheological simplicity. This trend splices neatly with the trend that we previously calculated using  $v_{\rm sp}$ - $q_{\rm cool}$  analysis for the same network as can be seen in the figure. [9] Also in that figure, we show 1) the experimental  $a_{\rm T}$  values extracted from the literature; [16] 2) the  $a_{\rm T}$  values needed to superpose the experimental creep compliance trends in the literature; [16] and 3) the material-specific  $a_{\rm T}$ - $a_{\rm T}$  trend calculated using the WLF [67] equation.

A gap exists between the experimental and computational time-shift factors, which arises because of the vast mismatch in the timescales accessed by the two methods. Nevertheless, as can be seen in Figure 6a, this gap is neatly bridged by the WLF

time-shift factor trend. At temperatures below the simulation  $T_{\rm g}$  value, the time-shift factors obtained by superposing the simulation dynamics deviate from the experimental and the WLF trends, as was expected. Below the simulation  $T_{\rm g}$ , the atoms of the model networks are trapped in molecular cages, and the  $\alpha$ -relaxation of the network essentially ceases. Consistent with this observation, the response of the specific volume to temperature diminishes, as seen in Figure 4. [9] Hence, unlike the trend above the simulation  $T_{\rm g}$ , the  $\alpha_{\rm T}$ -T trend does not correspond to the material-specific relaxation time, and thus cannot be compared for different experiments/simulations.

These findings are also consistent with that of Sirk et al. [18] While we have seen agreement between the WLF and the simulation  $a_{\rm T}$ -T trends at  $T > T_{\rm ref} + 150$ , this result is potentially fortuitous and not to be generally expected. Neither the experimental nor the WLF time-shift factors were used to guide our superposition effort here. The simulation  $a_{\rm T}$ -T trend was obtained independently from the superposition. As can be seen in the Video, Supporting Information, the ability to form a master curve is acutely sensitive to small variations in the time-shift factors

#### 4.8. Vertical-Shift Factors ( $b_T$ )

We followed the recommendations of Dealy and Plazek<sup>[17]</sup> for the vertical-shift factors ( $b_{\rm T}$ ) above the  $T_{\rm g}$  (Figure 6b). The  $b_{\rm T}$  accounts for the relatively weak dependence of the virial stress magnitude on  $T^{[17]}$  In the rubbery regime, virial stress is the primary source of molecular friction that impedes the dynamics in the network. As the temperature increases, this factor mildly retards the dynamics. However, this retardation will be rather marginal compared to the accelerating effect of increasing temperatures on the dynamics.<sup>[17]</sup> At  $T > T_{\rm g}$ , we use the following relationship for the vertical-shift factors

$$b_{\rm T} = \frac{\rho T}{\rho_{\rm ref} T_{\rm ref}} \tag{3}$$

where  $\rho$ ,  $\rho_{\rm ref}$ , and  $\rho_{T_g}$  are the densities at temperatures T,  $T_{\rm ref}$ , and  $T_{\rm g}$ , respectively. The value of density at the reference temperature is obtained from the EoS.<sup>[9]</sup>

Unlike the MSD trends, TTS of the time-scaling exponent does not require vertical shifts. This fact has two advantages: 1) above  $T_{\rm g}$ , the successful superposition of the exponent trends at the same  $a_{\rm T}$ -T trends as that for the MSD trends validates the use of the equation; and 2) below  $T_{\rm g}$ , the  $a_{\rm T}$ -T trend can be determined from the superposition of the exponent trends, and then the  $b_{\rm T}$ -T trend can be independently determined from the superposition of the MSD trends.

The simultaneous empirical fitting of both  $a_{\rm T}$  and  $b_{\rm T}$  causes both factors to lose physical significance, [17] and such fitting is highly discouraged. Above the  $T_{\rm g}$ , we have made no attempts to improve our superposition effort by altering the values of  $b_{\rm T}$  from those calculated using Equation (3).[17] Below the  $T_{\rm g}$ , the empirical fitting of the two trends could be performed independently.

In the context of the broad range in the temperatures used in this work ( $\Delta T = 450$  K), the vertical-shift factors used are

relatively small. Between the simulation  $T_{\rm g}$  value of 489 K and the highest investigated temperature of 800 K, the value of  $1/a_{\rm T}$  increases by a factor of about  $3.3 \times 10^4$ , while the value of  $b_{\rm T}$  only increases by a factor of 1.4. Preliminary tests have also shown that alterations to the values of  $b_{\rm T}$  have a negligible impact on the  $a_{\rm T}$ -T trend of the network above  $T_{\rm g}$ . The simultaneous superposition of the MSD and the exponent trends has, thus, enabled a robust basis for determining both the shift factors and validating the superposition.

#### 5. Conclusions

To summarize, we studied the MSD trends of two sets of atoms that differ in their topological constraints, in a cross-linked epoxy network using atomistic MD simulations. For both the sets, we find that the MSD trends below the  $T_{\rm g}$  show a relatively low time-dependence, which is characteristic of the glassy behavior and is consistent with previous work. [19] Above the  $T_{\rm g}$ , there is a sharp increase in the MSD trends with time, and peak time-dependences of about  $t^{0.5}$  are seen. Finally, at longer times in the rubbery regime, the MSD trends show plateauing due to topological localization. These observations are corroborated by the behavior of the time-scaling exponent trends.

We find that TTS can be used for both the MSD and timescaling exponent trends to form master curves. The temporal features of the reduced master curves in simulations show excellent quantitative agreement with the experimental creep compliance master curve in the literature. [16] We show that the molecular origin of the onset of the rubbery plateau corresponds to the point where diffusive behavior of the atoms maximizes. We also show that the atoms which differ in their topological constraints exhibit asynchronous dynamics. We have called this feature TIA dynamics and have used this to identify the soft boundaries for the transition between the glassy, transition, and rubbery regimes in the TTS-reduced time-space. Furthermore, we find that the time-shift factors needed to obtain the master curves using simulation data show excellent agreement with those obtained from both experimental data<sup>[16]</sup> in the literature and our recent work<sup>[9]</sup> on the specific volume–cooling rate analysis.

Altogether, such quantitative comparison between atomistic MD simulations and experiments presents an integrated view relating the MD of the network with its macroscale viscoelastic characterization. Here, we show empirical evidence that such an approach is indeed productive. To the best of our knowledge, this is the first such report for network polymers.

Besides presenting a possible method to extend nanosecond simulations to macroscale timescales, these results should also be of interest to experimentalists. Many measurement techniques in current use infer structure and properties from the atomistic level dynamics, for example, NMR and quasi-elastic neutron scattering. The results obtained here indicate that measurements of higher level segmental features could also be useful.

In the literature, [68] a decoupling of the segmental and chain dynamics for various polymers has been reported, which causes a breakdown in thermorheological simplicity and a failure of TTS. We believe that due to the elevated temperatures used in our simulations and the highly cross-linked nature of the network, the two modes of dynamics remain coupled.

The evaluation of thermorheological simplicity/complexity of materials is contextual.[17] Here, we are interested in bridging the vast mismatch in time-scales accessible via atomistic simulation and experiment for quantitative integration, and the assumption of simplicity is justified.

Already, TTS is frequently used to compare simulation and experimental values of *T*g despite the vast differences in the cooling rates, for various polymers.[5,8–10,18,35,36,38] However, considerable further work will be necessary to test the general applicability of the TTS principle in other cross-linked networks and other glass-forming systems. Since the analysis of different sets of atoms can be isolated in simulations, the thermorheological complexity arising from the motion of the main chains, phenyl rings, and other side groups can be separately characterized. Experimental thermomechanical measurements typically cannot access such details. We plan to follow-up with other measures of translational and rotational dynamics of this network in future work.

#### **Supporting Information**

Supporting Information is available from the Wiley Online Library or from the author.

#### **Acknowledgements**

K.S.K. gratefully acknowledges the financial and facilities support via the Material Measurement Laboratory Professional Research Experience Program (PREP-MML) at the U.S. National Institute of Standards and Technology (NIST) via grant number 70NANB16H005 to Georgetown University. This work used the Extreme Science and Engineering Discovery Environment (XSEDE), which was supported by the U.S. National Science Foundation grant number ACI-1053575.[69] Certain commercial materials are identified in this article to foster understanding. Such identification does not imply recommendation or endorsement by NIST, nor does it imply that the materials identified are necessarily the best available for the purpose. Official contribution of NIST – not subject to copyright in the United States.

#### **Conflict of Interest**

The authors declare no conflict of interest.

#### **Author Contributions**

The simulation and analysis strategy was executed by K.S.K. under the supervision of F.R.P. Jr. Both authors were involved in the project planning and preparation of the manuscript.

#### **Keywords**

epoxy, glass transition, polymer dynamics and mechanics, molecular dynamics simulation, thermosets

> Received: June 19, 2019 Revised: September 13, 2019 Published online: December 18, 2019

- [1] C. Li, A. Strachan, *J. Polym. Sci., Part B: Polym. Phys.* **2015**, *53*, 103.
- [2] The White House Office of Science and Technology Policy, *Materials Genome Initiative: Strategic Plan*, Washington, DC **2014**.
- [3] K. S. Khare, R. Khare, *J. Phys. Chem. B* **2013**, *117*, 7444.
- [4] K. S. Khare, F. Khabaz, R. Khare, *ACS Appl. Mater. Interfaces* **2014**, *6*, 6098.
- [5] K. S. Khare, R. Khare, *Macromol. Theory Simul.* **2012**, *21*, 322.
- [6] S. Kirkpatrick, C. D. Gelatt, M. P. Vecchi, *Science* **1983**, *220*, 671.
- [7] R. Khare, M. E. Paulaitis, S. R. Lustig, *Macromolecules* **1993**, *26*, 7203.
- [8] P.-H. Lin, R. Khare, *Macromolecules* **2009**, *42*, 4319.
- [9] K. S. Khare, F. R. Phelan Jr., *Macromolecules* **2018**, *51*, 564.
- [10] A. Soldera, N. Metatla, *Phys. Rev. E* **2006**, *74*, 061803.
- [11] P. N. Patrone, A. Dienstfrey, A. R. Browning, S. Tucker, S. Christensen, *Polymer* **2016**, *87*, 246.
- [12] Epon 1001F is a chemical trade name for the diglycidyl ether of bisphenol A (DGEBA) with a degree of polymerization of 2.
- [13] Registered Trademark of Hexion, Inc., Corporation, 180 East Broad St., Columbus, OH.
- [14] I.-C. Choy, D. J. Plazek, *J. Polym. Sci., Part B: Polym. Phys.* **1986**, *24*, 1303.
- [15] D. J. Plazek, I.-C. Choy, *J. Polym. Sci., Part B: Polym. Phys.* **1989**, *27*, 307.
- [16] C. A. Bero, D. J. Plazek, *J. Polym. Sci., Part B: Polym. Phys.* **1991**, *29*, 39.
- [17] J. Dealy, D. Plazek, *Rheol. Bull.* **2009**, *78*, 16.
- [18] T. W. Sirk, K. S. Khare, M. Karim, J. L. Lenhart, J. W. Andzelm, G. B. McKenna, R. Khare, *Polymer* **2013**, *54*, 7048.
- [19] P.-H. Lin, R. Khare, *Macromolecules* **2010**, *43*, 6505.
- [20] J. D. Lemay, B. J. Swetlin, F. N. Kelley, in *Characterization of Highly Cross-Linked Polymers* (Eds: S. S. Labana, R. A. Dickie), American Chemical Society, Washington, DC **1984**, pp. 165–183.
- [21] S. Plimpton, *J. Comput. Phys.* **1995**, *117*, 1.
- [22] J. Wang, W. Wang, P. A. Kollman, D. A. Case, *J. Mol. Graph. Model.* **2006**, *25*, 247.
- [23] J. Wang, R. M. Wolf, J. W. Caldwell, P. A. Kollman, D. A. Case, *J. Comput. Chem.* **2004**, *25*, 1157.
- [24] General Amber force field (version 1.8, March 2015) included in AmberTools 16.
- [25] M. J. S. Dewar, E. G. Zoebisch, E. F. Healy, J. J. P. Stewart, *J. Am. Chem. Soc.* **1985**, *107*, 3902.
- [26] A. Jakalian, B. L. Bush, D. B. Jack, C. I. Bayly, *J. Comput. Chem.* **2000**, *21*, 132.
- [27] A. Jakalian, D. B. Jack, C. I. Bayly, *J. Comput. Chem.* **2002**, *23*, 1623.
- [28] H. Sun, *J. Phys. Chem. B* **1998**, *102*, 7338.
- [29] R. W. Hockney, J. W. Eastwood, *Computer Simulation Using Particles*, Institute of Physics Publishing, Philadelphia, PA **1988**.
- [30] S. Nosé, *J. Chem. Phys.* **1984**, *81*, 511.
- [31] W. G. Hoover, *Phys. Rev. A* **1985**, *31*, 1695.
- [32] W. Shinoda, M. Shiga, M. Mikami, *Phys. Rev. B* **2004**, *69*, 134103.
- [33] H. C. Andersen, *J. Comput. Phys.* **1983**, *52*, 24.
- [34] J.-P. Ryckaert, G. Ciccotti, H. J. C. Berendsen, *J. Comput. Phys.* **1977**, *23*, 327.
- [35] P.-H. Lin, R. Khare, *J. Therm. Anal. Calorim.* **2010**, *102*, 461.
- [36] N. J. Soni, P.-H. Lin, R. Khare, *Polymer* **2012**, *53*, 1015.
- [37] F. Khabaz, K. S. Khare, R. Khare, *AIP Conf. Proc.* **2014**, *1599*, 262.
- [38] T. W. Sirk, M. Karim, K. S. Khare, J. L. Lenhart, J. W. Andzelm, R. Khare, *Polymer* **2015**, *58*, 199.
- [39] G. Wisanrakkit, J. K. Gillham, *J. Appl. Polym. Sci.* **1990**, *41*, 2885.
- [40] The term "simulated annealing" is also used by other researchers in computational material science to refer to step-wise cooling (i.e., annealing) of models using simulations. Here, we consistently use the term to refer to the optimization strategy.
- [41] S. Izrailev, S. Stepaniants, M. Balsera, Y. Oono, K. Schulten, *Biophys. J.* **1997**, *72*, 1568.

- **www.advancedsciencenews.com www.mts-journal.de**
- [42] E. R. Duering, K. Kremer, G. S. Grest, *J. Chem. Phys.* **1994**, *101*, 8169.
- [43] LAMMPS manual, Sandia National Laboratory, rerun command, <https://lammps.sandia.gov/doc/rerun.html>(accessed: September 2019).
- [44] These windows are not statistically independent and have only been used to reduce noise. Uncertainty estimates are based on independent replicas.
- [45] The first 10 ns of each trajectory was discarded to minimize aging effects. Within the uncertainty of our results, this protocol had no significant impact on the results reported here.
- [46] A. Rohatgi, *WebPlotDigitizer*, Version 3.10, Austin, TX, US **2016**.
- [47] A. Bondi, *J. Phys. Chem.* **1964**, *68*, 441.
- [48] N. R. Kenkare, S. W. Smith, C. K. Hall, S. A. Khan, *Macromolecules* **1998**, *31*, 5861.
- [49] A. V. Lyulin, N. K. Balabaev, M. A. J. Michels, *Macromolecules* **2002**, *35*, 9595.
- [50] T. A. Vilgis, G. Heinrich, *Phys. Rev. E* **1994**, *49*, 2167.
- [51] M. Mondello, G. S. Grest, E. B. Webb, P. Peczak, *J. Chem. Phys.* **1998**, *109*, 798.
- [52] K. Kremer, G. S. Grest, *J. Chem. Phys.* **1990**, *92*, 5057.
- [53] A. Lozovoi, C. Mattea, M. Hofmann, K. Saalwaechter, N. Fatkullin, S. Stapf, *J. Chem. Phys.* **2017**, *146*, 224901.
- [54] J. Ferry, *Viscoelastic Properties of Polymers*, 3rd ed., Wiley, New York **1980**.
- [55] The following qualities of the component trends would tend to reduce any ambiguity in their superposition: 1) high values of slope and 2) variation in the slope. For illustration, consider the cases of the superposition of 1) two vertical lines; 2) two horizontal lines; and 3) two curves.

- [56] D. Katz, A. V. Tobolsky, *Polymer* **1963**, *4*, 417.
- [57] M. Doi, S. F. Edwards, *The Theory of Polymer Dynamics*, Clarendon Press, Oxford, UK **1988**.
- [58] P. G. de Gennes, *J. Chem. Phys.* **1971**, *55*, 572.
- [59] For this analysis, we have used the average of the MSD values from the two master curves to find the √MSD values. The use of either of the two curves does not alter any of the observations.
- [60] B. Cordero, V. Gómez, A. E. Platero-Prats, M. Revés, J. Echeverría, E. Cremades, F. Barragán, S. Alvarez, *Dalton Trans.* **2008**, 2832.
- [61] Interestingly, the distribution of the end-to-end distances of the molecular units only show modest temperature dependence.
- [62] A. C. Grillet, J. Galy, J.-F. Gérard, J.-P. Pascault, *Polymer* **1991**, *32*, 1885.
- [63] D. Long, F. Lequeux, *Eur. Phys. J. E: Soft Matter Biol. Phys.* **2001**, *4*, 371.
- [64] G. Lois, J. Blawzdziewicz, C. S. O'Hern, *Phys. Rev. E* **2009**, *102*, 015702.
- [65] D. B. Knorr, J. H. Yu, A. D. Richardson, M. D. Hindenlang, I. M. McAninch, J. J. La Scala, J. L. Lenhart, *Polymer* **2012**, *53*, 5917.
- [66] T. W. Sirk, M. Karim, J. L. Lenhart, J. W. Andzelm, R. Khare, *Polymer* **2016**, *90*, 178.
- [67] M. L. Williams, R. F. Landel, J. D. Ferry, *J. Am. Chem. Soc.* **1955**, *77*, 3701.
- [68] Y. Ding, A. P. Sokolov, *Macromolecules* **2006**, *39*, 3322.
- [69] J. Towns, T. Cockerill, M. Dahan, I. Foster, K. Gaither, A. Grimshaw, V. Hazlewood, S. Lathrop, D. Lifka, G. D. Peterson, R. Roskies, J. R. Scott, N. Wilkins-Diehr, *Comput. Sci. Eng.* **2014**, *16*, 62.