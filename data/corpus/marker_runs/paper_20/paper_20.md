PAPER View Article Online
View Journal | View Issue

**Cite this**: *Soft Matter*, 2017, **13**. 8392

Received 22nd August 2017, Accepted 19th October 2017

DOI: 10.1039/c7sm01695f

rsc.li/soft-matter-journal

# Topological structure and mechanics of glassy polymer networks

Robert M. Elder pab and Timothy W. Sirk \*\*

The influence of chain-level network architecture (i.e., topology) on mechanics was explored for unentangled polymer networks using a blend of coarse-grained molecular simulations and graph-theoretic concepts. A simple extension of the Watts-Strogatz model is proposed to control the graph properties of the network such that the corresponding physical properties can be studied with simulations. The architecture of polymer networks assembled with a dynamic curing approach were compared with the extended Watts-Strogatz model, and found to agree surprisingly well. The final cured structures of the dynamically-assembled networks were nearly an intermediate between lattice and random connections due to restrictions imposed by the finite length of the chains. Further, the uni-axial stress response, character of the bond breaking, and non-affine displacements of fully-cured glassy networks were analyzed as a function of the degree of disorder in the network architecture. It is shown that the architecture strongly affects the network stability, flow stress, onset of bond breaking, and ultimate stress while leaving the modulus and yield point nearly unchanged. The results show that internal restrictions imposed by the network architecture alter the chain-level response through changes to the crosslink dynamics in the flow regime and through the degree of coordinated chain failure at the ultimate stress. The properties considered here are shown to be sensitive to even incremental changes to the architecture and, therefore, the overall network architecture, beyond simple defects, is predicted to be a meaningful physical parameter in the mechanics of glassy polymer networks

#### I. Introduction

Polymer networks with strands of low molecular weight are a key category of adhesive and glass-forming composite matrix materials. The covalent connectivity of such networks is formed during a crosslinking reaction, in which short chains are linked to form a network. The assembly of the network is a complex process in which small clusters of chains undergo intra- and inter-cluster reactions to form increasingly large clusters. After a sufficient number of combinations, a macromolecule is formed that spans the entire material. The final network is a disordered, high-dimensional structure. In principle, the final connectivity depends on both the chemical structure of network constituents, such as the chemical functionality of the chains and crosslinkers, as well as factors that alter kinetics of assembling the network such as the diffusivity of the species, relative reactivity of bonds, and presence of a solvent or a catalyst. Indeed these factors, and others, are inputs to statistical and kinetic theoretical models that describe the curing evolution of networks.<sup>1</sup>

The number density of covalent crosslinking sites influences a range of material properties, including the glass transition temperature, elastic modulus, thermal expansion, and swelling behavior. In practice, analysis of the network architecture is usually limited to detecting small-scale defects that lower the number density of crosslink sites that are capable of supporting an external force. Characterizing intra-molecular loop defects is especially useful, since they are chemically stable, and can be easily controlled in experiments through solvent loading. Interestingly, recent work with hydrogels has shown that the overall cyclic structure of elastomer polymer networks can be inferred by counting only the single-chain loop defects, as these are kinetically linked to more complex higher-order loops.<sup>2</sup> Further, recent theoretical work has shown the importance of such cycles, and extended the successful affine and phantom network theories to account for their presence.<sup>3</sup>

In the context of molecular simulations, few attempts have been made to quantify which connectivity characteristics are needed for a realistic physical model and how structures with such characteristics could be created. Instead, a stable cured state of coordinates and connectivity is typically found by either (1) assigning bonds sequentially during dynamic simulations through the use of distance-based bonding rules,<sup>4</sup> which neglect sterics and chemical kinetics, or (2) static<sup>5</sup> or optimization

<sup>&</sup>lt;sup>a</sup> Polymers Branch, U.S. Army Research Laboratory, Aberdeen Proving Ground, Maryland 21005, USA

<sup>&</sup>lt;sup>b</sup> Bennett Aerospace, Inc, Cary, North Carolina 27511, USA

Paper

methods<sup>6,7</sup> that seek a final low-energy state with a predetermined cure and defect density.

To establish an understanding of how differences in the resulting network structure alter physical properties, it is useful to consider a polymer network as a graph object, i.e., a collection of 'nodes' (crosslinks) connected by 'edges' (chains). Solé and Valverde<sup>8</sup> categorized graphs by the qualitative features of heterogeneity (H), randomness (R), and modularity (M), These generic features capture well-known graph structures, such as meshes and trees  $(\downarrow H, \downarrow R, \downarrow M)$ , scale-free  $(\uparrow H, \downarrow R, \downarrow M)$  $\downarrow$ M), and random networks ( $\downarrow$ H,  $\uparrow$ R,  $\downarrow$ M). The polymer curing reaction can be thought of as a series of connected states tracing a path inside this 3-parameter space, where the end of the path is the final cured state. The degree to which the path of the polymer network can be altered is not well-understood, other than the special cases of intentionally introducing short cycles, e.g., through dilution of the reaction mixture or the use of protective groups in a multi-step synthesis. Near the fully cured state of a neat network, each crosslinker has approximately the same number of chemical bonds and, with some exceptions for steric considerations, forms bonds indiscriminately with spatially neighboring species. Therefore, cured polymer networks can intuitively be expected to have a low heterogeneity and low modularity relative to other physical networks. Indeed, this intuition is supported by recent numerical predictions of heterogeneity and modularity from stochastic models.9 The role of randomness is less clear. Although the finite length of the constituent chains allows only spatially close crosslinkers to be linked, a large number of connected states are still possible, and a uniform chain connectivity cannot be expected to emerge in the final cured state. Important exceptions can occur when coordination is enforced, e.g., in metal-organic frameworks and coordination polymers, periodic substrates that confine the network components, 10 or planar sp2 crosslink bonds. 11 Even in cases where well-ordered structures can be created, defects will inevitably introduce at least some degree of randomness and a corresponding change in the mechanical properties.

This work will describe the connected structure of polymer networks with graph metrics, and explore differences in mechanical properties resulting from changes in the ordering of the structure. To aid in this effort, we develop a simple model to apply randomness to the internal network structure and use molecular dynamics simulations to study changes in the mechanical properties. Our previous efforts with relatively small simulations of polymer glasses have shown very little sensitivity of mechanical properties to the way in which the network is assembled. 12 However, variations in the short-range structure strongly affect material properties in other contexts. For example, differences in the structure of non-covalent interactions in the glassy and crystalline states of the same material significantly change the material properties. The atoms or molecules of a glass lack inversion symmetry with their neighbors and, as a consequence, the elastic modulus of the glass is weakened by increased internal work and heating.<sup>13</sup> In the context of polymer networks, Gurtovenko, Gotlib and others studied the

relaxations and viscoelasticity of several topologies, including mesh- and tree-like networks, 14,15 dendrimers, 16 and heterogeneous domains. 17 Stevens 18 studied force-displacement behavior for lattice-like polymer networks that span interfaces, and found differences in stress related to the covalent network structure. More recently, Kryven et al. 9 studied network formation with Monte Carlo simulations, and successfully captured structure and kinetic features from the adjacency matrix of the network. Although the importance of connectivity in polymer networks has been appreciated by these works and others, 19 it is still unclear to what degree linear and non-linear mechanics can be altered.

## II. Graph representations of polymer networks

The number of possible connected states in the network can be expected to rapidly increase with the number of chains and crosslinkers, as is the case for many combinatorial problems. Thus, we expect that a large ensemble of states is available, even in the case of small networks like those in molecular simulations, and that at least some of these states will be stable under our simulation conditions. To characterize the network structure, we represent the polymer network as a graph object composed of nodes (crosslinker molecules) connected by edges (chains), and then describe the mesoscale structure with simple properties derived from the graph. We focus on the final cured state, since this state is chemically stable and particularly relevant for applications.

We define a 'polymer network graph' as  $N_x$  crosslinker molecules (nodes) spanned with  $N_{\rm m}$  linear chains (edges). In this definition, defects cause the number of edges per node to be different than what would be expected from the extent of the reaction. The average degree of crosslinker nodes is found as

$$D = \frac{2}{N_x} \left[ N_{\rm m} - N_{\rm feh} - N_{\rm f} - N_{\rm L} - \sum_{k=1}^{f_x} N_{{\rm e},k}(k-1) \right]$$
 (1)

where  $N_{\rm m}$ ,  $N_{\rm fch}$ ,  $N_{\rm f}$ , and  $N_{\rm L}$  are the number of chains, free chains, free ends, and self loops;  $N_{e,k}$  is the number of crosslinker pairs linked by k chains, where any k > 1 is a redundant link. The graph abstraction is shown in Fig. 1. It is convenient to normalize D as  $\phi_d = D/f_x$  where  $f_x$  is the maximum number of edges per node. We take the extent of reaction as the fraction of reacted sites,

$$\phi_{\rm c} = (2N_{\rm m} - 2N_{\rm fch} - N_{\rm f})/(f_x N_x).$$
 (2)

Combining eqn (1) and (2), the normalized degree  $\phi_d$  and extent of reaction  $\phi_c$  are related as

$$\phi_{c} = \phi_{d} + \frac{N_{f} + 2N_{L} + 2\sum_{k=1}^{f_{x}} N_{e,k}(k-1)}{f_{x}N_{x}}.$$
 (3)

From eqn (3), it is clear that  $\phi_c$  is always greater than  $\phi_d$  due to free end and loop defects which form covalent bonds without

dangling 6 end 2 multiple self-loop links 1

Fig. 1 Abstraction of a polymer network into a graph. Each crosslinker is represented as a node and each chain as an edge, with two exceptions: (1) multiple links are reduced to one edge; and (2) dangling ends and self-loops are omitted.

bridging crosslinker molecules, and due to redundant links created between the same two crosslinkers.

The most well-known graph measure in elastomeric polymers is likely the cycle, due to its use in Paul Flory's landmark theory of rubber elasticity.20 In this context, a cycle is considered as a closed path beginning from an origin node, through non-repeating neighboring nodes, and back to itself. The number of chains in a single cycle was recently shown to be significant when compared with the size of molecular simulations<sup>21</sup> and recent analysis of network heterogeneities suggest that fissures in the network topology, which are related to the cycle size, can act as cavitation sites.<sup>22</sup> Although the number and size of cycles is a useful measure of network structure, identifying cycles is a problem of high algorithmic complexity, thus efficient solutions become difficult or impossible for large networks. As an alternative, we consider the clustering profile<sup>23</sup> as a measure of the local neighborhood structure of a node. The cluster profile of a node describes the length of the shortest paths connecting all pairs of firstneighbor nodes, i.e.,

$$c_d = \frac{|\{p_{ij}\} = d|}{|\{p_{ij}\}|} \tag{4}$$

where  $\{p_{ij}\}$  is the set of shortest path lengths between all pairs of first-neighbor nodes, d is the length of the path measured in chains, and vertical bars find the cardinality (number of elements) of the set. The denominator of eqn (4) normalizes the result by the number of chains in an isolated cluster, i.e.,  $|\{p_{ij}\}| = \frac{1}{2}f_x(f_x - 1)$ . Eqn (4) can be interpreted as a measure of how closely neighbors of a crosslinker node are connected among themselves. (A diagram demonstrating the use of eqn (4) is shown in Fig. 3b). A scalar measure of the short-range clustering can be found by summing over the

cluster profile as

$$c_{\rm s} = \sum_{\rm d \le d} c_{\rm d} \tag{5}$$

where  $d_{\text{max}}$  truncates the sum to consider only small clusters. Another useful measure is the mean shortest path of the network, which physically represents the number of chains bridging each pair of crosslinker beads, averaged over all pairs, *i.e.*,

$$\langle L_{\rm p} \rangle = \frac{1}{N_x(N_x - 1)} \sum_i \sum_j L_{{\rm p},ij}$$
 (6)

where  $L_{p,ij}$  is the smallest number of chains bridging nodes i and j, and  $N_x$  is the number of nodes.

## III. Polymer network structures

We followed two approaches to create the network structures for coarse-grained molecular dynamics (MD) simulations. In both cases, we use the usual freely-jointed bead-spring model for polymer chains with Langevin dynamics described by Kremer and Grest. <sup>24</sup> Each linear chain contains N=6 beads that interact through the truncated-shifted Lennard-Jones potential  $U_{\rm LJ}(r)=4\varepsilon[(\sigma/r)^{12}-(\sigma/r)^6-(\sigma/r_{\rm c})^{12}+(\sigma/r_{\rm c})^6]$ , where  $\varepsilon$  and  $\sigma$  are chosen as unity, and the force cutoff  $r_{\rm c}$  is  $2^{1/6}$  or 2.5. The chain ends form at most one new bond  $(f_{\rm m}=2)$  with a crosslinker bead, and each crosslinker can form a bond with as many as six linear chains  $(f_x=6)$ . All simulations considered  $N_{\rm m}=24\,000$  linear chains and  $N_x=8000$  crosslinkers with a total of 192 000 beads.

#### **Dynamic network models**

Networked structures used in MD simulations are typically grown with reactive, dynamic simulations that rely on local bonding rules. Thus, the network structure is decided implicitly by the polymer dynamics without any need to understand the character of the final connectivity. We apply a typical dynamic strategy for creating crosslink bonds where nearby chain ends that approach within a capture radius<sup>4,12</sup> of a crosslinker bead are reacted. A stoichiometric mixture of  $N_{\rm m}$  chains and  $N_x$ crosslinker beads were distributed in the simulation box; the mixture was equilibrated well above the glass transition temperature  $(T = 1.0\varepsilon/k_B)$ ; crosslink bonds were formed during dynamics when the reactive sites approach within  $1.5\sigma$  until the system reached 98% cure; step-wise annealing was applied across the glass transition, from T = 1.0 to T = 0.3 at a rate of  $10^{-4}$   $\varepsilon/(k_{\rm B}\tau)$ . All dynamic simulations used the LAMMPS package. 25 OVITO, 26 VMD, 27 NetworkX, 28 and graph-tool 29 were

During curing and equilibration, all bonds were represented with the finite extensible nonlinear elastic (FENE) potential,  $U_{\rm FENE}(r) = (-KR_0^2/2) \ln[1 - (r/R_0)^2]$  with  $R_0 = 1.5\sigma$  and  $K = 30\varepsilon\sigma^2$ . After thermal annealing, the unbreakable FENE potential is replaced by a breakable quartic potential,  $U_{\rm q}(r) = k_4(y - b_1)(y - b_2)y^2 + U_0$ , where the parameters  $k_4 = 1434.3$ ,  $b_1 = -0.7589$ ,  $b_2 = 0$ ,  $y = r - \Delta r$  with  $\Delta r = 1.5$ , and  $U_0 = 67.2234$  were chosen to match with the minimum of the FENE potential. <sup>18</sup> Four variations were considered

Paper Soft Matter

Fig. 2 Dynamic curing of polymer networks under several conditions: (a)  $r_c = 2^{1/6}$ , T = 1.0; (b)  $r_c = 2.5$ , T = 1.0; (c)  $r_c = 2^{1/6}$ , T = 5.0, (d)  $r_c = 2^{1/6}$ , T = 1.0where 3 bonds are formed on each crosslinker before the remaining bonds form. The extent of reaction  $\phi_c$  and fraction of elastically active chains  $\phi_d$  are shown as solid and dashed lines, respectively.

in the curing phase to represent weak van der Waals attractions  $(T = 1.0, r_c = 2^{1/6})$ , strong attraction  $(T = 1.0, r_c = 2.5)$ , fast dynamics  $(T = 5.0, r_c = 2^{1/6})$  and two-stage curing where a portion of the reactive groups are initially protected (T = 1.0,  $r_c = 2^{1/6}$ ). In the latter case, all crosslinker beads were required to form three bonds before the remaining bonds were allowed. Fig. 2 shows the progression of the curing reactions over time. We note that in all cases the extent of reaction  $\phi_c$  is substantially greater than the normalized degree  $\phi_d$ due to defects.

The clustering profile of each network was computed according to eqn (4). As shown in Fig. 3a, the results of all four networks were very similar. The clustering profile initially increases with increasing d, reaches a maximum at  $c_4$ , and vanishes at large values of d. These results show that, for all curing conditions, the first neighbors of a crosslinker are often separated by 3 or 4 linear chains, and separations of 6 or more chains do occur albeit less frequently. The similarity of the cluster profiles show that the character of the connectivity cannot easily be altered by changes in thermodynamic conditions, chemistry, or protection of reactive groups in the neat cure considered here. This result is significant, since it indicates the dynamic curing method does not provide a simple route to our goal of exploring alternative connectivities of the network.

#### Watts-Strogatz-like model

We describe a simple model to increment the network connectivity from an ordered to a random state. The Watts-Strogatz (W-S) model<sup>30</sup> was used as a starting point, as it is a typical choice to study the transition from lattice-like to randomly connected graphs. In the W-S model, nodes along a lattice are initially connected with neighboring sites. Each edge of each node is then rewired with a probability p to a randomly selected node elsewhere in the network. Self-loops and repeated edges are not permitted. By measuring the clustering and shortest paths with increasing p, a transition is seen from the initial lattice state, to "small-world" behavior, and finally to a random state described by the Erdős-Rényi model.<sup>31</sup>

We follow an approach inspired by the W-S model, but with three additional features needed for polymer networks. First, the degree of individual nodes must be held constant during

Fig. 3 Clustering profiles of cured polymer networks ( $f_x = 6$ ) (a) assembled from dynamics at temperature T and force cutoff  $r_c$ , where "T-S" indicates a two-stage curing. (b) A restricted Watts-Strogatz model along the lattice-to-random transition. Stable (solid) and unstable (dash) structures are shown for varying  $L_{\rm p}$ -values. "R" is a fully random Erdős-Rényi graph. Inset: Use of eqn (4) for a node (solid black) with four firstneighbor nodes (open black) connected by shortest paths (each colored uniquely).

re-wiring, since the degree represents the chemical functionality of the crosslinker molecule. Second, the range of individual rewiring moves must be limited to reflect the finite length of the linear chains as the edges in our model represent polymer chains with a fixed contour length. Third, because the rewiring moves are local, the same nodes must be rewired several times to arrive at a

random-like network. The initial state of our networks ( $f_x = 6$ ) is taken as a cubic lattice connectivity for compatibility with the cubic shape of the simulation box. The following steps were carried out: (1) a cubic lattice is created such that 6-functional nodes are bridged by edges, (2) an edge is randomly selected, (3) a second edge is selected such that both of its nodes are either one or two edges distant from the nodes of the first edge, (4) the pair of edges are rewired with an "edge swap move", and (5) steps 2-4 are repeated until the network approaches the random state. The move of step (4) exchanges one node of an edge with another node of a nearby edge, thereby altering the connections of four nodes. This move preserves both the degree distribution of the entire network and the degree of individual nodes. Although long-range topological rewiring of nodes is disallowed, local edge swaps can accumulate so that spatially long connections are established, and a random-like network is recovered with a sufficiently large number of swaps. We report the normalized number of edge swaps as  $p = 2N_s/N_m$ , where  $N_s$  is the number of edge swaps. An example series of edge swaps is shown in Fig. 4.

The evolution of this "restricted" W–S model along the ordered-to-random path is shown in Fig. 5 in terms of the normalized shortest paths  $(L_{\rm p}')$  and clustering (C'), where a value of zero indicates the value for a random network. We sampled the structure along this path in 10 steps of approximately 1.0  $\langle L_{\rm p} \rangle$  each, referred to as structures a–j. As observed for the W–S model, 30 the shortest paths and clustering decay with increasing p, with the shortest paths decaying fastest, and the final values approaching that of a random network.

A detailed description of the path taken through the topological parameters is shown in Fig. 6, in terms of  $\langle L_{\rm p} \rangle$  and the first eight components of the clustering profile. The clustering profiles themselves are shown in Fig. 3b. The structure begins as a cubic lattice ( $\langle L_{\rm p} \rangle = 15.0, c_d = 0, 0.8, 0, 0.2, 0, 0, 0, 0$ ) and is incrementally distorted, as described above to create structures a-j. For further comparison, a structure with completely random connections and the four dynamic structures are also shown. First, it is seen that the final value of the model approaches the random result for each of the clustering coefficients. This is a necessary feature of the model, since we aim to develop and test properties of structures along an ordered-to-random transition. We note that, despite the large parameter space, the structures assembled with dynamics are seen to be similar to the restricted W–S model for small cluster

Fig. 5 Normalized clustering (C', circle) and mean shortest path ( $L_{\rm p}'$ , square) during the ordered to random transition of a restricted Watts–Strogatz model, where  $C'=(c_{\rm s}(p)-c_{\rm s,r})/c_{\rm s}(0)$ ,  $L_{\rm p}'=(\langle L_{\rm p}(p)\rangle-\langle L_{\rm p,r}\rangle)/\langle L_{\rm p}(0)\rangle$ ,  $p=2N_{\rm s}/N_{\rm m}$  where  $N_{\rm s}$  is the number of edge swaps, and the subscript r indicates the value for a random structure.  $c_{\rm s}$  was calculated using eqn (5) with  $d_{\rm max}=3$ .

coefficients, as quantified by  $L_{\rm p}-\{c_1,\,c_2,\,c_3\,$  and  $c_4\}$  in the upper panel of Fig. 6. However, the larger clusters, taken as  $L_{\rm p}-\{c_5,\,c_6,\,c_7\,$  and  $c_8\}$  in the lower panel of Fig. 6, are consistently higher for the dynamically created systems. Loops or other defects were not permitted by the restricted W–S model. Thus, this difference is likely created by the presence of defects in the dynamic structures, which effectively remove connections (as shown in eqn (3)) and create more sparsely connected networks.

# IV. Relationship to mechanics

High-strain deformations of polymer networks are common in applications. We used the ten network structures described above (structures a-j) to study the relationship between network topology and mechanics. Following the earlier protocol, all structures were collapsed into a condensed state and thermally

Fig. 4 A restricted Watts-Strogatz-like model. A pair of topologically nearby edges are selected (thick orange), followed by an edge swap move (dashed red). A large number of edge swaps generates a random network. The edge length N is fixed at 6 beads for all moves.

Paper Soft Matter

Fig. 6 Topology of a restricted Watts-Strogatz model (black circles, structures a-j), dynamically cured networks (red triangles), and random networks (red 'X'). The first eight coefficients of the cluster profile are labeled as  $c_1$ - $c_8$ . Inset diagrams illustrate the meaning of each coefficient. The central node (solid black circle), for which the coefficient is calculated, is linked (solid black lines) to nearest neighbors (open black circles). Each shortest path of length d edges between these nearest neighbors (dashed gray lines and open gray circles) contributes to  $c_d$  for the central node according to eqn (4)

annealed until reaching the glassy state (T = 0.3). The structures of the four networks having the lowest values of  $\langle L_{\rm p} \rangle$  were unstable, and could not be annealed with all the covalent bonds intact; for example, structures g and h ( $\langle L_p \rangle = 8.2, 7.4$ ) broke 3.5% (848) and 38% (9121) of the bonds, respectively. Thus, mechanical properties of structures g-j were not considered. The covalent bonds of the remaining structures (a-f) were fully intact and the glassy state thermodynamic properties were essentially identical. (The differences in potential energy and volume of these remaining structures spanned less than 0.02% and 0.1%, respectively). Although only one chain length is studied here, we note that stable networks with longer polymers have the possibility to exhibit more randomness, since the contour length  $L_c$  of the chain limits the maximum Cartesian distance between crosslink sites, and  $L_{\rm c}$  grows faster than the coil size, *i.e.*,  $L_{\rm c} \propto N$  while  $R_{\rm g} \propto \sqrt{N}$ . In this way, long chains can more easily link distant crosslink sites and, in the limiting case, long chains in a small simulation box could approach the Erdős-Rényi (E-R) prediction<sup>31</sup> for purely random networks. For the relatively large simulation sizes we consider, ( $N_x$  = 8000,  $N_{\rm m}$  = 24 000,  $f_x = 6$ ), the E-R prediction of the mean shortest path  $\langle L_{\rm p.r} \rangle = \ln(N_x)/\ln(f_x) = 5.01$  is obtained only at the highest values of p, which are very unstable networks.

Uniaxial tensile loading was applied to network structures a-f at a rate of  $10^{-4} \tau^{-1}$  until reaching a strain of 3.0. The stress results are shown in Fig. 7a along with a zoomed view of the modulus and flow regions in Fig. 7b. The trends in mechanical properties are as follows. The modulus and yield point were found to be essentially identical in all cases. After yield,

decreasing  $\langle L_{\rm p} \rangle$  increased the flow stress and decreased the ultimate stress. No trends were evident in the strain at failure.

It is useful to decompose the virial stress into additive bonded and non-bonded contributions.<sup>32</sup> Fig. 8a demonstrates their relative contribution for one structure. It can be seen from Fig. 8b that the non-bonded contributions are dominant for the modulus region but have a diminishing influence approaching the ultimate stress. Because the non-bonded contributions are similar for all the networks, the modulus and yield stress are also similar. Thus, the differences in network-level constraints act on a lengthscale that is well-separated from the short-range non-bonded interactions, and are not activated at small strains. The bonded contribution is primarily responsible for differences in the features of the stress-strain curves. The bond stress is small at low strain, builds with increasing strain until becoming dominant near the ultimate stress, then releases back to zero after failure. At a given strain in the flow stress region, there is a clear trend of increasing total stress and bond stress with p. We rationalize this behavior in terms of increasingly asymmetric constraints that act on each crosslinker bead as p increases. In the ordered state (p = 0), each crosslinker is surrounded with chains that do not pull taut during the strain until the end of the flow regime, and the crosslinker beads are free to affinely follow the strain field. As p increases, new constraints are introduced into the motion of each crosslinker bead as the shortest paths shrink. Continuous conformational rearrangements are then needed to accommodate the strain field, which should manifest as non-affine motion. To test this, we apply the non-affine squared displacement D2 described by Falk and Langer, 33 where we measure

strain

0.3

0.4

0.5

0.2

0.0

0.1

the displacements of crosslink beads that deviate from the global strain field applied to the cell. The per-crosslink average and maximum values of  $D^2$  are shown in Fig. 9a and b for structures a and f. The average  $D^2$  of both structures sharply increases at the yield strain then decreases to a plateau in the flow regime. The primary differences occur in the flow regime, where we see an increase of  $\sim 10\%$  in the average  $D^2$  for the disordered structure f, and at the strain corresponding to the ultimate stress, where a large spike of non-affinity occurs for the ordered structure a. The trends of the maximum values of  $D^2$  are similar, with the exception of the flow region. For structure f, the maximum  $D^2$  increases with strain throughout the flow regime.

These differences support a picture of contorted crosslink motions beyond the yield point, where crosslink sites must follow a more irregular trajectory in the presence of a disordered network. Pockets of high nonaffinity, as measured by the maximum  $D^2$ , begin to occur at a lower strain and gradually build in intensity for the disordered structure, whereas an abrupt onset at the ultimate stress is seen in the ordered structure. We note that the spike of nonaffinity in the ordered structure occurs at a strain of 1.75. As discussed below, this strain corresponds closely with the highest rate of bond scission.

Fig. 8 Stress-strain response of networks. (a) relative contributions to the stress for structure a, (b) non-bonded contributions for structures a–f, and (c) bonded contributions.

The onset of bond scission can be better understood by comparing the shortest paths through the network with the box length. First, we estimate the maximum strain before bond scission as the strain  $\varepsilon$  needed to pull the end-to-end distance  $R_0$  between crosslinks in the equilibrated structure to the contour length  $L_c$ . Taking  $L_c = nl$ , where the number of bonds n = 7, bond length l = 0.97,  $R_0 = 3.39$ , <sup>34</sup> leads to  $\varepsilon = (L_c - R_0)/R_0 \approx$ 1.36. If the projection of an average end-to-end vector is taken along the strain direction,  $R_0$  is reduced by a factor of  $1/\sqrt{3}$ , 35 resulting in  $\varepsilon \approx 2.60$ . We observe initial bond breaking in the mesh-like structure a at a strain of  $\varepsilon \approx 1.55$ , which corresponds well with the first of the two estimates. We note that bond scission in non-uniform structures, i.e., those other than structure a, can be expected to begin at smaller strains due to both the reduced number of chains that span across the network, and the preexisting uncoiling of the chains along the shortest paths in the network. At zero strain, we estimate that bond scission can be expected spontaneously (i.e., the structure is unstable) if the Paper Soft Matter

Fig. 9 Non-affine displacement of crosslinker beads for two structures (a, f) as a function of strain. (a) Averaged value over all crosslinker beads and (b) maximum value

projection of  $\langle L_p \rangle$  onto the box length is less than half the size of the simulation box  $L_0$ . In this case, the box is longer than the polymer contours along the average shortest path, i.e.,

$$f_{\rm p}nl\langle L_{\rm p}\rangle < \frac{1}{2}L_0,$$
 (7)

where the factors  $f_p$  and  $\frac{1}{2}$  account for the projection of shortest paths onto the box direction and the presence of periodic boundary conditions, respectively. Assuming the shortest paths of interest are completely along the box direction ( $f_p = 1.0$ ) and the box length  $L_0$  = 56.5, we estimate the minimal value of  $\langle L_{\rm p} \rangle$  as 8.07. This estimate agrees well with the  $\langle L_{\rm p} \rangle$  of networks that experience spontaneous bond scission during equilibration, where the largest value was  $\langle L_p \rangle = 8.2$  for structure g. Stevens<sup>18</sup> reported that chains spanning the shortest path between surfaces are the first to become taut, and the strain along these shortest paths was an indicator of the onset of bond scission. We apply this concept to a periodic system by considering the strain of the contour length of  $\langle L_p \rangle$ , taken as  $\varepsilon_{L_p} =$  $\Delta L/(nl\langle L_{\rm p}\rangle)$  where a deformation of  $\Delta L$  is needed to break the first bonds. Solving in terms of the engineering strain of the box,

$$\varepsilon = \frac{\varepsilon_{L_{\rm p}} n l}{L_0} \langle L_{\rm p} \rangle. \tag{8}$$

Fig. 10 Mechanical failure as a function of network architecture. (a) Strain at the onset of bond breaking. (b) Ultimate stress.

From eqn (8), we expect a linear decrease of the strain at the first bond breaking with decreasing  $\langle L_{\rm p} \rangle$ . The relationship of  $\langle L_{\rm p} \rangle$  and the onset on bond scission are shown in Fig. 10. Indeed, reducing the  $\langle L_p \rangle$  reduces the strain at the first bond scission in an approximately linear form. A linear fit to this data arrives at  $\varepsilon_{L_n} nl/L_0 = 0.213$ , which corresponds to  $\varepsilon_{L_n} = 1.71$ . Comparing this result with the system strains shown in Fig. 10, it is seen that only structure a permits the system strain to approach the strain of the shortest paths without bond breaking.

As the strain increases, the motions of crosslinkers become more restricted and many chain scissions are needed to accommodate the increasing length of the simulation box. Fig. 11 shows the number of broken bonds during the deformation. For high p, broken bonds occur at lower strain, and are distributed over a wide range of strain without an abrupt change of bond breaking at any particular strain. In contrast, for low p bonds break at a higher strain and with a high intensity, i.e., many bonds break over a small window of strain. Indeed, the trend of a lowering intensity and broadening of the bond breaking with increasing p (Fig. 11) is consistent with the decline of the ultimate stress and broadening of the stressstrain curve with p (Fig. 7), as well as the broadening of the nonaffine motions near the ultimate stress (Fig. 9). Together, these indicate a loss of coordination in stress loading of the chains with increasing p.

Fig. 11 Broken bonds for network structures a–f. (a) Count of broken bonds and (b) bonds broken per unit strain, with the maximum peak height labeled for structures a–c for clarity.

## V. Conclusions

The connectivity of polymer networks was explored as a mesoscale structural parameter of the material. For this purpose, a simple non-dynamical model to generate polymer networks was developed with the goal of synthetically controlling the network topology. The new model was based in the well-known Watts-Strogatz model, but with restrictions to make the resulting structures compatible with polymer networks. Specifically, the network structures produced by the model allow only topologically close connections to form between crosslinker beads to account for the finite length of linear chains, and respect the chemical functionality of crosslinker beads. Generating structures in this way allowed the chain connectivity to be smoothly varied from ordered to disordered network structures while maintaining an otherwise ideal network without defects and at full cure. Surprisingly, we found that much of the topological character of networks created with independent simulations of dynamical curing could be captured by this restricted Watts-Strogatz model, even for differing chemical interactions, thermodynamic conditions, and synthesis procedures. The networks from the W-S model and dynamic method were shown to have similar 'clustering profiles', where the first four terms of the profile were captured well by the model.

Systematically altering the chain connectivity did not change the density or linear mechanical properties of the glassy networks until a critical number of edge swap moves, after which the network abruptly became unstable. However, the flow stress, the strain at the onset of bond breaking, and the ultimate stress were smoothly changed as the network structure became increasingly disordered. In the flow stress regime, a mechanism was proposed in which the disordered network connections enhanced the non-affine motions of the crosslink junctions, resulting in increased plasticity and higher stress. It is proposed this result is similar to the effect of non-affinity in other contexts where the irregular neighborhood at the atomic scale increases dissipation and reduces elasticity. The onset of chain scission began at high strains in ordered networks, and decreased monotonically as disorder was introduced. This effect was rationalized by demonstrating that the strain at the initial chain scission in the different network structures can be collapsed onto a single value corresponding to  $\sim 1.7$  of the strain of the shortest covalent paths through the network. Further, chain scission in the ordered structures was delayed until almost the same strain as the ultimate stress, thereby forcing many of the extended chains to break simultaneously and strongly increase the ultimate stress.

Polymer network structures must occur between the lattice and randomly-connected states. Critically, these results predict the structure of real networks to fall along a region in which the ultimate stress and other mechanical measures are very sensitive to changes in the connected structure of the chains. Therefore, our results suggest experimental control of network architectures as a potentially powerful design parameter to enhance mechanical performance and, conversely, that polymer networks used in molecular simulations could give spurious results in the absence of realistic connected structures.

### Conflicts of interest

There are no conflicts to declare.

# Acknowledgements

The authors thank Alexandre Abdo (INRA, Paris) and Christopher Rinderspacher (USARL) for useful discussions. RME was supported in part by an appointment to the Postgraduate Research Participation Program at the U.S. Army Research Laboratory (ARL) administered by the Oak Ridge Institute for Science and Education through an interagency agreement between the U.S. Department of Energy and ARL. The research reported in this document was performed in connection with contract W911QX-16-D-0014 with ARL. The views and conclusions contained in this document are those of Bennett Aerospace, Inc. and ARL. Citation of manufacturer's or trade names does not constitute an official endorsement or approval of the use thereof. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation hereon.

Paper Soft Matter

#### References

- 1 K. Dusek and M. Duskova-Smrckova, *Prog. Polym. Sci.*, 2000, **25**, 1215–1260.
- 2 R. Wang, A. Alexander-Katz, J. A. Johnson and B. D. Olsen, *Phys. Rev. Lett.*, 2016, **116**, 188302.
- 3 M. Zhong, R. Wang, K. Kawamoto, B. D. Olsen and J. A. Johnson, *Science*, 2016, 353, 1264–1268.
- 4 E. R. Duering, K. Kremer and G. S. Grest, *J. Chem. Phys.*, 1994, **101**, 8169–8192.
- 5 I. Yarovsky and E. Evans, Polymer, 2002, 43, 963-969.
- 6 S. Kirkpatrick, C. D. Gelatt and M. P. Vecchi, et al., Science, 1983, 220, 671–680.
- 7 P. H. Lin and R. Khare, Macromolecules, 2009, 42, 4319-4327.
- 8 R. Solé and S. Valverde, Lect. Notes Phys., 2004, 650, 189-207.
- 9 I. Kryven, J. Duivenvoorden, J. Hermans and P. D. Iedema, *Macromol. Theory Simul.*, 2016, 25, 449–465.
- 10 O. Ourdjini, R. Pawlak, M. Abel, S. Clair, L. Chen, N. Bergeon, M. Sassi, V. Oison, J.-M. Debierre and R. Coratger, et al., Phys. Rev. B: Condens. Matter Mater. Phys., 2011, 84, 125421.
- 11 J. Ma, D. Alfe, A. Michaelides and E. Wang, *Phys. Rev. B: Condens. Matter Mater. Phys.*, 2009, **80**, 033407.
- 12 C. Jang, T. W. Sirk, J. W. Andzelm and C. F. Abrams, *Macromol. Theory Simul.*, 2015, 24, 260–270.
- 13 A. Zaccone, J. R. Blundell and E. M. Terentjev, *Phys. Rev. B: Condens. Matter Mater. Phys.*, 2011, **84**, 174119.
- 14 A. A. Gurtovenko and Y. Y. Gotlib, *Macromolecules*, 1998, 31, 5756–5770.
- 15 Y. Y. Gotlib, A. A. Gurtovenko, I. A. Torchinskii, V. A. Shevelev and V. P. Toshchevikov, *Macromol. Symp.*, 2003, 131–140.
- 16 A. A. Gurtovenko, D. Markelov, Y. Y. Gotlib and A. Blumen, J. Chem. Phys., 2003, 119, 7579–7590.

- 17 A. A. Gurtovenko, Y. Y. Gotlib and H.-G. Kilian, *Macromol. Theory Simul.*, 2000, **9**, 388–397.
- 18 M. J. Stevens, Macromolecules, 2001, 34, 2710-2718.
- 19 G. Gündüz, M. Dernaika, G. Dikencik, M. Fares and L. Aras, *Mol. Simul.*, 2008, **34**, 541–558.
- 20 P. Flory, Polym. J., 1985, 17, 1-12.
- 21 A. A. Gavrilov, P. V. Komarov and P. G. Khalatur, *Macro-molecules*, 2014, **48**, 206–212.
- 22 M. Zee, A. J. Feickert, D. Kroll and S. Croll, *Prog. Org. Coat.*, 2015, **83**, 55–63.
- 23 A. H. Abdo and A. de Moura, arXiv preprint physics/0605235, 2006.
- 24 K. Kremer and G. S. Grest, J. Chem. Phys., 1990, 92, 5057-5086.
- 25 S. Plimpton, J. Comput. Phys., 1995, 117, 1-19.
- 26 A. Stukowski, Modell. Simul. Mater. Sci. Eng., 2009, 18, 015012.
- 27 W. Humphrey, A. Dalke and K. Schulten, *J. Mol. Graphics*, 1996, 14, 33–38.
- 28 D. A. Schult and P. Swart, Proceedings of the 7th Python in Science Conferences (SciPy 2008), 2008, pp. 11–16.
- 29 T. P. Peixoto, "The graph-tool python library", *figshare*, 2014, DOI: 10.6084/m9.figshare.1164194, https://graph-tool.skewed.de/static/doc/faq.html.
- 30 D. Watts and S. Strogatz, Nature, 1998, 393, 440-442.
- 31 P. Erdős and A. Rényi, Publ. Math., 1959, 6, 290-297.
- 32 T. W. Sirk, S. Moore and E. F. Brown, *J. Chem. Phys.*, 2013, 138, 064505.
- 33 M. Falk and J. Langer, *Phys. Rev. E: Stat. Phys., Plasmas, Fluids, Relat. Interdiscip. Top.*, 1998, 57, 7192.
- 34 Y. R. Sliozberg and J. W. Andzelm, *Chem. Phys. Lett.*, 2012, 523, 139–143.
- 35 J. Rottler and M. O. Robbins, *Phys. Rev. E: Stat. Phys., Plasmas, Fluids, Relat. Interdiscip. Top.*, 2003, **68**, 011801.