## MIT Open Access Articles

# *Molecular mechanism of force induced stabilization of collagen against enzymatic breakdown*

The MIT Faculty has made this article openly available. *[Please](https://libraries.mit.edu/forms/dspace-oa-articles.html) share* how this access benefits you. Your story matters.

**Citation:** Chang, Shu-Wei, Brendan P. Flynn, Jeffrey W. Ruberti, and Markus J. Buehler. "Molecular Mechanism of Force Induced Stabilization of Collagen Against Enzymatic Breakdown." Biomaterials 33, no. 15 (May 2012): 3852–3859.

**As Published:** http://dx.doi.org/10.1016/j.biomaterials.2012.02.001

**Publisher:** Elsevier

**Persistent URL:** <http://hdl.handle.net/1721.1/101438>

**Version:** Author's final manuscript: final author's manuscript post peer review, without

publisher's formatting or copy editing

**Terms of use:** Creative Commons [Attribution-NonCommercial-NoDerivs](http://creativecommons.org/licenses/by-nc-nd/4.0/) License

*Biomaterials*. Author manuscript; available in PMC 2013 May 1.

Published in final edited form as:

Biomaterials. 2012 May ; 33(15): 3852–3859. doi:10.1016/j.biomaterials.2012.02.001.

## **Molecular mechanism of force induced stabilization of collagen against enzymatic breakdown**

**Shu-Wei Chang**1, **Brendan P. Flynn**2, **Jeffrey W. Ruberti**2, and **Markus J. Buehler**1,3,4,†

- 1 Laboratory for Atomistic and Molecular Mechanics, Department of Civil and Environmental Engineering, Massachusetts Institute of Technology, 77 Massachusetts Ave. Room 1-235A&B, Cambridge, MA, USA
- 2 Department of Mechanical and Industrial Engineering, Northeastern University,360 Huntington Avenue, Boston, MA, USA
- 3 Center for Materials Science and Engineering, Massachusetts Institute of Technology, 77 Massachusetts Ave., Cambridge, MA, USA
- 4 Center for Computational Engineering, Massachusetts Institute of Technology, 77 Massachusetts Ave., Cambridge, MA, USA

## **Abstract**

Collagen cleavage, facilitated by collagenases of the matrix metalloproteinase (MMP) family, is crucial for many physiological and pathological processes such as wound healing, tissue remodeling, cancer invasion and organ morphogenesis. Earlier work has shown that mechanical force alters the cleavage rate of collagen. However, experimental results yielded conflicting data on whether applying force accelerates or slows down the degradation rate. Here we explain these discrepancies and propose a molecular mechanism by which mechanical force might change the rate of collagen cleavage. We find that a type I collagen heterotrimer is unfolded in its equilibrium state and loses its triple helical structure at the cleavage site without applied force, possibly enhancing enzymatic breakdown as each chain is exposed and can directly undergo hydrolysis. Under application of force, the naturally unfolded region refolds into a triple helical structure, potentially protecting the molecule against enzymatic breakdown. In contrast, a type I collagen homotrimer retains a triple helical structure even without applied force, making it more resistant to enzyme cleavage. In the case of the homotrimer, the application of force may directly lead to molecular unwinding, resulting in a destabilization of the molecule under increased mechanical loading. Our study explains how force may regulate the formation and breakdown of collagenous tissue.

#### **Keywords**

|           | Collagen; collagenolysis; collagen degradation; molecular mechanics; protein structure; local |  |  |  |  |
|-----------|-----------------------------------------------------------------------------------------------|--|--|--|--|
| unfolding |                                                                                               |  |  |  |  |

**Publisher's Disclaimer:** This is a PDF file of an unedited manuscript that has been accepted for publication. As a service to our customers we are providing this early version of the manuscript. The manuscript will undergo copyediting, typesetting, and review of the resulting proof before it is published in its final citable form. Please note that during the production process errors may be discovered which could affect the content, and all legal disclaimers that apply to the journal pertain.

**Supporting Information Available:** Detailed information on materials and methods used, including kinetic models and structural analysis of simulation results are available.

<sup>© 2012</sup> Elsevier Ltd. All rights reserved.

<sup>†</sup> *Corresponding author, electronic address:* mbuehler@MIT.EDU, *Phone: +1-617-452-23750, Fax: +1-617-324-4014*.

## **Introduction**

Collagen is a triple helical molecule and the most abundant protein in vertebrates, provides mechanical stability, elasticity and strength to organisms [1, 2]. Collagen cleavage is crucial for many biological and pathological processes such as tissue modeling and remodeling, growth, wound healing, cancer invasion and organ morphogenesis [3-5]. Normal physiological remodeling processes involve precisely regulated collagen degradation, where excessive or deficient degradation has been associated with numerous diseases. Accelerated breakdown of collagen has been associated with arthritis, atherosclerotic heart disease, tumor cell invasion, glomerulonephritis, and cell metastasis [6-9]. On the other hand, deficient degradation of collagen has been shown to result in spontaneous abnormal growth plate and increased trabecular bone in mice [10]. Therefore, understanding the mechanism of collagenolysis and how it is possible to modulate its activity is crucial for developing treatments for a variety of diseases.

Collagenases of the matrix metalloproteinase (MMP) family [11] are mammalian proteases involved in the physiological cleavage of collagen. The most prevalent collagenases, including MMP-1, MMP-8 and MMP-13, consist of propeptide, catalytic and hemopexin domains. They play an important role in cleaving collagen in the extracellular matrix, resulting in characteristic ¾- and ¼-fragments. The specific cleavage site is after the 775th residue (a Gly amino acid), in the sequences of G-IA for alpha-1 chain and G-LL for alpha-2 chain . Interestingly, there are several other sites in the collagens that contain the same G-I/L bonds but they are not hydrolyzed . Since the amino acid sequence alone is not sufficient to explain the high specificity of collagen recognition by MMPs [12], the local conformation at the vicinity of the cleavage site might play an important role in providing a recognition signal for MMPs.

Normal type I collagen is a heterotrimer and consists of two alpha-1 chains and one alpha-2 chain. A variation of the natural type I collagen molecule is the type I homotrimer, which consists of three alpha-1 chains, and has been found in fetal tissues [13], fibrotic tissues [14], carcinomas [15], and fetal and cancer cells [15, 16] in human. It is also found in a mouse model of the genetic brittle bone disease, *osteogenesis imperfecta* (OI), the *oim* mutation of type I collagen. Experimental studies have shown that the mechanical strength of *oim* bone and tail tendon is significantly less than that of the normal mice [17, 18].

Previous studies have shown that the type I heterotrimer and homotrimer have distinct degradation behaviors (**Figure 1**). Type I homotrimers are found to be resistant to all mammalian collagenases [15, 19, 20], with a cleavage rate much slower for homotrimers than for heterotrimers (**Figure 1(a)**). The MMP resistance of homotrimers may play an important role in homotrimer-related diseases or in early development, during which necessary collagen degradation may be hindered with detrimental results. For example, it has been shown in fibers reconstituted from mouse tail tendon collagen that a minor fraction of homotrimer-based fibers may grow instead of being disassembled during tissue remodeling cycles, which may eventually result in tissue disorganization [20].

Moreover, it has been shown experimentally that mechanical force applied to collagen molecules alters the cleavage rate [21-23]. Notably, conflicting results between homo- and heterotrimer collagen have been reported. Experiments using a single, collagen homotrimer snippet (14 kDa) have shown that mechanical load induces an 81±3-fold increase in the rate of collagen proteolysis (**Figure 1(b)**) [21]. On the contrary, it has been found that mechanical load stabilizes heterotrimer (arranged in reconstituted fibrils) against enzymatic degradation (**Figure 1(c)**) [22]. The lower specificities in **Figures 1(a-b)** can be at least partially explained by the lower reaction temperatures.

Here we provide a possible mechanistic explanation for these discrepancies through a detailed molecular analysis of the mechanics of collagen molecules using a combination of molecular dynamics simulation and mechanochemical experiments.

## **Materials and methods**

#### **Collagen molecule generation**

We use full atomistic simulations to study the structure in the vicinity of the cleavage site of the type I mouse heterotrimer and homotrimer. The real sequences of type I alpha-1 and type I alpha-2 chains of *mus musculus* (wild type mouse) are used to generate the collagen molecules. The heterotrimer collagen molecule is built of two alpha-1 chains and one alpha-2 chain while the homotrimer collagen molecule is built of three alpha-1 chains. The sequences are adopted from NCBI protein database [\(http://www.ncbi.nlm.nih.gov/protein\)](http://www.ncbi.nlm.nih.gov/protein): AAH50014.1 for the alpha-1 chain and NP\_031769.2 for the alpha-2 chain. The entire alpha-1 and alpha-2 chains consist of 1014 residues with repeated G-X-Y triplets, excluding the C-terminal and N-terminal sequences. Segments of real sequences with 63 residues long centered at the MMP-1 cleavage site (748th to 810th residues) of alpha-1 and alpha-2 chains are chosen to construct the heterotrimer and homotrimer collagen molecules. The chosen sequences are

#### **alpha-1**:

GPPGPAGEKGSPGADGPAGSPGTPGPQGIAGQRGVVGLPGQRGERGFPGLPGPS GEPGKQGPS and

#### **alpha-2**:

GPPGFVGEKGPSGEPGTAGAPGTAGPQGLLGAPGILGLPGSRGERGLPGIAGAL GEPGPLGIS.

Note that "-" indicates the scissile bond (the bond cut by MMP) at the cleavage site (G-IA for alpha-1 chain and G-LL for alpha-2 chain) of MMP-1.

The collagen molecules are created by inputting the sequences of three chains into the software THeBuScr (an interactive triple-helical collagen building script) [24], which enables us to build triple-helical molecules based on any specified amino acid sequence using conformations derived from statistical analyses of high-resolution X-ray crystal structures of triple-helical peptides. To neutralize the terminals, two ends of the heterotrimer and homotrimer are capped by assigning the first residues to ACE, the acetylated Nterminus, and the last residues to CT3, the N-Methylamide C-terminus. The length of the collagen molecule is about 170 Å and is solvated in a periodic water box composed of TIP3 water molecules. Visual Molecular Dynamics (VMD) [25] is used to solvate the molecule and to neutralize the system by adding ions. The final solvated all-atom system contains ≈70,000 atoms, with a periodic box size of 230 Å × 60 Å × 60 Å.

#### **All-atom modeling and equilibration**

Full atomistic simulations are performed using NAMD [26] and the CHARMM force field [27] that includes parameters for hydroxyproline amino acids [28]. This force field has been broadly validated for a variety of biochemical models of proteins including collagen [29-31]. An energy minimization using a conjugate gradient scheme is performed before molecular dynamics simulations. Rigid bonds are used to constrain hydrogen atoms, thus allowing an integration time step of 2 fs. Nonbonding interactions are computed using a cut-off for neighbor list at 13.5 Å, with a switching function between 10 and 12 Å for van der Waals interactions. The electrostatic interactions are modeled by the particle mesh Ewald summation (PME) method. After energy minimization, the collagen molecule is fixed and the system is simulated with *NPT* at a constant temperature of 310 K and 1.013 bar pressure

to reach an equilibrium box size with water molecules. Thermal fluctuations are thought to play an important role in the cleavage site of collagen molecule [32]. To ensure a sufficient exploration on the configurations of collagen molecules, after reaching an equilibrium box size, both heterotrimer and homotrimer are equilibrated with a *NVT* ensemble at 310 K for 80 ns. During the simulations, the configurations of collagen molecule are recorded every 20 ps, resulting in 4,000 frames (all used in the analyses). The total simulation takes about 12,800 CPU hours for each collagen molecule.

#### **In silico mechanical testing**

In order to examine the effects of applied force on the structural changes of heterotrimer and homotrimer collagen and to explore how force affects the MMP cleavage, we use steered molecular dynamics to apply a constant 100 pN force to each chain of heterotrimer and homotrimer after 80 ns equilibrium. The Cα atom of the 13th residue (the 5th Gly) of each chain is fixed and a constant force is applied at the Cα atom of the 49th residue (the 17th Gly) of each chain. With the constant force applied, the simulations of heterotrimer and homotrimer are performed for 25 ns. During the simulation the configurations of collagen molecule are recorded every 20 ps.

#### **Analysis of molecular structures**

We focus the structural analyses on the cleavage site, *i.e.* the 18 residues at the center of the collagen molecules used in our simulations. The unit heights and radii are calculated every 0.02 ns and an averaged of the data from 60 ns to 80 ns are taken to study the structural behavior of heterotrimer and homotrimer without applying force. For the analyses of the structural behavior of heterotrimer and homotrimer with applying force, an average of the data from 15 ns to 25 ns with force is used. We use the same method as in our previous work [33] to calculate the unit heights and radii. The length of the cleavage site under applying force is calculated by summing the unit heights from the 5th height to 16th height.

#### **Analysis of H-bonds at the cleavage site**

Averaged N-O distances for heterotrimer and homotrimer with and without force at the cleavage site are calculated to study the H-bond forming. We calculate the distance from N of Gly residue to O of the residue at the X position of (GXY)n triplets. The bond distances with and without force are calculated by averaging from 60 ns to 80 ns for *NVT* simulation and from 15 ns to 25 ns for steered molecular dynamics simulation respectively. There are three H-bonds can be formed at the Gly at the scissible bonds: from Gly at chain A to X position at chain B, from Gly at chain B to X position at chain C and from Gly at chain C to X position at chain A. If the N-O distance is smaller than 0.4 nm, it is assumed that there exists an H-bond. The life time of an H-bond is calculated by the ratio of the time an H-bond exists to the total time of observation. We calculate the life time of H-bond with and without force by observing from 60 ns to 80 ns for *NVT* simulation and from 15 ns to 25 ns for steered molecular dynamics simulation respectively.

#### **Experimental mechanochemical testing of native collagen fibrils**

Experiments are performed for comparison with theoretical findings. Individual, native type I collagen fibrils were isolated from bovine sclera and attached to force-calibrated glass micro-needles as previously described [34]. Briefly, scleral tissue was washed in 2M NaCl, mechanically separated using a blender (Chefmate), then dialyzed to remove NaCl. Individual fibrils were removed from solution using glass micro-needles controlled by micromanipulators, then affixed to force-calibrated micro-needle tips using epoxy (DP100, 3M). Each suspended fibril was washed in 2M NaCl, then 1X Tris Buffer (0.05M Tris, 0.2M NaCl, 5mM CaCl2). Mechanical testing was performed on each fibril prior to enzymatic

testing to establish a baseline for mechanical properties. Fibrils were adjusted to zero-load (slack), low-load (0.7pN/monomer), or high-load (70pN/monomer) then exposed to activated MMP-8, 0.665μM in tris-buffer, and maintained at 37°C for four hours (*n*=1 for each load). Additional mechanical testing was performed at intervals to determine the enzyme-induced loss in mechanical integrity, and the stiffness-degradation rate was converted to a radial degradation rate. Molecular cleavage rates were calculated by using axial symmetry to simplify fibril degradation to 1-D erosion and dividing radial rates by the intermolecular spacing of collagen in fibrils, 1.6 nm [35]. The ratio *k*cat/*K*m is calculated from radial degradation rates using the solution to the reaction-diffusion equations governing insoluble collagen degradation in cases of low enzyme and substrate [36] (see **Supporting Material**).

## **Results and discussion**

We find that the heterotrimer is thermally unfolded locally in the vicinity of the cleavage site at body temperature. In contrast, the homotrimer is thermally more stable and retains the triple helical structure characteristic of collagen molecules, suggesting stark differences in the structure of hetero-versus homotrimer molecules (**Figure 2**). Further quantitative geometric analysis supports this notion. Specifically, the unit heights and radii in the vicinity of the cleavage sites of the heterotrimer and homotrimer are shown in **Figure 3**. Comparing to a statistical analysis of high-resolution X-ray crystal structures of triple-helical peptides [37] (a study which reports that the unit height of the triple helical structure of collagen molecule is around 9 Å and the radius is about 2 Å), we find that the structure of the homotrimer in the vicinity of cleavage site is a rather stable triple-helical structure (**Figure 2**). However, decreased unit heights and increased radii are found in the vicinity of the cleavage site of the heterotrimer (**Figure 3**), supporting the visual analysis of the two structures. Notably, a large radius, each chain is separated and exposed ( 5.7 ± 0.4 Å, is found at the scissile bond of the heterotrimer, indicating that **Figure 2**) and may play a role in the recognition of MMPs.

Next we explore the response of the molecular structures to the application of mechanical force. We find that the heterotrimer and homotrimer also show very different mechanical behaviors, as revealed in **Figure 4**. A detailed analysis of the strain distribution along the twisting axis of the homotrimer reveals that it features a uniform strain distribution with a total strain close to 0.03 (**Figure 4(c)**). By assuming the diameter of collagen molecule is 15 Å, we obtain a 5.7 GPa Young's modulus, which is within reasonable bounds established in earlier studies [38, 39] and suggests that the strain is entirely elastic strain. In contrast, for the heterotrimer, elastic strains similar to the homotrimer are only found at the regions outside of the vicinity of the cleavage site. Interestingly, large strains beyond the elastic strains are found *highly localized* in the vicinity of the cleavage site. Indeed, the maximum local strain is about 0.7 found at the scissile bond, which results in a small, 0.24 GPa, local Young's modulus indicating the applying force not only induces elastic strain but also stretches the collagen by reducing the entropy of the unfolded region.

The time history of the length of the cleavage site (**Figure 4(b)**) provides further evidence that there are two regimes of stretching for the heterotrimer. This confirms that the applied force contributes to both entropic and energetic strain. In the first regime, the applied force reduces the entropy of the unfolded region and stabilizes the cleavage site by refolding it into the triple helical structure, which explains why a rather large local strain is observed in the vicinity of the cleavage site. In the second regime, the deformation of collagen molecules is in the energetic region, where the applied force induces elastic strain. Remarkably, after just 15 ns of application of force, the heterotrimer and the homotrimer feature the same lengths. This indicates that the applying the force pulls the heterotrimer out

of the entropic elasticity region (**Figure 4(a)**). Structural analyses show increases in the unit heights and decreases in the radii in the vicinity of the cleavage site (**Figure S2** in **Supporting Material**), which provides additional direct evidence that applied force stabilizes (*i.e.*, refolds) the cleavage site. This is consistent with experimental studies which show that applied force enhances thermal stability of collagen [40]. It is worth mentioning that although a large force (~100 pN) is applied to speed up the force induced stabilization in our atomistic simulations, we anticipate a low force (~10 pN) within the entropic elastic region would result in the same stabilization mechanism in the heterotrimer since the conformational changes at the cleavage site of the heterotrimer are primarily due to entropic contributions (**Figure 4(c)**).

An analysis of averaged N-O distances and the H-bond life time in the vicinity of the cleavage site also shows that the heterotrimer thermally unfolded without applying force and that the thermal stability is increased by applying force at the cleavage site (**Figure 5**). Without applying force, all H-bonds close to the scissile bond are stably formed in the homotrimer, but none of them are formed in the heterotrimer (**Figure 5**). When force is applied, no significant change is observed in the homotrimer, *i.e*. the applied force does not alter the life time of stably formed H-bonds. Interestingly, a recovery of H-bonds close to the scissile bond is found in the heterotrimer, providing direct evidence for the stabilizing effect of mechanical force. The simulations show that H-bonds are broken 87% (no force) versus 17% (with force) of the time (**Figure 5**). Thus, we conclude that the applying force stabilizes the cleavage site of the heterotrimer. Most importantly, the same effect is not found for the homotrimer since it is thermally stable without applying force.

Experimental mechanochemical testing results further support the conclusion that force stabilizes the heterotrimer. MMP-8 degradation rates were extremely low and no fibrils failed completely during degradation experiments. However, evaluation of the reduction in fibril stiffness reveals that enzymatic degradation did occur, and that the degradation rate was modulated by applied tension. The calculated molecular cleavage rates (*k*cat) were 0.0014, 0.0002, and 0.0000 s-1 for applied loads 0 (*n*=3), 0.7 (*n*=1), and 70 (*n*=1) pN/ molecule, respectively. These values for *k*cat are three orders of magnitude lower than published values for MMP-8 acting on soluble a-telocollagen [41, 42], indicating that the presence of the native non-helical terminal peptides and molecular packing into fibrils drastically reduce the enzymatic susceptibility of type I collagen. Calculated *k*cat/*K*m values are 4.17, 0.52, and 0.00min-1μM-1 for zero, low and high loads, respectively. The value of 4.17min-1μM-1 is in good agreement with published values for *k*cat/*K*m , indicating that while molecular cleavage rate is reduced for fibrillar telo-collagen, the enzyme specificity remains the same. Though *n* = 1 for non-zero loads, the results follow the same pattern found previously using bacterial collagenase (BC)[34], with a small applied load (0.7pN/ molecule), corresponding to less than 0.1% strain, drastically reducing the molecular cleavage rate. This finding supports the theory that locally unstable loops must form for collagen cleavage. The low forces required to stabilize collagen molecules indicate the mechanism is an entropic return to triple-helical conformation. Also in agreement with the previous BC investigation, high tensile load significantly reduces molecular cleavage rates.

It is widely accepted that collagen cleavage by MMPs involves three steps [20, 43, 44]: enzyme binding, helix unwinding at the cleavage site and sequential hydrolysis of the chains. However, two different models have been proposed to explain the cleavage mechanism. In the first model, due to the thermal instability of the cleavage site, collagen is thermally unwound locally at the vicinity of the cleavage site before MMPs bind [43]. In the second model, collagen is unwound after MMPs bind to the cleavage site [44]. Without a *priori* assumption about the effect of MMPs on the local triple helix unwinding, these two models have been integrated into a more general mechanism [20] as shown in **Figure S3**. By

setting *k*2 = *k*–2 = 0, the degradation scheme reduces to the first model and by setting *k*3 = *k*–3 = *k*4 = *k*–4 = 0, the scheme reduces to the second model.

An analysis of the cleavage kinetics shown in **Figure S3** under steady state assumption shows that the thermal stability, *K*4 = *k*4/*k*–4, plays a role in the MMP resistance of the homotrimer (see **Supporting Material**). When *K*4 << 1, the cleavage rate is small and is not sensitive to the thermal stability of the collagen molecule. When thermal stability is low (*K*<sup>4</sup> >> 1), the collagen is thermally unfolded at a fast rate and the cleavage rate increases and becomes sensitive to the thermal stability. Therefore, the MMP resistance of the homotrimer results from its high thermal stability, which is in good accordance with the experimental results [20].

From the results of mechanical simulations and mechanochemical tests, we see that applied force stabilizes the cleavage site of the heterotrimer, indicating that *K*4 (heterotrimer, force) >> *K*4 (heterotrimer, no force). Because the cleavage rate of the heterotrimer depends on the thermal stability and the increase of thermal stability slows down the cleavage rate (**Supporting Material**), we postulate that the applying force slows down the rate of thermally unfolding and should also slow down the cleavage rate of the heterotrimer. This suggests a molecular mechanism in which force induces stabilization of collagen and against enzymatic breakdown.

The applied force is less likely to slow down the cleavage rate by the same mechanism for the homotrimer because it is thermally stable without applying force. However, the remaining question is why the force speeds up the cleavage rate of the homotrimer [21]. One possible explanation is that in the homotrimer, force results in homogeneously distributed elastic strain that eventually leads to unfolding of the molecule, suggesting that force leads to destabilization of the molecular structure. This explanation would not affect the fact that applying force slows down the cleavage of the heterotrimer since it primarily follows path I and the degradation rate is mainly governed by the thermal stability.

## **Conclusion**

Molecular modelling showed that while the vicinity of the cleavage site of the heterotrimer is thermally unfolded at body temperature, the vicinity of the cleavage site of the homotrimer remains triple helical, as it is thermally stable. We conclude that the higher thermal stability of the homotrimer is also responsible for its greater resistance to MMP degradation in free solution. Moreover, we find that in heterotrimers, the application of force alters the structure of the cleavage domain such that the molecule exhibits enhanced stability and behaves in a manner similar to the homotrimer. Thus, force appears to "switch" the heterotrimer from unstable to stable by analogy. Direct quantitative evidence was found to support this finding, as established by the observation that the molecular radii decreases and that the H-bond lifetime increases, by forming a triple helical geometry at the cleavage site. The reformation of a triple helical structure should protect against enzymatic breakdown if unwinding is necessary for alpha chains to enter the cleavage site on MMPs [3.,11]. If, however, thermal fluctuations at the cleavage site make catalysis more difficult, then the applied force with its associated triple helix reformation could accelerate the cleavage rate. Such behavior may have been observed in the work of Adhikari *et al*. [21], but the nearly 100 fold acceleration in MMP-1 cleavage is difficult to fully explain. In the homotrimer force results in homogeneously distributed elastic strain that likely leads to frank unfolding of the molecule, suggesting that force leads to destabilization of the entire molecular structure.

Complementary experimental mechanochemical tests at the single native fibril level support the finding that applied tension stabilizes the collagen triple helix against enzymatic degradation, and the extreme sensitivity of the mechanism supports the theory that heterotrimer collagen cleavage requires spontaneously formed non-triple-helical loops at the cleavage site. Our study suggests a molecular mechanism by which thermal stability plays a role in the cleavage mechanism and the potential explanation of the force-stabilization results found in experimental studies as well as the distinct behaviours of homo- and heterotrimer collagen. In addition to providing a possible explaination for the seemingly conflicting experimental results reported in earlier papers based on a mechanistic model, we anticipate that our study is crucial for developing new biomaterials that serve as platforms for treatments for a variety of diseases. For example, through tuning the mechanical force, one can precisely regulate collagen degradation. This concept can also find application in tunable collagen-based biomaterials, and may be important to understand the development of tissue.

## **Supplementary Material**

Refer to Web version on PubMed Central for supplementary material.

## **Acknowledgments**

We acknowledge support from NSF-CAREER (CMMI-0642545), a NSF-IGERT Nanomedicine Award (DGE-0504331), and NIH NEIEY0155500.

## **References**

- 1. Fratzl, P. Collagen: structure and mechanics. Springer; 2008.
- 2. Buehler MJ, Yung YC. Deformation and failure of protein materials in physiologically extreme conditions and disease. Nat Mater. 2009; 8:175–88. [PubMed: 19229265]
- 3. Nagase H, Visse R. Matrix metalloproteinases and tissue inhibitors of metalloproteinases Structure, function, and biochemistry. Circ Res. 2003; 92:827–39. [PubMed: 12730128]
- 4. Baragi VM, Qiu L, GunjaSmith Z, Woessner JF, Lesch CA, Guglietta A. Role of metalloproteinases in the development and healing of acetic acid-induced gastric ulcer in rats. Scand J Gastroenterol. 1997; 32:419–26. [PubMed: 9175201]
- 5. Helary C, Ovtracht L, Coulomb B, Godeau G, Giraud-Guille MM. Dense fibrillar collagen matrices: A model to study myofibroblast behaviour during wound healing. Biomaterials. 2006; 27:4443–52. [PubMed: 16678257]
- 6. Barnes MJ, Farndale RW. Collagens and atherosclerosis. Exp Gerontol. 1999; 34:513–25. [PubMed: 10817807]
- 7. Bode MK, Mosorin M, Satta J, Risteli L, Juvonen T, Risteli J. Complete processing of type III collagen in atherosclerotic plaques. Arterioscler Thromb Vasc Biol. 1999; 19:1506–11. [PubMed: 10364082]
- 8. McDonnell S, Morgan M, Lynch C. Role of matrix metalloproteinases in normal and disease processes. Biochem Soc Trans. 1999; 27:734–40. [PubMed: 10917674]
- 9. Riley GP, Harrall RL, Watson PG, Cawston TE, Hazleman BL. Collagenase (MMP-1) and TIMP-1 in destructive corneal disease associated with rheumatoid arthritis. Eye. 1995; 9:703–18. [PubMed: 8849537]
- 10. Stickens D, Behonick DJ, Ortega N, Heyer B, Hartenstein B, Yu Y, et al. Altered endochondral bone development in matrix metalloproteinase 13-deficient mice. Development. 2004; 131:5883– 95. [PubMed: 15539485]
- 11. Nagase H, Visse R, Murphy G. Structure and function of matrix metalloproteinases and TIMPs. Cardiovasc Res. 2006; 69:562–73. [PubMed: 16405877]

12. Xiao J, Addabbo RM, Lauer JL, Fields GB, Baum J. Local conformation and dynamics of isoleucine in the collagenase cleavage site provide a recognition signal for matrix metalloproteinases. J Biol Chem. 2010; 285:34181–90. [PubMed: 20679339]

- 13. Jimenez SA, Bashey RI, Benditt M, Yankowski R. Identification of collagen alpha1(I) trimer in embryonic chick tendons and calvaria. Biochem Biophys Res Commun. 1977; 78:1354–61. [PubMed: 562664]
- 14. Ehrlich HP, Brown H, White BS. Evidence for type V and I trimer collagens in Dupuytren's Contracture palmar fascia. Biochem Med. 1982; 28:273–84. [PubMed: 6819866]
- 15. Makareeva E, Han S, Vera JC, Sackett DL, Holmbeck K, Phillips CL, et al. Carcinomas contain a matrix metalloproteinase-resistant isoform of type I collagen exerting selective support to invasion. Cancer Res. 2010; 70:4366–74. [PubMed: 20460529]
- 16. Minafra S, Luparello C, Rallo F, Pucci-Minafra I. Collagen biosynthesis by a breast carcinoma cell strain and biopsy fragments of the primary tumour. Cell Biol Int Rep. 1988; 12:895–905. [PubMed: 3224373]
- 17. Misof K, Landis WJ, Klaushofer K, Fratzl P. Collagen from the osteogenesis imperfecta mouse model (oim) shows reduced resistance against tensile stress. J Clin Invest. 1997; 100:40–5. [PubMed: 9202055]
- 18. McBride DJ, Choe V, Shapiro JR, Brodsky B. Altered collagen structure in mouse tail tendon lacking the alpha2(I) chain. J Mol Biol. 1997; 270:275–84. [PubMed: 9236128]
- 19. Narayanan AS, Meyers DF, Page RC, Welgus HG. Action of mammalian collagenases on type I trimer collagen. Coll Relat Res. 1984; 4:289–96. [PubMed: 6090054]
- 20. Han S, Makareeva E, Kuznetsova NV, DeRidder AM, Sutter MB, Losert W, et al. Molecular mechanism of type I collagen homotrimer resistance to mammalian collagenases. J Biol Chem. 2010; 285:22276–81. [PubMed: 20463013]
- 21. Adhikari AS, Chai J, Dunn AR. Mechanical load induces a 100-fold increase in the rate of collagen proteolysis by MMP-1. J Am Chem Soc. 2011; 133:1686–9.
- 22. Flynn BP, Bhole AP, Saeidi N, Liles M, Dimarzio CA, Ruberti JW. Mechanical strain stabilizes reconstituted collagen fibrils against enzymatic degradation by mammalian collagenase matrix metalloproteinase 8 (MMP-8). PLoS One. 2010; 5:e12337. [PubMed: 20808784]
- 23. Ellsmere JC, Khanna RA, Lee JM. Mechanical loading of bovine pericardium accelerates enzymatic degradation. Biomaterials. 1999; 20:1143–50. [PubMed: 10382830]
- 24. Rainey J, Goh M. An interactive triple-helical collagen builder. Bioinformatics. 2004; 20:2458–9. [PubMed: 15073022]
- 25. Humphrey W, Dalke A, Schulten K. VMD: Visual molecular dynamics. J Mol Graph. 1996; 14:33. [PubMed: 8744570]
- 26. Nelson MT, Humphrey W, Gursoy A, Dalke A, Kale LV, Skeel RD, et al. NAMD: A parallel, object oriented molecular dynamics program. Int J High Perform Comput Appl. 1996; 10:251–68.
- 27. MacKerell AD, Bashford D, Bellott M, Dunbrack RL, Evanseck JD, Field MJ, et al. All-atom empirical potential for molecular modeling and dynamics studies of proteins. J Phys Chem B. 1998; 102:3586–616.
- 28. Anderson, D. Collagen self-assembly: A complementary experimental and theoretical perspective. University of Toronto; Toronto, Canada: 2005.
- 29. Gautieri A, Buehler MJ, Redaelli A. Deformation rate controls elasticity and unfolding pathway of single tropocollagen molecules. J Mech Behav Biomed Mater. 2009; 2:130–7. [PubMed: 19627816]
- 30. Gautieri A, Uzel S, Vesentini S, Redaelli A, Buehler MJ. Molecular and mesoscale mechanisms of osteogenesis imperfecta disease in collagen fibrils. Biophys J. 2009; 97:857–65. [PubMed: 19651044]
- 31. Srinivasan M, Uzel SGM, Gautieri A, Keten S, Buehler MJ. Alport syndrome mutations in type IV tropocollagen alter molecular structure and nanomechanical properties. J Struct Biol. 2009; 168:503–10. [PubMed: 19729067]
- 32. Stultz CM, Nerenberg PS, Salsas-Escat R. Do collagenases unwind triple-helical collagen before peptide bond hydrolysis? Reinterpreting experimental observations with mathematical models. Proteins. 2008; 70:1154–61. [PubMed: 17932911]

33. Chang SW, Shefelbine SJ, Buehler MJ. Structural and mechanical differences between collagen homo- and heterotrimers: Relevance for the molecular origin of brittle bone disease. Biophys J. 2012; 102

- 34. Flynn BP, Tilburey GE, Ruberti JW. Highly sensitive single fibril erosion assay demonstrates mechanochemical switch in native collagen fibrils. Biomech Model Mechanobiol. under submission.
- 35. Meek KM, Leonard DW. Ultrastructure of the corneal stroma: a comparative study. Biophys J. 1993; 64:273–80. [PubMed: 8431547]
- 36. Tzafriri AR, Bercovier M, Parnas H. Reaction diffusion model of the enzymatic erosion of insoluble fibrillar matrices. Biophys J. 2002; 83:776–93. [PubMed: 12124264]
- 37. Rainey J, Goh M. A statistically derived parameterization for the collagen triple-helix. Protein Sci. 2002; 11:2748–54. [PubMed: 12381857]
- 38. Sasaki N, Odajima S. Stress-strain curve and Young's modulus of a collagen molecule as determined by the X-ray diffraction technique. J Biomech. 1996; 29:655–8. [PubMed: 8707794]
- 39. Gautieri A, Vesentini S, Redaelli A, Buehler MJ. Hierarchical structure and nanomechanics of collagen microfibrils from the atomistic scale up. Nano Lett. 2011; 11:757–66. [PubMed: 21207932]
- 40. Humphrey JD, Wells PB, Thomsen S, Jones MA, Baek S. Histological evidence for the role of mechanical stress in modulating thermal denaturation of collagen. Biomech Model Mechanobiol. 2005; 4:201–10. [PubMed: 16261328]
- 41. Gioia M, Fasciglione GF, Marini S, D'Alessio S, De Sanctis G, Diekmann O, et al. Modulation of the catalytic activity of neutrophil collagenase MMP-8 on bovine collagen I - Role of the activation cleavage and of the hemopexin-like domain. J Biol Chem. 2002; 277:23123–30. [PubMed: 11953425]
- 42. Marini S, Fasciglione GF, de Sanctis G, D'Alessio S, Politi V, Coletta M. Cleavage of bovine collagen I by neutrophil collagenase MMP-8 - Effect of pH on the catalytic properties as compared to synthetic substrates. J Biol Chem. 2000; 275:18657–63. [PubMed: 10749856]
- 43. Stultz CM. Localized unfolding of collagen explains collagenase cleavage near imino-poor sites. J Mol Biol. 2002; 319:997–1003. [PubMed: 12079342]
- 44. Chung L, Dinakarpandian D, Yoshida N, Lauer-Fields JL, Fields GB, Visse R, et al. Collagenase unwinds triple-helical collagen prior to peptide bond hydrolysis. EMBO J. 2004; 23:3020–30. [PubMed: 15257288]

**Figure 1. Comparisons of the cleavage rates between the heterotrimer and the homotrimer without and with applying forces**

(a), Cleavage rates of type I human heterotrimer and homotrimer (data from [20]). The homotrimer is resistant to MMP cleavage (MMP-1, 25°C). (b), Force enhances the cleavage rate of the type I collagen homotrimer (data from [21], MMP-1, Room Temperature). (c), Force slows down the cleavage rate of type I bovine heterotrimer exposed to MMP-8 (data from [22], 37 °C).

**Figure 2. Snapshots of collagen molecules for heterotrimer (top) and homotrimer (bottom) with and without force**

The vicinity of the cleavage site of the heterotrimer is thermally unfolded while the homotrimer behaves stable triple helical structure, indicating it is thermally stable. The applying force stabilizes the cleavage site of the heterotrimer but does not affect the triple helical structure of the homotrimer.

**Figure 3. Structural analyses of the vicinity of cleavage site of a type I mouse heterotrimer and homotrimer collagen molecule without applied force**

(a) The unit heights of heterotrimer at the vicinity of the cleavage site. (b) The radii of heterotrimer at the vicinity of the cleavage site. (c) The unit heights of homotrimer at the vicinity of the cleavage site. (d) The radii of homotrimer at the vicinity of the cleavage site. This figure shows that the cleavage site of the homotrimer adopts a stable triple helical structure while the heterotrimer is thermally unfolded at the cleavage site. The radius of the heterotrimer at the cleavage site is three times larger than it of the homotrimer indicating that each chain is exposed.

**Figure 4. Changes in the lengths of cleavage site with applying force** (a), The lengths of the cleave site (summation from the 5th height to 16th height) with and without force. (b), The time history of the increase of the length for heterotrimer and homotrimer. (c), The strain of each unit height at the vicinity of cleavage site. This figure shows that force induces both entropic and energetic strain to the heterotrimer but only induces energetic strain to the homotrimer. In the heterotrimer, the applied force reduces the entropy of the unfolded region and stabilizes the cleavage site by refolding it into the triple helical structure. In contrast, the force only induces uniform strain in the homotrimer.

**Figure 5. Analyses of the N-O distance at the vicinity of the cleavage site of heterotrimer and homotrimer with and without forces**

(a), Average N-O distances for heterotrimer and homotrimer with and without force. The arrow indicates the distance point from N of Gly residue to O of the residue at the X position of (GXY)n triplets. Distances larger than 0.4 nm are marked in red. (b), Bond life time of N-O distance point from the Gly at the scissile bond for heterotrimer. (c), Bond life time for homotrimer. This figure shows that the force increases the H-bond life time of the heterotrimer but does not alter the H-bond life time of the homotrimer at the cleavage site, indicating that the force induces stabilization in the heterotrimer but not in the homotrimer.