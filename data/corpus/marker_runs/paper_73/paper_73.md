### Contents lists available at [ScienceDirect](www.sciencedirect.com/science/journal/physe)

## Physica E

journal homepage: <www.elsevier.com/locate/physe>

# Mechanical properties of monolayer graphene under tensile and compressive loading

Yuanwen Gao, Peng Hao -

Key Laboratory of Mechanics on Western Disaster and Environment, Ministry of Education, Department of Mechanics and Engineering Science, School of Civil Engineering and Mechanics, Lanzhou University, Lanzhou 730000, PR China

#### article info

Article history: Received 13 March 2009 Received in revised form 15 April 2009 Accepted 24 April 2009 Available online 8 May 2009

Pacs: 61.48.De 62.25.g 31.15.bu 62.20.de 81.05.Uw

Keywords: Monolayer graphene Zigzag graphene Armchair graphene Mechanical property Quantum molecular dynamics

## abstract

The mechanical properties of zigzag graphene and armchair graphene nanoribbon under tensile and compressive loading are studied by the use of quantum mechanics as well as quantum molecular dynamics (MD) method based on the Roothaan–Hall equation and the Newton motion laws. The similar failure mechanisms and different mechanical properties are found in zigzag graphene and armchair graphene subjected to mechanical load. Under tensile or compressive loadings, the critical loading of the zigzag graphene is larger than that of the armchair graphene. Both zigzag graphene and armchair graphene begin to break at the outmost carbon atomic layers. Applied mechanical loading indeed changes the electronic properties of graphene.

& 2009 Elsevier B.V. All rights reserved.

### 1. Introduction

Graphene sheets are a few atoms thick but are nonetheless stable under ambient conditions, metallic, and of remarkably high quality [\[1\]](#page-5-0). Since Novoselov et al. [\[1\]](#page-5-0) and Stankovich et al. [\[2\]](#page-5-0) have recently achieved some valuable and cheaper methods of preparing graphene sheets for mass production usage, the probability of employing graphene sheets for ordinary application has been increased. Many researches on the mechanical and electrical properties of the graphene sheets have been conducted all over the world [\[1–14\].](#page-5-0) The exceptional electronic properties of graphenes, with its charge carriers mimicking relativistic quantum particles and its formidable potential in various applications, have ensured a rapid growth of interest in this new material [\[3\].](#page-5-0) Many graphene nanodevices and structures have already developed, such as the smallest graphene transistor [\[3\]](#page-5-0), graphene fieldeffect transistors [\[4,5\]](#page-5-0) and the bilayer graphene which can be used in low-noise applications [\[6\]](#page-5-0).

To study the mechanical properties of graphene is crucial and very useful in the design and control of nanographene structures or devices. Many researches have already been conducted to measure Young's Modulus of graphene sheets by experimental method and molecular dynamics (MD) simulation in literatures. Frank and Tanenbaum [\[7\]](#page-5-0) measured the effective spring constants of stacks of graphene sheets (less than 5 layers) using atomic force microscope (AFM) and extracted Young's modulus (0.5 Tpa). Gomez-Navarro et al. [\[8\]](#page-5-0) determined the elastic modulus of freely suspended graphene monolayer through tip-induced deformation experiments and obtained Young's modulus (0.25 Tpa). Lee et al. [\[9\]](#page-5-0) measured the elastic properties and intrinsic breaking strength of free-standing monolayer graphene membranes by nanoindentation in an atomic force microscope, and their experiments established graphene as the strongest material ever measured. Except these experimental measurements, the atomistic simulation methods are also employed to obtain Young's modulus, shear modulus, Poisson's ratio and the thickness of graphene [\[10,11\].](#page-5-0) The macroscopic fracture parameters [\[12\]](#page-5-0) and the J-integral [\[13\]](#page-5-0) were investigated on 2D graphene systems by molecular dynamics method. The effects of the large defects and cracks on the mechanical properties of the nanotubes and graphenes were

<sup>-</sup> Corresponding author. Tel.: +86 09318912572. E-mail address: [haop06@lzu.cn \(P. Hao\).](mailto:haop06@lzu.cn)

<span id="page-1-0"></span>investigated by using the coupled Quantum mechanics and molecular mechanics (MM) calculation [14]. Sakhaee-Pour [15] studied the elastic buckling of the single graphene sheets using an atomistic modeling approach. Xu et al. [16] discussed the elastic response of a circular single graphene under a transverse central load using molecular and continuum mechanics modeling. Duan et al. [17] investigated the deformation of a single layer, circular, graphene sheet under a central point load using molecular mechanics simulations. He et al. [18,19] analysed the vibration and resonance of multilayered graphene sheets using continuum mechanics model. Lu et al. [20,21] developed a theoretical framework of nonlinear continuum mechanics for graphene and a new formula of elastic bending modulus for monolayer graphene. Although many researches on mechanical properties and mechanical behaviors of graphene sheets have been conducted by using continuum mechanics model and molecular dynamics method, the influences of electronic properties are not considered. And the fracture mechanism and mechanical behaviors between different types of graphene nanoribbons are not differentiated in detail.

In this paper, we investigate the deformations and the failure process of zigzag graphene and armchair graphene under mechanical loading using quantum molecular dynamics (QMD). Some mechanical and electronic properties of the two typical graphene structures are displayed, and the failure mechanism of zigzag graphene and armchair graphene is investigated. The differences of mechanical behaviors between the two typical graphene are also presented.

#### 2. Simulation techniques

We use quantum mechanics and quantum molecular dynamics technique to carry out the simulations. The positions and velocities of the atoms in the system are predicted by Newton's laws of motion, and the interactions among the atoms are determined by solving the Schrodinger equation as follows:

$$H\Psi = E\Psi \tag{1}$$

where H is the Hamiltonian operator, E is the total energy of the system and  $\Psi$  is the wavefunction of the system. Because it is difficult to solve the Schrodinger equation for large molecular systems, the Born–Oppenheimer approximation which distinguishes the motion of electrons from the motion of the nuclei and the Hartee–Fock approximation which transfers a multi-electronic problem into single electronic problem are employed, so that we get

$$H_i \Psi_i = \varepsilon_i \Psi_i \tag{2}$$

where  $H_i$  is the effective one-electron Hamiltonian,  $\Psi_i$  is the molecular orbital and  $\varepsilon_i$  is the orbital energy of the electron in

molecular orbital  $\Psi_i$ . According to the assumption of the linear combination of atomic orbitals (LACO),  $\Psi_i$  can be written as

$$\Psi_i = \sum_{\mu} C_{\mu i} \phi_{\mu} \tag{3}$$

where  $\phi_{\mu}$  is the atomic  $\mu$ th orbital and  $C_{\mu i}$  is the coefficient.

The matrix form of Hartree–Fock equation, which is the Roothaan equation, can be written as [22]

$$FC = SCE \tag{4}$$

where the Fock matrix  $\mathbf{F}$  can be described as follows [23,24]:

$$\mathbf{F}_{\mu\nu} = \int d_{\nu}\phi_{\mu} \left[ -\frac{1}{2}\nabla_{i}^{2} - \sum_{A=1}^{M} \frac{Z_{A}}{r_{iA}} \right] \phi_{\nu}$$

$$+ \sum_{\lambda=1}^{K} \sum_{\sigma=1}^{K} P_{\lambda\sigma}[(\mu\nu|\lambda\sigma) - \frac{1}{2}(\mu\lambda|\nu\sigma)] + V_{\mu\nu}$$
(5)

where  ${\bf P}$  is the charge density matrix,  ${\bf C}$  is the coefficient matrix, and  ${\bf E}$  is the orbital energy diagonal matrix.  $P_{\lambda\sigma}=2\sum_{i=1}^{N} \frac{1}{2}c_{\lambda i}c_{\sigma i}(\mu\lambda|\lambda\sigma)$  and  $(\mu\lambda|\lambda\sigma)$  are the two-electron integrals that may involve up to four different basis functions  $(\phi_{\mu}, \phi_{\nu}, \phi_{\lambda}, \phi_{\sigma})$ ,  $V_{\mu\nu}$  is the influence of external fields,  ${\bf S}$  is the overlap integrals matrix and  $S_{\mu\nu}=\int d_{\nu}\phi_{\mu}\phi_{\nu}$ .

The semi-empirical quantum mechanics method PM3 [25] is employed in the calculation. By solving Roothaan–Hall equation, the total energy and atomic interaction force can be obtained. The total energy contains the electronic energy and the energy of the interaction between nuclei.

#### 3. Results and discussion

## 3.1. Initial condition

The geometrical structure of zigzag graphene and armchair graphene is shown in Fig. 1. The size of zigzag graphene is  $1.48 \times 0.7 \, \mathrm{nm}^2$  (length  $\times$  width); the size of armchair graphene is  $1.99 \times 0.738 \, \mathrm{nm}^2$  (length  $\times$  width). By increasing and decreasing the displacement of the atom in the frame of Fig. 1, we can apply tensile and compressive loading [26,27]. The temperature of the calculation is set to 0 K and then the influence of thermal vibration is neglected.

The total energy minimization of system is implemented by quantum mechanical method. Unlike in molecular dynamics, the kinetic energy is not considered in the energy minimization (geometry optimization) to obtain more stable and reliable equilibrium. And the most important advantage of quantum mechanics method is that the effects between the mechanical and electronic behaviors can be investigated [23].

Fig. 1. The structure of zigzag and armchair graphene.

In the released state of equilibrium, the charge distribution in graphene deviates from the electric neutral state, and the charge decreases from the edge to the middle of graphene.

## 3.2. Deformation analysis

To find the differences of the mechanical properties between zigzag graphene and armchair graphene, we calculate the deformation of zigzag graphene and armchair graphene when they are stretched and compressed, respectively.

Fig. 2 shows the curve of the deformation of graphene versus the tensile load. The arrows in the figure indicate the critical fracture load of the graphene. Apparently, the critical fracture load of the zigzag graphene is larger than that of the armchair graphene. The total critical deformation of zigzag graphene is about 0.24 nm, corresponding stretched ratio is about 16%; it is about 3% lower than that of the armchair graphene. Linear

Fig. 2. Deformation of zigzag and armchair graphene versus tensile load.

Fig. 3. The strain of zigzag and armchair graphene versus stress.

relationship between tensile load and deformation can be observed in the whole stretched process of zigzag graphene, but significant nonlinear deformation can be seen over 358 kcal/mol/A˚ for armchair graphene.

Fig. 3 shows the relationship between the strain and the stress. The slope of the curve indicates that the Young's modulus of zigzag graphene and armchair graphene decreases slowly with the strain increasing. The Young's modulus of armchair graphene decreases nearly to zero when the strain increases to 0.14. The Young's modulus of zigzag graphene is about 0.6 Tpa concluded from the curve, when the thickness of monolayer graphene taken as 0.335 nm [\[28\].](#page-5-0) The Young's modulus of armchair graphene is about 1.1 Tpa, which is larger than that of zigzag graphene. The result is close to the experimental measurement [\[7,9\]](#page-5-0) and the prediction of numerical simulation [\[10\].](#page-5-0)

Under the compressive loading, the deformation of graphene in-plane is very small before losing stability. The characteristic curves of the relationship between the uniaxial strain and the transverse displacement are plotted in Fig. 4. The critical strain of zigzag graphene is larger than that of armchair graphene. We find the critical load of zigzag graphene is 3.75 nN while that of armchair graphene is 2.8 nN.

The critical load of zigzag graphene is larger than that of armchair graphene under tensile and compressive loading, because the C–C bonds of zigzag graphene and armchair graphene subjected to mechanical load are different. The C–C bonds subjected to applied force of zigzag graphene are those intersectant with the applied force. In the case of the armchair graphene, the C–C bonds subjected to applied force are those parallel to the applied force as shown in [Fig. 1.](#page-1-0)

## 3.3. Failure mechanism of graphene under mechanical loading

[Figs. 5](#page-3-0)(a) and (b) show the equilibrium C–C bond length among each layer of carbon atoms along the uniaxial direction of the graphene under the tensile load. The abscissa axis in the figure indicates the layer number as denoted in [Fig. 1.](#page-1-0) When zigzag graphene and armchair graphene are stretched, the increment of the C–C bond between the 2nd and 3rd layer is the largest as shown in [Figs. 5](#page-3-0)(a) and (b). In [Fig. 5](#page-3-0)(a), with the tensile load increasing, the localization of deformation is more significant. When the tensile load reaches 564.05 kcal/mol/A˚ , which is a value very close to the critical failure value, the C–C bond length at the edge of zigzag graphene increases sharply. The similar

Fig. 4. The strain of zigzag and armchair graphene versus transverse displacement.

<span id="page-3-0"></span>Fig. 5. The C–C bond lengths change with external actions: (a) change of zigzag C–C bond length and (b) change of armchair C–C bond length.

characteristics of armchair graphene can be found in Fig. 5(b). It can be noticed that both of the two graphene begin to break at their outmost atom layer under mechanical loading in this figure.

When graphene resists compressive loading, the largest increment of C–C bond length of zigzag graphene and armchair graphene appears at the outmost atom layer of the graphene, as shown in Figs. 6(a) and (b).

[Figs. 7\(](#page-4-0)a) and (b) show the failure form of zigzag graphene and armchair graphene under tensile loading. It is clearly revealed that the broken C–C bonds appear at the edge atom layer of graphene. All of these indicate that the graphene will break at the outmost layer under the external tensile and compressive load.

## 3.4. The electronic properties under load

Previous researches have shown that both geometrical and electronic effects on the field-emission properties of open-ended CNTs are significant [\[29\],](#page-5-0) and that the effects of mechanical load on electronic properties of nanotubes have extraordinary importance [\[29\].](#page-5-0) The electronic structure of graphene under different planar strain distributions is changed [\[30\].](#page-5-0) In this

Fig. 6. The C–C bond lengths change with external actions under compressive load: (a) change of zigzag C–C bond length and (b) change of armchair C–C bond length.

section, the effects of the mechanical load on the electronic properties of graphene are investigated by using quantum mechanics outlined in Section 2.

[Figs. 8\(](#page-4-0)a) and (b) show the highest energy-occupied molecular orbital (HOMO), the lowest energy-unoccupied molecular orbital (LUMO) energy and the energy gap of zigzag graphene and armchair graphene under tensile load. As shown in [Fig. 8](#page-4-0)(a), the energy gap of zigzag graphene increases first from -3.343 to -3.093 eV with the increase of tensile load, then decreases at -3.883 eV when the tensile load reaches 30.28 nN, which is the value close to the critical failure value of zigzag graphene. The range of the energy gap is between -0.25 and 0.54 eV. However, the energy gap of armchair graphene decreases by about 87% with the increase of the tension before the fracture occurs in the graphene. It is obvious that the effect of tensile load on the electronic properties of armchair graphene is much larger than that of zigzag graphene. In other words, the electronic properties of zigzag graphene are more stable than those of armchair graphene under tensile loading.

When graphene resists to compressive load, the highest energy-occupied molecular orbital, the lowest energy-unoccupied molecular orbital energy and the energy gap of zigzag graphene and armchair graphene unremarkably change as shown in [Fig. 9.](#page-4-0) However, the energy gap of zigzag graphene increases with the

<span id="page-4-0"></span>Fig. 7. The fracture of graphene under tensile load.

Fig. 8. Variation of the Lumo, Homo and energy gap with tensile load. (a) Zigzag; (b) Armchair.

Fig. 9. Variation of the Lumo, Homo and energy gap with applied compressive load. (a) Zigzag; (b) Armchair.

increase of compressive load, while that of armchair graphene decreases.

## 4. Conclusions

The mechanical properties of zigzag graphene and armchair graphene nanoribbon are found different by using the quantum mechanics and quantum molecular dynamics. It can be concluded that the critical mechanical loads for failure and buckling of zigzag graphene are larger than those of armchair graphene. The simulation results indicate that both zigzag graphene and armchair graphene break at the outmost atom layer under the external mechanical load. The external mechanical load can alter the electronic properties of graphene, which affects field-emission properties of graphene. The results are important in the design of graphene nanodevices.

#### Acknowledgments

We thank Prof. Xiaojun Yao, Miss Ying Yang and Mr. Longfei Li for valuable comments and suggestions. This work was supported by the National Natural Science Fund of China (10672070) and Program for the New Century Excellent Talents (NCET-06-0896).

#### <span id="page-5-0"></span>References

- [1] K.S. Novoselov, A.K. Geim, S.V. Morozov, D. Jiang, Y. Zhang, S.V. Dubonos, I.V. Grigorieva, A.A. Firsov, Science 306 (2004) 666.
- [2] S. Stankovich, Dmitriy A. Dikin, Geoffrey H.B. Dommett, Kevin M. Kohlhaas, Eric J. Zimney, Eric A. Stach, Richard D. Piner, SonBinh T. Nguyen, Rodney S. Ruoff, Nature 442 (2006) 282.
- [3] L.A. Ponomarenko, F. Schedin, M.I. Katsnelson, R. Yang, E.W. Hill, K.S. Novoselov, A.K. Geim, Science 320 (2008) 356.
- [4] Chong-an Di, Dacheng Wei, Gui Yu, Yunqi Liu, Yunlong Guo, Daoben Zhu, Adv. Mater. 20 (2008) 328.
- [5] Xinran Wang, Yijian Ouyang, Xiaolin Li, Hailiang Wang, Jing Guo, Hongjie Dai, Phys. Rev. Lett. 100 (2008) 206803.
- [6] Yu-Ming Lin, Phaedon Avouris, Nano Lett. 8 (2008) 2119.
- [7] I.W. Frand, D.M. Tanenbaum, A.M. van der Zande, P.L. McEuen, J. Vac. Sci. Technol. B 25 (2007) 2558.
- [8] Cristina Go´mez-Navarro, Marko Burghard, Klaus Kern, Nano Lett. 8 (2008) 2045.
- [9] Changgu Lee, Xiaoding Wei, Jeffrey W. Kysar, James Hone, Science 321 (2008) 385.
- [10] Fang Liu, Pingbing Ming, Ju Li, Phys. Rev. B 76 (2007) 064120.
- [11] Y. Huang, J. Wu, K.C. Hwang, Phys. Rev. B 74 (2006) 245413.

- [12] Y. Jin, F.G. Yuan, J. Nanosci. Nanotech. 5 (2005) 601.
- [13] Y. Jin, F.G. Yuan, J. Nanosci. Nanotech. 5 (2005) 2099.
- [14] Roopam Khare, Steven L. Mielke, Jeffrey T. Paci, Sulin Zhang, Roberto Ballarini, George C. Schatz, Ted Belytschko, Phys. Rev. B 75 (2007) 075412.
- [15] A. Sakhaee-Pour, Solid State Commun. 149 (2009) 91.
- [16] Xiaojing Xu, Kin Liao, Mater. Phys. Mech. 4 (2001) 148.
- [17] W.H. Duan, C.M. Wang, Nanotechnology 20 (2009) 075702.
- [18] S. Kitipornchai, X.Q. He, K.M. Liew, Phys. Rev. B 72 (2005) 075443.
- [19] X.Q. He, S. Kitipornchai, K.M. Liew, Nanotechnology 16 (2005) 2086.
- [20] Q. Lu, M. Arroyo, R. Huang, J. Phys. D: Appl. Phys. 42 (2009) 102002.
- [21] Q. Lu, R. Huang, J. Comput. Theor. Nanosci., 2008, accepted.
- [22] C.C.J. Roothaan, Rev. Mod. Phys. 23 (1951) 69. [23] Yufeng Guo, Wanlin Guo, J. Phys. D: Appl. Phys. 36 (2003) 805.
- [24] A.R. Leach, Molecular Modelling, Addision Wesley Longman Limited, London, 1996.
- [25] J.J.P. Stewart, J. Comput. Chem. 10 (1989) 209.
- [26] Shen Haijun, Yao Weixing, Shi Youyin, Comput. Appl. Chem. 21 (2004) 485.
- [27] Shen Haijun, Shiyoujin, J. Mech. Sci. Eng. 25 (2007) 341.
- [28] G. VanLier, C. VanAlsenoy, V. Vandoren, P. Geerlings, Chem. Phys. Lett. 326 (2000) 181.
- [29] Gang Zhou, Wenhui Duan, Binglin Gu, Phys. Rev. Lett. 87 (2001) 095504.
- [30] Gui Gui, Jin Li, Jianxin Zhong, Phys. Rev. B 78 (2008) 075435.