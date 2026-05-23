Contents lists available at [ScienceDirect](http://www.elsevier.com/locate/eml)

## Extreme Mechanics Letters

journal homepage: [www.elsevier.com/locate/eml](http://www.elsevier.com/locate/eml)

# Reduced ballistic limit velocity of graphene membranes due to cone wave reflection

Zhaoxu Meng [a](#page-0-0) , Amit Singh [b](#page-0-1) , Xin Qin [c](#page-0-2) , Sinan Keten [a,](#page-0-0)[b,](#page-0-1) [\\*](#page-0-3)

- <span id="page-0-0"></span><sup>a</sup> *Department of Civil and Environmental Engineering, Northwestern University, 2145 Sheridan Road, Evanston, IL 60208-3111, United States*
- <span id="page-0-1"></span><sup>b</sup> *Department of Mechanical Engineering, Northwestern University, 2145 Sheridan Road, Evanston, IL 60208-3111, United States*
- <span id="page-0-2"></span>*Theoretical and Applied Mechanics Program, Northwestern University, 2145 Sheridan Road, Evanston, IL 60208-3111, United States*

## a r t i c l e i n f o

#### *Article history:* Received 11 April 2017 Received in revised form 30 May 2017 Accepted 1 June 2017 Available online 12 June 2017

*Keywords:* Graphene membrane Ballistic limit velocity *V*<sup>50</sup> Cone wave reflection Size-dependent Molecular dynamics simulation Analytical relationship

## a b s t r a c t

Recent microscale ballistic experiments have revealed that multilayer graphene membranes exhibit exceptionally high ballistic limit velocity and specific penetration energy. A key feature contributing to the exceptional performance of these systems is the cone wave that develops at impact, which propagates radially at a very high speed for ultra-light and stiff graphene membranes, distributing the kinetic energy of the projectile away from the impact zone. Current theories on ballistic impact consider infinitely wide membranes, and atomistic simulations involve very small projectiles and specimen dimensions, and thus cannot ascertain whether microscale ballistics observations are scalable and size-independent. Here, we discover a particular size effect due to the reflection of cone wave that has not been previously observed or considered. We present molecular dynamics simulations showing that there exists a critical membrane size below which the cone wave reflections from the boundaries induce perforation; a phenomenon that is particularly relevant for microballistic testing of graphene membranes. We present an analytical relationship, verified by simulation data, which predicts the critical membrane size simply as a function of the projectile size, membrane thickness and the ratio of the projectile and membrane densities. Our findings provide timely guidance for future microscale experiments and atomistic simulations for accurate characterization of the impact performance of 2D nanomaterials.

© 2017 Elsevier Ltd. All rights reserved.

## **1. Introduction**

Understanding ballistic impact response of materials is important for diverse applications that occur at small as well as large scales, including therapeutic delivery of nanoparticles to cells [\[1](#page-7-0)[,2\]](#page-7-1), designing body armors [\[3\]](#page-7-2), or space debris shields [\[4\]](#page-7-3). In designing a ballistic barrier, the common goal can be simply stated as resisting the highest projectile velocity with the lowest mass possible [\[5\]](#page-7-4). The figures of merit for the ballistic performance of materials at the macroscale are ballistic limit velocity (*V*50), which is referred to as the velocity required for the projectile to reliably (at least 50% of the time) penetrate the barrier [\[6\]](#page-7-5), or the loss of projectile kinetic energy upon penetration [\[7](#page-7-6)[,8\]](#page-7-7). Membranes or thin films with excellent out-of-plane flexibility and low mass to area ratio are known to perform well as ballistic barriers [\[9](#page-7-8)[–14\]](#page-7-9).

Micro-scale ballistic experiments, most notably the laserinduced projectile impact tests (LIPIT) established recently broke new ground in this area [\[15](#page-7-10)[,16\]](#page-7-11), bringing projectile and membrane sizes down to the micro and nanoscale. This now allows

<span id="page-0-3"></span>\* Corresponding author. *E-mail address:* [s-keten@northwestern.edu](mailto:s-keten@northwestern.edu) (S. Keten). direct characterization of the influence of nanostructures on ballistic performance, leading to new barrier concepts based on polymer nanocomposites and 2D nanomaterials [\[17–](#page-7-12)[20\]](#page-7-13). Graphene is naturally a great candidate as ballistic membranes because of its exceptional strength and stiffness as measured from atomic force microscopy measurements [\[21–](#page-7-14)[23\]](#page-7-15). LIPIT experiments on multilayer graphene (MLG) have shown that the specific penetration energy for MLG membrane is about 10 times more than the corresponding literature values for macroscopic steel sheets [\[16\]](#page-7-11), which illustrates MLG's potential as a ballistic barrier material. These findings can be explained by the delocalization of concentrated stress, as well as graphene's exceptional specific strength. Specifically, the failure process of MLG membrane involves a cone wave developed in the wake of the tensile implosion wave, and radially propagating cracks that form petals [\[16\]](#page-7-11). The deformation of the ultra-thin free-standing membrane around the expanding cone wave propagates the kinetic energy transferred through projectile impact to a much larger area, which gives rise to the excellent energy dissipation capability of MLG.

Theoretical analyses on membrane ballistic impact response carried out so far assume that the cone wave propagates continually outwards on an infinite membrane. In reality, experimental specimens are finite and the typical free span of MLG samples in the microballistic response experiments are around 85 um in width [16]. Although atomistic molecular dynamics (MD) simulations have been successful in simulating the failure mechanisms of graphene under ballistic projectile impact [24-26], they have had projectile dimensions that are very small to minimize computational effort, typically a few nanometers, which leads to highly localized failure processes. These studies have typically involved computationally expensive bond-order potentials to account for bond rupture [27–30]. Since no substantial chemical reactions or bond reconfigurations are observed under the short time-scale of ballistic impact [25], multiscale computational approaches such as coarse-grained (CG) MD simulations and continuum theories based on mechanical deformation offer greater promise to bridge different length and time scales in order to better understand failure processes. Systematic size-dependence studies are yet to be carried out, and many questions remain unanswered in this regard. For example, how are micro-scale ballistic experiments different from macro-scale experiments, and could there be physical phenomena that occur when the thickness and length of the membranes become small as in the case of micro-scale ballistic experiments? Likewise, can the observations made from MD simulations be generalized to explain these experiments?

Here we study the effect of finite specimen size on the ballistic response of single and multi-layer graphene membranes, particularly focusing on the effects of cone wave reflection from boundaries on the ballistic performance, by performing MD simulations with previously developed and validated CG model of graphene. [31]. We also study the projectile impact behaviors of graphene membranes with different sizes, thicknesses and shapes for the suspended free-standing region.

## 2. Materials and methods

We employ the Large-scale Atomic/Molecular Massively Parallel Simulator (LAMMPS) software to carry out all the MD simulations [32]. We use the Visual Molecular Dynamics (VMD) software to visualize the atomic configurations during the impact and penetration process [33].

The CG model of graphene used in this study adopts 4-to-1 mapping scheme conserving the hexagonal lattice symmetry of graphene. The force field parameters are calibrated by experimentally reported mechanical properties [31]. We note that the Morse bond potential is recently revised as  $D_0=479.535~{\rm kcal/mol}$ ,  $\alpha=0.99~{\rm Å}^{-1}$  and  $r_{cut}=3.5~{\rm Å}$  [23] to match the nanoindentation predicted strength of  $130\pm10~{\rm GPa}$  [21]. The bonds that are stretched beyond  $r_{cut}$  are broken and subsequently deleted in the simulations. We use the command 'fix bond/break' in LAMMPS package to simulate bond breaking during the simulation run, and the attempt for possible bond breaking is performed every 5 MD time-steps, equaling 20 femtoseconds. The model has been shown to capture the elasticity, strength and fracture properties of monolayer and multilayer graphene very well and has the potential to capture mesoscale failure mechanisms and size effects in the MLG systems [17,18,23].

The projectiles are made of CG beads forming a diamond cubic lattice structure with a lattice constant of 0.72 nm, which is double the actual lattice constant of diamond, consistent with the CG mapping scheme of graphene. The default projectile bead mass is 96 g/mole, which yields a density of 3.42 g/cm³, close to the actual diamond density of 3.5 g/cm³ [34]. In our simulations, the projectile is treated as a rigid body, since no appreciable deformation of the projectile is found in the microballistic experiments [16]. Previous atomistic MD simulation also shows that there is negligible deformation for the projectile during the impact process when modeled using Tersoff potential for the atomistic diamond structure [25]. The interactions between the projectile and graphene

membrane is modeled by the same 12-6 Lennard Jones (LJ) potential as the ones employed between graphene beads in the original CG model:  $\varepsilon_{IJ}=0.82$  kcal/mol and  $\sigma_{IJ}=3.46$  Å [31]. We have verified that the LJ parameters have a negligible effect on the ballistic response; specifically, the reactive force and projectile velocity evolution are approximately the same with varying  $\varepsilon_{IJ}$  parameter. Higher cohesive energies ( $\varepsilon_{IJ}$ ) result in a larger variance in the force measurements, but the mean values do not vary substantially.

The system is first equilibrated under an NVT ensemble at 10 K with the projectile fixed at the initial position, with the radial coordinate the same as the center of the square sheet. The initial perpendicular distance between the surface of the projectile and membrane is larger than 5 nm so that there is no initial interaction between them. A low initial temperature is chosen to minimize thermal noise in the results. To simulate ballistic impact, the projectile is given an initial velocity projected towards the graphene membrane with the system running under NVE ensemble to conserve the total energy, following same protocol used in previous studies [24–26]. We then obtain the projectile velocity and reactive force profiles during the impact process for further analysis.

#### 3. Results and discussions

First, we present results for the dependence of  $V_{50}$  on the projectile radius and velocity and show an interesting size-dependent phenomenon. We obtain  $V_{50}$  as the lowest velocity that permits penetration in the simulations, in line with the convention used in the analysis by Phoenix and Porwal [9].

An analytical expression for the  $V_{50}$  of an infinite, linear elastic, thin isotropic membrane [9] has been derived as:

<span id="page-1-0"></span>
$$V_{50} = \sqrt{2}(1+\Gamma)c_0(\varepsilon_{\text{max}}/K_{\text{max}})^{3/4},\tag{1}$$

where  $\Gamma=\frac{m_m}{m_p}$  is the mass ratio of the part of the membrane  $(m_m)$  in contact with the projectile  $(m_p)$ ,  $c_0=\sqrt{\frac{E}{\rho}}$  is the elastic wave speed in the membrane,  $\varepsilon_{\max}$  is the failure strain of the membrane and  $K_{\max}$  is a strain concentration factor depending on both  $\Gamma$  and  $\varepsilon_{\max}$ . For simplicity, we assume that  $K_{\max}$  mildly depends on  $\Gamma$ , so for a given membrane material,  $\varepsilon_{\max}$  and  $K_{\max}$  can be taken as constants. Given that  $m_m \sim r_p^2$ , and  $m_p \sim r_p^3$  for spherical projectiles, a rough scaling yields that  $V_{50} \sim A + B/r_p$  from Eq. (1).

Fig. 1(a) presents CG MD simulation results based on a freestanding membrane with a radius of a = 100 nm to compare the effect of different projectile sizes and initial velocities (velocity before impact)  $(V_0)$  on the residual velocities (velocity after impact)  $(V_r)$ . The ballistic impact behavior of the membrane can be broken into three distinct regions based on  $V_r$ . Region I corresponds to the scenario when the projectile cannot penetrate the membrane and bounces off, as indicated by the negative values of  $V_r$ . At greater initial velocities, the residual velocity suddenly changes from negative to positive, marking the onset of region II where there is a transition from projectile bouncing back to penetration. In this regime, an obvious cone wave forms and expands outwardly before projectile penetrating the membrane, and thus dissipates more projectile kinetic energy through larger area. Further increasing the initial velocity leads to region III where the residual velocity scales linearly with the initial velocity. In this regime, the projectile immediately perforates the membrane locally, where a linear scaling can be expected from conservation of momentum. Fig. 1(a) also illustrates that  $V_{50}$  decreases with increasing projectile size, which is consistent with the continuum prediction from Eq. (1) and macroscale experiments [9].

The most intriguing observation from Fig. 1(a) is that for a larger projectile radius of  $r_p = 15$  nm plotted in magenta, even very small initial velocities result in perforation. The specific  $V_{50}$  values for a wider range of projectile sizes are presented in Fig. 1(b).

<span id="page-2-0"></span>**Fig. 1.** (a). Residual velocity (*Vr*) vs. initial velocity (*V*0) relationship for different projectile sizes for a circular membrane with a radius of 100 nm. The inset shows the general shape of *Vr*–*V*<sup>0</sup> relationship and the three corresponding regions. (b). Ballistic limit velocity *V*<sup>50</sup> vs. projectile radius. The analytical scaling function of *V*<sup>50</sup> = *A* + *B*/*r<sup>p</sup>* is shown in the red dashed line. The error bars correspond to the largest range of *V*<sup>50</sup> for each case obtained after multiple runs. (For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)

To get the statistical errors of *V*50, we have run multiple replications of the simulation using a range of initial velocities in region II with an increment of 10 m/s for each case. We fit a curve *V*<sup>50</sup> = *A*+*B*/*r<sup>p</sup>* for the projectile radii smaller than 8 nm based on the scaling of Γ . The *V*<sup>50</sup> values for smaller projectile sizes follow this scaling, however for larger projectiles, the *V*<sup>50</sup> deviates considerably from the predicted scaling. This size-dependent deviation cannot be explained by current theoretical models and to our knowledge has not been reported in any previous experimental or simulation studies.

To better understand the reason for this deviation, we visualize the complete impact process from simulation trajectories. We observe that for a large projectile, the cone wave propagates outward radially until it reflects from the circular clamped boundary of the membrane as seen in [Fig. 2\(](#page-2-1)b). The reflected cone wave travels back towards the projectile, further deforming the membrane downward, and upon reaching the surface of the projectile, it generates large local strain that leads to immediate perforation of the membrane, as shown in [Fig. 2\(](#page-2-1)c) and (d). A distinctive feature of this failure mechanism is that fracture petals in this case point towards the impact side of the membrane rather than the back side. These observations indicate that when the projectile to membrane size ratio is relatively large, the membrane loses part of the projectile proof capability due to the cone wave reflection

<span id="page-2-1"></span>d

**Fig. 2.** Snapshots for the cone wave (a) propagating outwardly, (b) getting reflected and returning towards the center (the cone wave front is highlighted by dark shadow), (c) encountering the projectile surface, and (d) the subsequent catastrophic failure of membrane. The whole process can be viewed in the supplementary video. The simulation conditions are: *a* = 100 nm, *r<sup>p</sup>* = 15 nm and *V*<sup>0</sup> = 500 m/s.

from the boundaries, as indicated by the significant drop of *V*<sup>50</sup> value compared to the theoretical prediction. This issue brings forward the following interesting questions: What is the critical size relationship between the membrane and the projectile dimensions that governs when the membrane loses its projectile proof capability? Which material properties influence this scaling relationship?

We define the critical membrane size *a<sup>c</sup>* for a given projectile size *r<sup>p</sup>* as the size at which the cone wave returns to the projectile at the instance of maximum deflection. At this moment, the projectile velocity becomes zero. We calculate two periods of time: (a) the total time *t<sup>c</sup>* taken by the cone wave to reflect from the boundary and return to the surface of the projectile, and (b) the time *t<sup>p</sup>* taken by the projectile to come to a full stop. Setting these two times as equal will yield the critical size relationship.

According to Phoenix and Porwal's work [9], after a sharp increase in the speed of expanding cone wave, the wave speed stays approximately constant  $(v_{c0})$  afterwards. However, the speed of the reflected cone wave will be different because the wave speed depends on the local membrane strain of the wave front. We have derived  $t_c$  in a similar fashion to Ref. [9], which is summarized in the Appendix A. This analysis yields the following relation for the total wave travel time:

$$t_c = k_1 \frac{2a_c}{v_{c0}}, \quad k_1 \cong 0.78,$$
 (2)

where  $k_1$  is introduced to account for the higher reflected wave speed.

To calculate  $t_p$ , the reactive force expression [9] is used:

<span id="page-3-0"></span>
$$F = \pi \rho_m h_m V(t) d(r_c)^2 / dt, \tag{3}$$

where  $r_c$  is the cone shape region radius,  $\rho_m$  and  $h_m$  are the membrane density and thickness, respectively, and V(t) is the projectile velocity as a function of time, which decreases with time during impact. Although  $r_p$  does not appear in Eq. (3), the deceleration rate depends on  $m_p$ , and thus,  $r_p$  plays a role inside V(t).

In the finite system, Eq. (3) is valid before the cone wave reaches the boundary at time  $t_1$ . To get a more accurate expression of the reactive force, we divide the overall process into two phases. In the first phase, the cone shape region expands with the constant cone wave speed,  $r_c = v_{c0}t$ , so the reactive force can be approximately written as:

<span id="page-3-1"></span>
$$F_1 = 2\pi \,\rho_m h_m V(t) \,v_{c0}^2 t. \tag{4a}$$

During the second phase, when the cone wave reaches the boundary of the membrane,  $r_c$  cannot increase any further. Simulations indicate that the reactive force stays more or less a constant in this phase. Thus, it is reasonable to express the constant force as:

<span id="page-3-2"></span>
$$F_2 = 2\pi \,\rho_m h_m V(t_1) \,v_{c0}^2 t_1. \tag{4b}$$

We compare force estimation Eqs. (4a)–(4b) using simulation results of V(t) and  $v_{c0}$  with the actual reactive force results. Fig. 3(a) shows the specific comparison for the largest system we have  $(a_c=200 \text{ nm} \text{ and } r_p=21 \text{ nm})$ . The force estimation matches well with the simulation data. Specifically, it captures the increasing-decreasing trend in the first phase. The total error is less than 15%, which is obtained by calculating the total areas under both actual force and estimated force curves (i.e.  $\int_0^{t_p} Fdt$ ). We note that the accuracy of Eqs. (4a)–(4b) increases with the size of the system, because the larger systems better satisfy the assumptions involved in deriving Eq. (3). We also note that the peak in the actual force in Fig. 3(a) occurs when the reflected wave encounters the projectile. Finally, the force drops to zero sharply, indicating the failure of the membrane (corresponding to Fig. 2(b)–(c)).

For the complete impact process, the impulse given to the projectile of mass  $m_p$  in time  $t_p$  is equal to the total change in its momentum due to the reactive force, F. This can be expressed as:

$$\int_0^{t_p} Fdt = \int_0^{t_1} F_1 dt + \int_{t_1}^{t_p} F_2 dt = m_p v_p.$$
 (5)

Solving the above equation by assuming that the velocity decays with a scaling that lies between linear  $(V(t) = v_p - v_p t/t_p)$  and quadratic function of time  $(V(t) = v_p - v_p t^2/t_p^2)$ , as shown in Fig. 3(b), we then obtain the projectile halting time:

$$t_p = \frac{2k_2}{v_{c0}} r_p^{1.5} \left(\frac{\rho_p}{\rho_m}\right)^{0.5} \left(\frac{1}{h_m}\right)^{0.5},\tag{6}$$

<span id="page-3-3"></span>**Fig. 3.** (a) Comparison of actual reactive force on the projectile from the simulation (black solid line) and the estimated reactive force using Eqs. (4a)–(4b) (red dashed line). (b) Actual velocity from simulation vs. linear and quadratic decay velocity. The simulation conditions for (a–b) are:  $a_c = 200$  nm,  $r_p = 21$  nm and  $V_0 = 500$  m/s. (c) Simulation results for critical membranes sizes and their corresponding projectile sizes. The red and blue curves represent the theoretical results based on Eq. (7). (For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)

where  $k_2$  lies in the range of 0.75–0.91, and the lower and upper bounds correspond to quadratic and linear velocity decay, respectively.

<span id="page-4-1"></span>**Fig. 4.** Snapshots for the reflected cone wave in the bilayer graphene membrane (a) returning towards the center, (b) encountering the projectile surface, and (c) the subsequent catastrophic failure of membrane (the inset shows the side view of the bilayer graphene membrane). The simulation conditions are:  $a_c = 100$  nm,  $r_p = 12.5$  nm,  $\rho_p = 6.84$  g/cm<sup>3</sup> and  $V_0 = 500$  m/s.

Setting  $t_p = t_c$ , we finally have

<span id="page-4-0"></span>
$$a_c = \frac{k_2}{k_1} r_p^{1.5} \left(\frac{\rho_p}{\rho_m}\right)^{0.5} \left(\frac{1}{h_m}\right)^{0.5},\tag{7}$$

In our simulations,  $\frac{\rho_p}{\rho_m}=1.55~(\rho_p=3.42~\frac{\rm g}{{\rm cm}^3})$  and  $\rho_m=2.2~\frac{\rm g}{{\rm cm}^3}),~h_m=0.34~{\rm nm}$  for monolayer graphene, as a result  $\frac{k_2}{h_2}(\frac{\rho_p}{\rho_m})^{0.5}(\frac{1}{h_m})^{0.5}$  lies between 2.05–2.49 nm<sup>-0.5</sup>, and the lower and upper bounds correspond to quadratic and linear velocity decay, respectively.

From CG MD simulation, we determine the critical membrane sizes for different projectile sizes by tracking whether the failure caused by the reflected cone wave happens at approximately the same time when the projectile attains the zero velocity. The results are shown in Fig. 3(c), where the theoretical predictions  $a_c = Cr_p^{1.5}$  with C equals to 2.05 and 2.49 nm<sup>-0.5</sup> (linear and quadratic velocity decay bounds) are also plotted. The theoretical fitting agrees very well with our simulation results.

To further verify the accuracy of the critical scaling relationship Eq. (7), such as the density ratio and the membrane thickness, we have conducted several additional simulations. First, we change the densities of the projectile and membrane while keeping their ratio  $\frac{\rho_p}{\rho_m}$  a constant and we observe that the critical size relationship between the membrane and the projectile remains the

same. Then, by doubling the projectile density  $\rho_p$  while keeping the same membrane density, we find that the critical projectile size decreases for a given membrane size, as shown in Table 1. Given  $a_c$ ,  $\rho_m$  and  $h_m$  as constants, Eq. (7) yields  $(\frac{r_{p1}}{r_{p2}})^{1.5} = (\frac{\rho_{p2}}{\rho_{p1}})^{0.5}$ , where  $\frac{\rho_{p2}}{\rho_{p1}} = 2$ . From our simulation data, we also have  $\log_2 \left(\frac{r_{p1}}{r_{p2}}\right)^{1.5} \approx 0.5$ , thus verifying the scaling order for  $\frac{\rho_p}{\rho_m}$  as 0.5. Finally, we have also run monolayer to trilayer membrane simulations, and we find that the projectile density should increase double and triple times accordingly for bilayer and trilayer membrane to keep the critical size relationship between membrane and projectile, thus verifying the exponent of  $\frac{1}{h_m}$  also as 0.5. Specifically, Fig. 4 shows the cone wave reflection process for a bilayer membrane with  $a_c = 100$  nm,  $r_p = 12.5$  nm and  $\rho_p = 6.84$  g/cm³. The reflected cone wave also emerges (Fig. 4(b)), and when it encounters the bullet surface, immediate perforation occurs (Fig. 4(c)).

Moreover, Eq. (7) has several important ramifications. First, the critical membrane size increases with the projectile size to the power of 1.5, showing greater increase needed to achieve criticality as the projectile size increases. For a silica projectile with size  $r_p = 2~\mu$ m, used in recent experiments [16], the calculated critical membrane size is  $\sim 200~\mu$ m for monolayer graphene. A larger thickness alleviates the reflected cone wave effect, while even for multilayer graphene, say 50 graphene layers, the critical membrane size is still  $\sim 28~\mu$ m. It should also be noted that the reflected cone wave effect intensifies with higher projectile density. If gold projectiles instead of silica were used in recent experiments [16], the corresponding critical membrane size would increase by 2.7 times. These estimations indicate that it should be possible to observe and study the cone wave reflection effect with current microballistic techniques [15,16].

Second, Eq. (7) is independent of the material properties, such as the Young's modulus. To verify this, we modify the force field parameters of the CG model in order to generate different membrane systems. Specifically, the bond and angle stiffness parameters are changed accordingly, which linearly scale with the Young's modulus of hexagonal symmetry sheet [20,31]. The other two systems are those which have either half or double the Young's modulus of pristine graphene. By keeping the projectile size constant, we observe from the simulations that the reflected cone wave always returns to the surface of the projectile at the time when the projectile reaches zero velocity for all the three systems. The analytical relationship derived here is thus generally applicable to membranes.

In addition, we note that although the derived theory for the reflected wave speed and projectile halting time is general, membranes with circular shape have the most reflected wave effect since the reflected wave returns to the center at the same time and causes the greatest strain concentration when encountering the projectile surface. An experimentally relevant question that arises from this analysis is what the free-standing membrane geometry should be to reduce the cone wave reflection effects on the ballistic performance, and vice versa, in which kind of boundary condition we could observe the most obvious wave reflection effect. For this purpose, we have also tested triangular and square shapes for the suspended free-standing area. Specifically, in the same square sheet with edge length of 105 nm, we define circular shape with diameter equal to 100 nm (Fig. 5(a)–(b)), square (Fig. 5(c)–(d)) and triangular shape (Fig. 5(e)–(f)) with length of edge equal to 100 nm. We also show the cone wave reflection process in the Fig. 5 by coloring the beads per their out-of-plane positions, with red corresponding to lower position (further from the reader) and blue corresponding to higher position (closer to the reader). For circular shape, after the cone wave reflects from the fixed boundary simultaneously, the wave keeps the circular shape (Fig. 5(a)) and gets back to the center at the same time, which results in large strain

<span id="page-5-2"></span>**Fig. 5.** Cone wave reflection for free standing membrane with circular shape (a–b), square shape (c–d) and triangular shape (e–f). The beads outside of the freestanding region are fixed during the simulation. The beads are colored per their out-of-plane position, with red corresponding to lower (further from the reader) and blue corresponding to upper (closer to the reader). (For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)

<span id="page-5-1"></span>**Table 1** Verifying the scaling order of the density factor.

| ac (nm) | rp1 (nm) (projectile density 3.42 g/cm3<br>) | rp2 (nm) (projectile density 6.84 g/cm3<br>) | )1.5<br>(<br>rp1<br>log2<br>rp2 |
|---------|----------------------------------------------|----------------------------------------------|---------------------------------|
| 50      | 7.5                                          | 6                                            | 0.48                            |
| 100     | 12.5                                         | 10                                           | 0.48                            |
| 150     | 16                                           | 12.5                                         | 0.53                            |
| 200     | 21                                           | 16.5                                         | 0.52                            |

concentration and failure of the membrane [\(Fig. 5\(](#page-5-2)b)). However, the square shape membrane has much less cone wave reflection effect because the reflected cone waves from four edges interfere with each other [\(Fig. 5\(](#page-5-2)c)), while traveling back they only partially return to the center [\(Fig. 5\(](#page-5-2)d)). For the triangular shape membrane, we see a moderate cone wave reflection effect. The shorter distance to the edges [\(Fig. 5\(](#page-5-2)e)) and less time being required for the reflected wave to travel back give rise to a stronger effect. However, only part of the reflected wave returns to the midpoint at a given instance, which limits the wave reflection induced strains compared to a circular membrane setting. In summary, we conclude that the clamped circular membrane setting is the boundary condition that maximizes the reflected cone wave effects among the geometries studied here.

## **4. Conclusions**

The cone wave developed in the wake of impact upon membrane materials dissipates the kinetic energy of the projectile by propagating away from the impact zone on one hand, but on the other hand, when it has enough time to return from the fixed boundary, it reduces the projectile proof capability of the membrane. To the best of our knowledge, for the first time, we observe and study the cone wave reflection induced size effect of graphene membrane under ballistic impact through simulations and theoretical models. The proposed theoretical analysis fills the gap for finite membrane behavior during projectile impact process. The critical relationship between the membrane size and the projectile size is found to depend only upon the density ratio of the projectile and the membrane and the membrane thickness. The study reveals that in the microballistic experiments, when the sample is ultra-thin and the lateral sample span is small, the reflected wave would play its own effect on the failure of samples, and that its effect would be maximum for circular free-standing membrane geometries. Future microballistic experiments could shed light on the behavior of different thin membrane materials including polycrystalline systems with grain boundaries. The critical size relationship obtained here provides guidance for future ballistic barrier geometric designs.

#### **Acknowledgments**

The authors acknowledge funding by the Army Research Office (award # W911NF-13-1-0241). The authors also acknowledge the support from the Departments of Civil and Environmental Engineering and Mechanical Engineering at Northwestern University, as well as the Northwestern University High Performance Computing Center for a supercomputing grant.

## <span id="page-5-0"></span>**Appendix A. Derivation of the reflected wave speed and cone wave travel time**

From Phoenix and Porwal's work [\[9\]](#page-7-8), the ratio of the speed of cone wave v*c*<sup>0</sup> to that of the tensile wave *c*<sup>0</sup> = √ *E*/ρ is related to the membrane strain at the wave front ε *c* as:

<span id="page-5-3"></span>
$$v_{c0}/c_0 \approx \sqrt{\varepsilon_c^0}$$
. (A.1)

In the following analysis, the speed of the expanding cone wave  $v_{c0}$  is treated as a constant. This approximation is reasonable as it has been verified in a previous study that the speed increases sharply before traveling a short distance smaller than  $2r_p$ , and then stays constant afterwards [9].

When the cone wave reaches the boundary (we denote this time as  $t_1$ ), the membrane is already in the deformed state, and the inplane strain  $\varepsilon_t$  is a function of the radial distance r from the center, as illustrated by the black line in Fig. A.1(a).  $\varepsilon_t$  can be obtained from the Ref. [9],

<span id="page-6-1"></span>
$$\varepsilon_t(r; t = t_1) \approx \varepsilon_c^0 \frac{a}{r},$$
 (A.2)

where *a* is the radius of the membrane.

We verify Eq. (A.2) with the help of the MD simulation data, as shown in Fig. A.1(b). A typical case has been chosen with the membrane radius a = 100 nm and the projectile radius  $r_p =$ 12.5 nm to calculate the strain. Since we adopt spherical projectile shape and circular membrane shape, therefore, the strain is axisymmetric. We pick a strip region along radial direction ( $\theta = 0$ ) from r = 12.5 nm (projectile radius) to r = 100 nm (circular membrane radius), with width of 0.5 nm. We obtain N=212atoms for this specific case. We store all coordinates of the Natoms,  $(x_{0_i}, y_{0_i}, z_{0_i}), i = 1, 2, ..., N$ , at equilibrium configuration at time t = 0, and calculate the radial coordinates,  $r_{0i} =$  $\sqrt{x_{0_i}^2+y_{0_i}^2}.$  At time  $t_1$ , we store the new coordinates of all these atoms,  $(x_i, y_i, z_i)$ , i = 1, 2, ..., N. The displacements along in-plane direction is calculated as  $u_i = x_i - x_{0i}$  (for this particular radial direction) and normal, out-of-plane direction is calculated as  $v_i =$  $z_i - z_{0i}$ . We then calculate the local strains,  $\varepsilon_{ti}$ , i = 1, 2, ..., N-1, according to Phoenix and Porwal's work [9]:

$$\varepsilon_{t_i} = \sqrt{\left(1 + \frac{\partial u}{\partial r}\right)^2 + \left(\frac{\partial v}{\partial r}\right)^2} - 1,$$
 (A.3)

where  $\frac{\partial u}{\partial r} \approx \frac{u_{i+1} - u_i}{r_{0i+1} - r_{0i}}$  and  $\frac{\partial v}{\partial r} \approx \frac{v_{i+1} - v_i}{r_{0i+1} - r_{0i}}$  for  $i = 1, 2, \ldots, N-1$ . The axisymmetric strain has been verified by choosing atoms

The axisymmetric strain has been verified by choosing atoms along different  $\theta$  where we find negligible differences in the final strain results.

The basic assumption is that at time  $t>t_1$ , when the reflected wave is traveling back, the incremental deformation caused by the projectile stretching the membrane further down is much smaller than the combination of the current strain and the additional strain caused by the reflected wave. The additional strain caused by the wave front of the reflected wave is still  $\varepsilon_c^0$  given that there is no energy dissipation during the reflection process. The total strain at the wave front of the reflected wave can be written as:

$$\varepsilon_c(r) \approx \varepsilon_c^0 \left( 1 + \frac{a}{r} \right).$$
(A.4)

Then the reflected wave speed is expressed in a similar fashion to Eq. (A.1) as:

<span id="page-6-2"></span>
$$v_c(r) = c_0 \sqrt{\varepsilon_c(r)} = c_0 \sqrt{\varepsilon_c^0 \left(1 + \frac{a}{r}\right)}.$$
 (A.5)

Comparing Eqs. (A.1) and (A.5), we clearly see that the speed of reflected wave is greater than that of the initial expanding cone wave, and it gradually increases while traveling back.

The time for the cone wave to first reach boundary from the projectile surface is:

$$t_1 = (a - r_p)/v_{c0} = (a - r_p)/(c_0\sqrt{\varepsilon_c^0}).$$
 (A.6)

The time for reflected wave traveling back to the surface of the projectile is:

<span id="page-6-3"></span>
$$t_{2} = \int_{r=r_{p}}^{r=a} 1/v_{c}(r) dr = \int_{r=r_{p}}^{r=a} 1/\left(c_{0}\sqrt{\varepsilon_{c}^{0}\left(1+\frac{a}{r}\right)}\right) dr.$$
 (A.7)

<span id="page-6-0"></span>**Fig. A.1.** (a) Illustration of the axisymmetric system. The back curve is the membrane shape produced by the cone wave at time  $t=t_1$  and the blue curve shows the extra deformation caused by the reflected wave. The blue dashed line shows the region around the reflected wave front which must match continuously with the black curve. (b) The local in-plane strain  $\varepsilon_t$  at time  $t=t_1$  as a function of radial distance r. Red solid and blue dashed lines correspond to MD and analytical results, respectively. (For interpretation of the references to colour in this figure legend, the reader is referred to the web version of this article.)

Further simplifying Eq. (A.7) gives the following:

<span id="page-6-4"></span>
$$t_2 = a/(c_0 \sqrt{\varepsilon_c^0}) \int_{r_0/a}^1 \frac{dx}{\sqrt{1+1/x}}.$$
 (A.8)

Solving Eq. (A.8),  $t_2$  is directly related to  $t_1$  by  $t_2 = Ct_1$ , where C lies in the range of 0.54–0.57 given  $r_p$  in the order of 0.01 to 0.1 of the membrane radius a for critical size relationship between them. The MD simulation results reveal C in the range of 0.6–0.68. The difference can be attributed to the simplified assumptions for the wave speed made in the analysis. The C value corresponding to the case of Fig. 3(a) is 0.63.

In the following analysis, we just use constant C=0.56 according to the theoretical solution. Moreover, explicit sensitivity analysis shows that the constant C in the range from 0.5 to 0.7 only contributes to less than 3% difference in the resulting reactive force and the projectile halting time  $t_p$ . Therefore, our conclusions are not sensitive to the constant value of C chosen here.

As a result, the total time for the cone wave to return to the surface of the projectile is

$$t_c = t_1 + t_2 = 1.56t_1 \cong 0.78 \frac{2a_c}{v_{c0}}.$$
 (A.9)

## Appendix B. Supplementary data

Supplementary material related to this article can be found online at http://dx.doi.org/10.1016/j.eml.2017.06.001.

## **References**

- <span id="page-7-0"></span>[1] [W.R. Sanhai, J.H. Sakamoto, R. Canady, M. Ferrari, Nature Nanotechnol. 3 \(2008\)](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb1) [242.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb1)
- <span id="page-7-1"></span>[2] [F. Soto, A. Martin, S. Ibsen, M. Vaidyanathan, V. Garcia-Gradilla, Y. Levin,](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb2) [A. Escarpa, S.C. Esener, J. Wang, ACS Nano 10 \(2015\) 1522.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb2)
- <span id="page-7-2"></span>[3] [D.A. Shockey, J.W. Simons, D.R. Curran, Int. J. Appl. Ceram. Technol. 7 \(2010\)](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb3) [566.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb3)
- <span id="page-7-3"></span>[4] [E. Grossman, I. Gouzman, R. Verker, MRS Bull. 35 \(2010\) 41.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb4)
- <span id="page-7-4"></span>[5] [E.D. Wetzel, R. Balu, T.D. Beaudet, J. Mech. Phys. Solids 82 \(2015\) 23.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb5)
- <span id="page-7-5"></span>[6] [P.M. Cunniff, Text. Res. J. 66 \(1996\) 45.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb6)
- <span id="page-7-6"></span>[7] [M. Chen, J.W. McCauley, K.J. Hemker, Science 299 \(2003\) 1563.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb7)
- <span id="page-7-7"></span>[8] [E.L. Thomas, Opportunities in Protection Materials Science and Technology for](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb8) [Future Army Applications, Wiley Online Library, 2011.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb8)
- <span id="page-7-8"></span>[9] [S.L. Phoenix, P.K. Porwal, Int. J. Solids Struct. 40 \(2003\) 6723.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb9)
- [10] [P.M. Cunniff, Text. Res. J. 62 \(1992\) 495.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb10)
- [11] [P.M. Cunniff, Text. Res. J. 66 \(1996\) 45.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb11)
- [12] [B.A. Cheeseman, T.A. Bogetti, Compos. Struct. 61 \(2003\) 161.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb12)
- [13] [C. Lim, V. Shim, Y. Ng, Int. J. Impact Eng. 28 \(2003\) 13.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb13)
- <span id="page-7-9"></span>[14] [J. Vinson, J. Walker, AIAA J. 35 \(1997\) 875.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb14)
- <span id="page-7-10"></span>[\[](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb15)15] [J.-H. Lee, D. Veysset, J.P. Singer, M. Retsch, G. Saini, T. Pezeril, K.A. Nelson,](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb15) [E.L. Thomas, Nature Commun. 3 \(2012\) 1164.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb15)
- <span id="page-7-11"></span>[16] [J.-H. Lee, P.E. Loya, J. Lou, E.L. Thomas, Science 346 \(2014\) 1092.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb16)

- <span id="page-7-12"></span>[17] [W. Xia, J. Song, Z. Meng, C. Shao, S. Keten, Mol. Syst. Des. Eng. 1 \(2016\) 40.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb17)
- <span id="page-7-24"></span>[18] [W. Xia, L. Ruiz, N.M. Pugno, S. Keten, Nanoscale 8 \(2016\) 6456.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb18)
- [19] [C. Shao, S. Keten, Sci. Rep. 5 \(2015\) 16452.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb19)
- <span id="page-7-13"></span>[\[](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb20)20] [Z. Meng, R.A. Soler-Crespo, W. Xia, W. Gao, L. Ruiz, H.D. Espinosa, S. Keten,](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb20) [Carbon 117 \(2017\) 476.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb20)
- <span id="page-7-14"></span>[21] [C. Lee, X. Wei, J.W. Kysar, J. Hone, Science 321 \(2008\) 385.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb21)
- [\[](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb22)22] [C. Lee, X.D. Wei, Q.Y. Li, R. Carpick, J.W. Kysar, J. Hone, Phys. Status Solidi B 246](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb22) [\(2009\) 2562.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb22)
- <span id="page-7-15"></span>[\[](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb23)23] [X. Wei, Z. Meng, L. Ruiz, W. Xia, C. Lee, J.W. Kysar, J.C. Hone, S. Keten, H.D.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb23) [Espinosa, ACS Nano 10 \(2016\) 1820.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb23)
- <span id="page-7-16"></span>[24] [B. Z.G. Haque, S.C. Chowdhury, J.W. Gillespie, Carbon 102 \(2016\) 126.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb24)
- <span id="page-7-20"></span>[25] [K. Xia, H. Zhan, D.a. Hu, Y. Gu, Sci. Rep. 6 \(2016\).](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb25)
- <span id="page-7-17"></span>[26] [K. Yoon, A. Ostadhossein, A.C. van Duin, Carbon 99 \(2016\) 58.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb26)
- <span id="page-7-18"></span>[\[](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb27)27] [Z. Meng, M.A. Bessa, W. Xia, W. Kam Liu, S. Keten, Macromolecules 49 \(2016\)](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb27) [9474.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb27)
- [28] [S.J. Stuart, A.B. Tutein, J.A. Harrison, J. Chem. Phys. 112 \(2000\) 6472.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb28)
- [\[](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb29)29] [D.W. Brenner, O.A. Shenderova, J.A. Harrison, S.J. Stuart, B. Ni, S.B. Sinnott, J.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb29) [Phys.: Condens. Matter 14 \(2002\) 783.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb29)
- <span id="page-7-19"></span>[\[](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb30)30] [A.C. Van Duin, S. Dasgupta, F. Lorant, W.A. Goddard, J. Phys. Chem. A 105 \(2001\)](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb30) [9396.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb30)
- <span id="page-7-21"></span>[31] [L. Ruiz, W. Xia, Z. Meng, S. Keten, Carbon 82 \(2015\) 103.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb31)
- <span id="page-7-22"></span>[32] [S. Plimpton, J. Comput. Phys. 117 \(1995\) 1.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb32)
- <span id="page-7-23"></span>[33] [W. Humphrey, A. Dalke, K. Schulten, J. Mol. Graph. 14 \(1996\) 33.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb33)
- <span id="page-7-25"></span>[34] [A.F. Wells, Structural Inorganic Chemistry, Oxford University Press, 2012.](http://refhub.elsevier.com/S2352-4316(17)30056-1/sb34)