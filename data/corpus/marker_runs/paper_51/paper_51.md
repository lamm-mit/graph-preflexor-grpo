www.acsami.org Research Article

# The Projectile Perforation Resistance of Materials: Scaling the Impact Resistance of Thin Films to Macroscale Materials

Katherine M. Evans, Shawn H. Chen, Amanda J. Souna, Stephan J. Stranick, Christopher L. Soles, and Edwin P. Chan\*

Cite This: ACS Appl. Mater. Interfaces 2023, 15, 32916–32925

**ACCESS** I

III Metrics & More

Article Recommendations

Supporting Information

ABSTRACT: From drug delivery to ballistic impact, the ability to control or mitigate the puncture of a fast-moving projectile through a material is critical. While puncture is a common occurrence, which can span many orders of magnitude in the size, speed, and energy of the projectile, there remains a need to connect our understanding of the perforation resistance of materials at the nano- and microscale to the actual behavior at the macroscale that is relevant for engineering applications. In this article, we address this challenge by combining a new dimensional analysis scheme with experimental data from micro- and macroscale impact tests to develop a relationship that connects the size-scale effects and materials properties during high-speed puncture events. By relating the minimum perforation velocity to fundamental material properties and geometric test conditions, we provide new insights and establish an alternative methodology for

evaluating the performance of materials that is independent of the impact energy or the specific projectile puncture experiment type. Finally, we demonstrate the utility of this approach by assessing the relevance of novel materials, such as nanocomposites and graphene for real-world impact applications.

KEYWORDS: impact mitigation, puncture mechanics, microprojectile impact test, polymeric materials, thin films, mechanical properties

#### 1. INTRODUCTION

There are many instances (Figure 1a-d) where mitigating or enhancing puncture by a fast-moving projectile into a material is necessary. A jellyfish stings their prey at up to 37 m/s from stingers that are no more than 30 nm in diameter, while snake bites range from 1 to 3 m/s from teeth that are 100s of  $\mu$ m in diameter. In drug delivery, a needle-free liquid jet punctures skin at speeds from 80 to 200 m/s with streams on the order of 100  $\mu$ m in diameter.<sup>2</sup> For impact protection, armor must stop mm to cm projectiles that travel upward of 1000 m/s.<sup>3</sup> In space exploration, the protective coatings must withstand a barrage of micrometeorites (space dust) that are just a few microns in diameter but move at speeds up to 10 km/s.<sup>4</sup> As illustrated in Figure 1e, these events span many orders of magnitude in dimensions, puncture velocity, and impact energy (from nJ to kJ) without a clear correlation, even though they are all projectile puncture events.

Traditionally the perforation resistance of a material is evaluated using large-scale or bulk samples. These macroscale projectiles impart large amounts of kinetic energy, often reaching hundreds of joules, and therefore require a large volume of sample to perform the tests. While these studies experimentally mimic real-life impact scenarios, they are prohibitively time-consuming, labor-intensive, and empirical. They also fail to meet many of the impact conditions presented

in Figure 1e. Microscale analogs, such as Laser-Induced Projectile Impact Test (LIPIT), 5,6 are an exciting new alternative that can replicate many of these impact conditions. They also require tiny amounts of material and can be conducted quickly. These LIPIT tests are beneficial for studying relatively thin samples (sub- $\mu$ m in thickness) as they use microprojectiles to impact or puncture a material. In a typical LIPIT experiment, a microprojectile is launched via laser ablation into a thin film of interest. Impact velocities on the order of 10 to 1000 m/s are routinely achieved, comparable to extreme macroscale ballistic impact measurements. Even though the projectiles in LIPIT can reach significant impact velocities, as seen in Figure 1e, the microprojectiles impart orders of magnitude smaller amounts of kinetic energy to the sample, often on the order of nano- to microjoules. While both macroscale impact tests and LIPIT assess the perforation resistance of materials, the fundamental question remains whether these different testing approaches

Received: April 10, 2023 Accepted: June 8, 2023 Published: June 29, 2023

<span id="page-1-0"></span>**Figure 1.** Examples of projectile penetration events over a broad range of velocity and size-scales. Experimental studies of various projectile impact studies such as (a) entry of a sphere into a viscoelastic fluid (reproduced from ref 38 with permission from Elsevier), (b) a macroprojectile impact testing of a polycarbonate sheet (reproduced from ref 18 with permission from Elsevier), (c) a microscale laser-induced projectile impact testing (LIPIT) of a polycarbonate thin film (reproduced from ref 11 with permission from Royal Society of Chemistry), and (d) discharge of a nematocyte by a jellyfish (reproduced from ref 1 with permission from Elsevier). (e) Comparing the puncture velocity vs projectile size for various impact events 1,2,11,14–19,39–51 illustrating the vast differences in velocity and size.

are even related or relevant given these vastly different size and energy scales. Practitioners of macroscale ballistic tests are dubious of the relevancy of the LIPIT technique. Researchers interested in microscale impact phenomena rarely look to bulk studies to understand their systems. Materials development can be significantly enhanced if these two communities can be connected.

In this work, we demonstrate for the first time that the minimum perforation velocity  $(V_0)$ , defined as the maximum projectile velocity that a material can arrest without catastrophic failure, can be compared across vastly different energy and size scales of an impact event. We then take the crucial next step to show, using a Buckingham  $\Pi$  analysis, that these  $V_0$  values, regardless of whether they were determined using micro- or macroscale impact studies, can be related through sample geometry and fundamental materials properties. We combine microprojectile impact experiments, literature results, and Buckingham  $\Pi$  dimensional analysis to establish the relationship between  $V_0$  and the size-scale of the impact experiment and materials properties that span over many orders of magnitude. The framework developed here

shows that the physics and mechanics that underpin these puncture events share similarities, a result that would not be otherwise evident from an empirical measurement. Our approach allows for direct comparison of  $V_0$  across different types of impact tests even when the impact energy, projectile size, and film thickness can vary by many orders of magnitude.

#### 2. RESULTS AND DISCUSSION

2.1. Microprojectile Impact Studies. Figure 2a shows a schematic of the LIPIT experiment. A microprojectile of radius  $a_v$  is launched via laser ablation from a substrate and perforates the material of interest. By imaging the impact event with a stroboscopic camera setup, the impact velocity  $(v_i)$  and residual velocity  $(v_r)$  of the microprojectile are measured before and after the penetration event. These velocities are used to calculate the energy dissipation of the film during the impact event based on a kinetic energy balance of an inelastic collision. The specific penetration energy,  $E_p^* = (v_i^2 - v_r^2)/m_{\text{plug}}$ defined as the kinetic energy loss normalized by the mass of the plug of film ejected by the projectile strike  $(m_{\text{plug}})$ , is reported as a measure of the puncture resistance of the material. However, measuring  $m_{\text{plug}}$  is nontrivial; it is usually approximated, making  $E_v^*$  less accurate in comparing the puncture resistance of thin films.8 Here we propose an alternative parameter to relate better the film's energy dissipated to its properties.

For the impact experiments, we analyze the perforation resistance of polymer-grafted nanoparticle polymethacrylate (npPMA) composite films via LIPIT. These nanocomposites consist of silica nanoparticles with covalently grafted polymethacrylate (PMA) chains. Similar to our previous study,9 we investigate the influence of the molecular mass of the PMA chains on the perforation resistance of these nanocomposites using LIPIT. For all the npPMA systems, films with a nominal film thickness  $h \approx 400$  nm were prepared. Details of the npPMA film preparation procedure are provided in the Materials and Methods section. In this study, we present new data exploring the effects of projectile size on perforation resistance, using silica or glass microprojectiles with radii from  $a_{\nu} \approx 3.8$  to 14  $\mu$ m and impact velocities  $v_i = 100$  to 400 m/s. The kinetic energy change due to this inelastic impact event can be determined by relating the square of the residual velocity of the microprojectile after impact  $(v_r^2)$  against the square of the impact velocity  $(v_i^2)$ , which can be defined as,

$$v_r^2 = \alpha v_i^2 - \gamma \tag{1}$$

We note that eq 1 is the kinetic loss expression of an inelastic collision event that is normalized by the total mass of the system, an expression that is similar to the Recht-Ipson model based on momentum and energy conservation concepts. 10 The parameters  $\alpha$  and  $\gamma$  are not empirical fitting constants but relate directly to the scale of the impact and the amount of energy dissipated for the size of the system, respectively. From energy and momentum conservation, Chen and co-workers showed that the mass of the system participating in the impact is defined by  $\alpha = m_p/(m_p + m_t)^{.11}$  This parameter represents the projectile's mass relative to the system's mass participating in the impact. It is the sum of the mass of the projectile  $(m_n)$ and the mass or amount of the target material  $(m_t)$  that experiences deformation during impact, which can be significantly greater than  $m_{\text{plug}}$ . As discussed in the Supporting Information,  $m_t$  can be estimated from the acoustic wave speed

<span id="page-2-0"></span>Figure 2. Microprojectile perforation experiments of the npPMA films using LIPIT. (a) Schematic of the LIPIT experiment illustrating the propulsion of a microprojectile via laser ablation. The microprojectile impacts the polymer film at an impact velocity  $v_p$  perforates it, and exits the film with a residual velocity  $v_r$ . (b) The microprojectile velocities are measured using stroboscopic imaging as shown by the optical micrograph showing a microprojectile perforating a 55 kDa npPMA film. Scalebar is 25  $\mu$ m. Representative plot of  $v_r^2$  vs  $v_i^2$  for a  $a_p = 10$   $\mu$ m projectile perforating a 55 kDa npPMA film. (c) Minimum perforation velocity ( $V_0$ ) as a function of microprojectile size and PMA molecular mass (M). All the curves correspond to the scaling relationship:  $V_0 \sim a_p^{-3/4}$ . (d)  $V_0$  versus M for three different projectile radii.

of the material and impact velocity. The parameter  $\gamma = E_d/(m_p + m_t)$  is the energy dissipation  $(E_d)$  of the film normalized by the system's mass. Both  $\alpha$  and  $\gamma$  suggest that the mass of the projectile and the portion of the film affected by the impact event are important factors that contribute to the perforation resistance of a material.

**2.2. Minimum Perforation Velocity.** The perforation resistance of a material can be defined by posing the following scenario. What is the critical impact velocity when the material completely arrests a projectile, i.e., when  $v_r = 0$ ? We use eq 1 to define this critical point as the minimum perforation velocity, expressed as

$$V_0 = \left(\frac{\gamma}{\alpha}\right)^{0.5} \tag{2}$$

As shown in Figure 2b, the minimum perforation velocity for the 55 kDa npPMA film is determined from the plot of  $v_r^2$  vs  $v_i^2$ , showing a linear relationship between the two quantities. The parameters  $\alpha$  and  $\gamma$  correspond to the slope and intercept of this curve, respectively. We note that this linear correlation between  $v_r^2$  and  $v_i^2$  is observed for all of the impact data sets studied here and is consistent with eq 1.

We use the minimum perforation velocity as a parameter to characterize the perforation resistance of the material due to the projectile impact of a given size. Compared to the traditional descriptor of performance,  $V_0$  offers a useful alternative to  $E_p^*$  as it enables comparison of perforation resistance of materials across a broad range of impact energies.

 $V_0$  defines a critical point at which a material can withstand an impact. On the other hand, it is challenging to use  $E_n^*$  to compare different materials and impact conditions as it is a parameter that can vary significantly depending on the impact velocity and the accurate estimate of the mass of material that contributes to the impact. A material can withstand perforation when  $v_i \leq V_0$ , whereas complete penetration will occur when  $v_i$  $> V_0$ . A material with a higher  $V_0$  is thus more perforation resistant and can absorb/dissipate more energy before failing catastrophically. The parameters  $\alpha$  and  $\gamma$  are physical quantities related to the scale and energy dissipated from the impact and can be extracted directly from projectile impact tests in instances where  $v_i$  and  $v_r$  are measured, thus enabling  $V_0$  to be determined straightforwardly even in instances where the exact failure mechanism is an unknown. Using this approach,  $V_0$  can easily be determined, which is beneficial when these values are compared across materials and experiment types. As projectile penetration events can span an expansive range of scales and velocities, it is important to understand how  $V_0$  changes across different size scales and materials systems. Thus a deeper understanding of the governing parameters that define  $V_0$  must be established.

We use LIPIT to study the penetration resistance of five different molecular mass npPMA systems as a function of projectile size and determine the  $\alpha$  and  $\gamma$  parameters for each system to determine  $V_0$  as a function of both the projectile size and molecular mass of the polymer. As shown in Figure 2c, a common trend pervades in all five systems in that there is a direct correlation between  $V_0$  and the radius of the projectile

<span id="page-3-0"></span> $a_p;\ V_0$  increases as we reduce  $a_p,$  which makes intuitive sense. For a given film thickness and impact velocity, a smaller projectile will reduce the impact energy of the penetration event (kinetic energy  $\sim 4\pi\rho_p a_p^3 v_i^2/3$ ), which in turn means that the minimum perforation velocity must increase to compensate for the reduced energy. From this figure, we empirically find that  $V_0$  scales as  $V_0 \sim a_p^{-3/4}$ . As we discuss below, this scaling relationship is not simply a function of the magnitude of the impact energy alone but also reflects the size-dependent changes in the mechanical properties of the material.

Looking at the impact results as a function of projectile size in detail (Figure 2d), we see a nonmonotonic change in  $V_0$  as a function of molecular mass of the PMA graft for a given projectile size. We consistently see a peak in the minimum perforation velocity for the 96 kDa npPMA system compared with those of the other molecular mass grafts. A similar trend in perforation resistance was reported earlier in terms of the specific puncture energy  $(E_p^*)$ , which was attributed to enhanced polymer dynamics and molecular dissipation at this critical molecular mass. As we show below, either  $V_0$  or  $E_p^*$ can be used to characterize the perforation resistance of impact-mitigating materials. However, the critical distinction for  $V_0$  lies in its utility to compare the impact performance of materials across different size and energy scales, which is particularly useful when predicting the performance of materials at the macroscale by leveraging impact results on a much smaller scale.

2.3. Connecting Different Energy Scales. While npPMAs are fascinating nanomaterials with an unusual impact response at the microscale, they are novel materials and tedious to synthesize. It would be a significant challenge to produce sufficient quantities of these materials for macroscopic impact tests. Our LIPIT results alone are not strong enough motivation for us to undertake this challenge. The same is true for the graphene films. There have been enticing LIPIT studies on either modeling single graphene 12 or experimentally measuring multilayer graphene<sup>7,13</sup> suggesting that these 2D nanomaterials also have excellent impact resistance. There is a critical need to understand whether these microscale measurements are relevant to bulk materials. In an attempt to establish this relevance, we turn to scaling relationships and augment our analysis to include polycarbonate (PC), a classic example of bulletproof glass which has been studied extensively using a variety of projectile tests at different scales ranging from the macroscale gas guns 14-19 to the microscale LIPIT studies. Taking the PC results from the literature for gas gun and LIPIT studies, the impact and residual velocity data set from each study were extracted and fit using eq 1 to obtain the respective  $\alpha$  and  $\gamma$  parameters. A summary of these parameters, along with the other relevant materials and geometric parameters, is presented in the Supporting Information. Using the  $\alpha$  and  $\gamma$  values from each study, we calculate the expected residual impact energy (=  $\alpha v_i^2 - \gamma$ ), i.e., eq 1, as a function of  $v_i$ . As shown in Figure 3a, it is striking that all data sets collapse onto a master curve without any adjustable parameters when comparing these energy values against the reported  $v_r^2$  values. The greater variation of the h/a = 0.005data set from the master curve is because these are the thinnest films studied with LIPIT, which result in greater variation in residual velocity.11

This master curve is striking and insightful in several ways. First, the perforation data sets were obtained from different projectile test methods on vastly different size and energy

Figure 3. (a) Master curve of residual kinetic energy, defined as  $v_r^2$ , versus kinetic loss for an inelastic collision  $(\alpha v_i^2 - \gamma)$  defined in eq 1 for PC samples over a broad range of thickness (h) and projectile size  $(a_p)$  from different projectile tests. Data of the impact and residual velocities from gas gun studies are obtained from,  $^{14-19}$  and from LIPIT studies are obtained from. (b) The minimum perforation velocity  $(V_0)$  for PC vs the size scale ratio  $(h/a_p)$  illustrates that  $V_0$  increases with increasing  $h/a_p$ . The purple diamond symbols correspond to data from Wright et al. Where  $V_{50}$  was reported but  $V_0$  cannot be determined.

scales from nJ to KJ (see Supporting Information). Yet, by extracting the  $\alpha$  and  $\gamma$  parameters, we can collapse all of the results onto a single master curve. This suggests that the kinetic energy loss expression described in eq 1 is appropriate in parametrizing the penetration performance of these materials under vastly different experimental conditions; the microscale tests are correlated with their macroscale analogs. Second, we see that the residual kinetic energy,  $v_r^2$ , increases with the scale of the test as the energy from a gas gun experiment is several orders of magnitude greater than the LIPIT results. This scale can be articulated by a geometric ratio, which we define as the ratio of the material thickness and the projectile radius  $(h/a_n)$ . In LIPIT experiments, the films are typically tens to hundreds of nanometers. In contrast, materials used for gas gun experiments can be a few centimeters thick, with a six-order magnitude difference in length scale. The projectiles used for different methods can vary between tens of micrometers to millimeters, spanning three orders of magnitude. Importantly, this master curve demonstrates the applicability of this energy balance approach in understanding projectile impact testing across different experiments and shows that there is a common link from studies taken on vastly different size and energy scales.

Next, we determined  $V_0$  for each system using the  $\alpha$  and  $\gamma$  parameters determined from each PC data set. By comparing  $V_0$  with  $h/a_p$  (Figure 3b), we find that the minimum perforation velocity increases with the thickness of a given PC material and projectile size. This is consistent with one's

<span id="page-4-0"></span>Figure 4. Predicting the perforation resistance of materials. (a) Buckingham Π dimensional analysis results of the PC samples relating the first Π term to the second and third Π terms. (b) Scaling relationship for  $V_0$  as a function of geometric and materials properties for all the materials investigated. The *y*-axis is the measured  $V_0$ , and the *x*-axis is the predicted  $V_0$  based on the geometric and material properties for the particular impact test. Note that PC<sup>‡</sup> corresponds to PC systems where the failure stress ( $\sigma_f$ ) is reported and  $\overline{\sigma}_f = \sigma_f/\rho_t$  with units of M²/L². (c) Failure stress as a function of projectile size ( $a_p$ ) for npPMA, PC, PS, and graphene (SLG and MLG). (d) Failure stress as a function of the approximate deformation rate,  $\dot{\varepsilon} = V_0/2a_p$ , for npPMA, PC, PS, and graphene.

intuition: if the material is infinitely thick, it should theoretically be able to arrest a projectile of any size over a broad range of impact velocities. Conversely, if the material is infinitely thin, any projectile can easily perforate the material even if it is intrinsically tough, such as PC films. Empirically, we observe that  $V_0$  scales with  $h/a_p$  with a power-law exponent of 0.4, suggesting an intimate relationship between the geometry of the impact and perforation resistance. While we can use Figure 3b to estimate the minimum perforation velocity at a given size scale ratio, it provides limited insight into how materials properties and geometric test conditions couple to define the perforation resistance of a material. Intuition and experience might suggest, for example, that a metal projectile will readily penetrate a given material, whereas an elastomeric projectile would not; or materials like PC are superior at arresting projectile perforation, whereas brittle materials such as polystyrene (PS) are not.

The complex interplay between size scale and material properties is a classic problem. Many models have been developed to understand how materials can be engineered to be more penetration-resistant. Classical mechanical models of penetration, such as the hydrodynamic model<sup>20</sup> or the Poncelet model,<sup>21</sup> have been developed to understand how impact velocity determines penetration in semi-infinite materials (see bottom right inset in Figure 4a). Such models are valuable guides for connecting materials and experimental parameters to the real-world problem of designing materials for penetration resistance. The Poncelet model was previously applied to LIPIT studies to predict the penetration depth,  $\delta_{v}$  into thick polymer gels.<sup>22</sup> However, we are interested in a

slightly different problem: the conditions of the impact test and the materials properties that determine the onset of complete penetration through a film of finite dimensions.

We turn to the Buckingham  $\Pi$  analysis to establish such a relationship. Buckingham  $\Pi$  is a form of dimensional analysis that identifies the critical nondimensional parameters for a given experiment, even if the exact form of the constitutive expression is unknown. Another appealing aspect of Buckingham  $\Pi$  is that it can develop physical relationships that connect across different energy or size scales, which is especially applicable for this study and has been applied in other microprojectile studies due to the self-similarity of the problem.

The details of the Buckingham  $\Pi$  analysis and the derivation of the relationships can be found in the Supporting Information section. In brief, we are interested in relating the minimum perforation velocity to the relevant physical parameters associated with a projectile impact test of a finite material. We note that the Buckingham  $\Pi$  analysis can generate additional terms with other nondimensional parameters, but from our preceding discussions on the different penetration models, we know that  $V_0$  is affected by the following set of material properties or geometric parameters of the impact test. We define  $V_0 = V_0(a_p, h, \rho_p, \rho_v$  and  $\sigma_f$ ) where the parameters, along with their physical dimensions, are summarized in Table 1. Following the Buckingham  $\Pi$  methodology, we define three dimensionless  $\Pi$  groups related to the parameters defined in Table 1,

<span id="page-5-0"></span>Table 1. Geometric and Materials Properties Used to Determine Eq  $3^a$ 

| Parameter      | Description                  | Physical dimensions |
|----------------|------------------------------|---------------------|
| $V_0$          | minimum perforation velocity | [L/T]               |
| $a_p$          | projectile radius            | [L]                 |
| h              | material thickness           | [L]                 |
| $\rho_p$       | projectile density           | $[M/L^3]$           |
| $\rho_t$       | material density             | $[M/L^3]$           |
| $\sigma_{\!f}$ | failure stress of material   | $[M/LT^2]$          |
| a              |                              | 1                   |

<sup>a</sup>The dimensions are L = length, M = mass, and T = time.

$$\Pi_{1} = \frac{\rho_{t}V_{0}^{2}}{\sigma_{f}}$$

$$\Pi_{2} = \frac{h}{a_{p}}$$

$$\Pi_{3} = \frac{\rho_{t}}{\rho_{p}}$$

$$\Pi_{1} = c_{0}\Pi_{2}\Pi_{3}$$
(3)

where  $c_0$  is a proportionality constant. Physically,  $\Pi_1$  represents the ratio of the impact energy  $(V_0^2)$  versus the specific impact resistance of the target material  $(\sigma_f/\rho_t)$ .  $\Pi_2$  and  $\Pi_3$  represent the geometry and mass of the impact test system, respectively. By analyzing other penetration models and evaluating the PC data shown in the Supporting Information, we determine that  $\Pi_1 = c_0\Pi_2\Pi_3$ . By correlating  $\Pi_1$  with  $\Pi_2\Pi_3$  using the PC data (Figure 4a), we find excellent agreement with eq 3 thus confirming the validity of our analysis approach.

While the  $\Pi$  groups are insightful in identifying the relevant experimental parameters for our impact studies, it is more useful to rearrange eq 3 such that we can relate  $V_0$  directly to the relevant materials and geometric properties. Specifically,  $V_0$  is defined as

$$V_0 = c_1 \left( \overline{\sigma}_f \frac{\rho_t}{\rho_p} \right)^{0.5} \left( \frac{h}{a_p} \right)^{0.5} \tag{4}$$

where  $c_1 \approx 1.62$  is an empirically defined constant (Figure 4a). We note that for the PC systems used to develop eq 4, we use the yield stress values  $(\sigma_Y)$  provided by the respective literature reference as a surrogate for  $\sigma_f$  because the failure stress values of the material were not provided. For most materials, we note that the failure stress and yield stress are not equivalent, and the failure stress is the key parameter used to quantify the penetration resistance in eq 4. However, we note that for PC, the failure stress has been shown via experiments and simulations to be comparable to the yield stress, albeit at a different critical strain, at sufficiently high deformation rates.  $^{26,27}$ 

We apply eq 4 to predict the properties for other materials studied from different projectile tests, where the failure stresses are unknown, but all the other parameters defined in Table 1 are reported. The first system is the npPMAs that motivated this work. Many of their properties, including failure stress, have not been measured. However, we can estimate the failure stress of the npPMA systems from our LIPIT data by collapsing the  $V_0$  results in eq 4. Figure 4b shows the resulting estimates of the npPMA films from the LIPIT results. We have

also leveraged eq 4 to estimate the failure stress of polystyrene (PS) films from experimental LIPIT studies, <sup>28</sup> single graphene (SLG) from modeling of LIPIT experiment, <sup>12</sup> and multilayer graphene (MLG) from experimental LIPIT studies, <sup>7,13</sup> as well as experimental gas gun studies of PC<sup>29</sup> where, in all cases, the failure stresses were not known.

## 3. DISCUSSION

The ballistic performance is traditionally assumed to be a function of the longitudinal acoustic wave speed of the material, 30,31 as the wave speed determines the boundary conditions, and thus the volume of material undergoing deformation, for a dynamic impact of a material with dimensions that is semi-infinite. However, eq 4 suggests that the minimum perforation velocity is related to other material properties that contribute to mitigating this type of impact  $(\rho_v)$  $\rho_v$  and  $\sigma_f$ ), as well as the geometric aspects of the impact test  $(a_p \text{ and } h)$ , which we attribute to the dimensions of a sample that should not be treated as semi-infinite. This insight poses interesting questions regarding designing perforation-resistant materials based on an intrinsic (acoustic wave speed) versus an extrinsic (size of material) property. It is a topic we are actively exploring, but we can explain the two different perspectives for  $V_0$  if we consider the failure of a linear elastic material that fails in a brittle manner, i.e.,  $\sigma_f = E_t \varepsilon_f$  with elastic modulus  $E_t$  and failure strain  $\varepsilon_f$ . Based on this assumption, eq 4 has a similar form to the Cunniff's expression for  $V_{50}$ ,  $^{3,31,32}$  which is a statistical alternative to  $V_0$  that is defined as the velocity at which 50% of the projectiles perforate the sample.<sup>33</sup> Specifically, our expression can be rederived as,  $V_0 \cong c_1(\rho_t h/\rho_p a_p)^{1/2} (\sigma_f \varepsilon_f/\rho_t)^{1/3} C_L^{1/3} \varepsilon_f^{-1/6}$ , where  $C_L = (E_t/\rho_t)^{0.5}$  is the longitudinal acoustic wave speed of the material. The derivation of this expression is provided in the Supporting Information section. We note that by using eq 4 instead of the classic Cunniff expression, we are assuming that the failure stress, as opposed to the work of fracture, defines the impact performance of these materials.

## 3.1. Projectile Size-Dependent Failure Strength. Figure 4c shows $\sigma_i$ for PC, npPMA, PS, and graphene (SLG, MLG) as a function of the projectile radius. The reported and calculated values of $\sigma_f$ are included in this plot. Superimposed upon this plot are dashed lines, indicating the scaling dependence of $\sigma_f \sim a_p^{-1/2}$ . This dependence comes from linear elastic fracture mechanics (LEFM), where the failure stress of a material also scales with the flaw size with the same exponent of -1/2. The larger the flaw size, the lower the failure stress. A careful examination of Figure 4c suggests that in these penetration events, including both the LIPIT and gas gun experiments, the projectile size can be considered as a flaw size. In LEFM, there is also a minimum or critical flaw size below which the failure stress becomes independent of the flaw size. For LIPIT experiments on the npPMA, PS, and MLG, the $\sigma_f \sim a_p^{-1/2}$ scaling appears to hold, suggesting that $a_p$ is always greater than each material's critical flaw size. On the other hand, there seems to be a plateau in $\sigma_f$ being reached with the microprojectiles used in LIPIT experiments on PC (especially when comparing the LIPIT data to the gas gun) and SLG, suggesting that $a_p$ may be less than the critical flaw size in these instances. This plateau for the SLG most likely reflects that defects were not included in the MD models, suggesting that the plateau represents a theoretical asymptote. Although additional LIPIT measurements using different microprojectile sizes are warranted to explore these observations in greater

<span id="page-6-0"></span>detail, these results strongly indicate that our methodology can predict the critical projectile size that defines the intrinsic versus extrinsic strength limits of materials used in impact mitigation applications.

Eq 4 shows that the minimum perforation velocity scales as  $V_0 \sim a_p^{-1/2}$ . At first glance, this relationship appears to contradict the results in Figure 2c, where we show  $V_0 \sim a_p^{-3/4}$ . They are, in fact, self-consistent. As seen in Figure 4c, we find that the failure stress scales as  $\sigma_f \sim a_p^{-1/2}$ . Ultimately,  $V_0$  scales with the size of the projectile in two separate terms so that  $V_0 \sim \sigma_f^{1/2} \cdot a_p^{-1/2}$ . Plugging in the dependence  $\sigma_f \sim a_p^{-1/2}$ ,  $V_0 \sim (a_p^{-1/2})^{1/2} \cdot a_p^{-1/2} \sim a_p^{-3/4}$ , exactly what was empirically determined in Figure 2c. Arriving at the same scaling relationship not only validates the relationship developed from the Buckingham  $\Pi$  analysis but also shows its utility in predicting the performance of materials due to various intrinsic and extrinsic factors. This is further demonstrated by the failure stress prediction for the gas gun experiment of PC, where the value was not reported, yet our prediction of  $\sigma_f$  matches the reported values from other gas gun experiments.

3.2. Predicting Perforation Resistance. Figure 4c also reveals several interesting findings. We note that npPMA with the 96 kDa graft length has a higher failure strength than PC at a projectile size of  $\approx 10 \ \mu m$ . We can make several comments about this. First, although we expect that the npPMA materials would be significantly weaker than PC as projectile size increases, they should be puncture-resistant against ultrasmall projectiles. Since our scaling analysis can guide material selection for different impact applications, these nanocomposite materials could be deployed in space applications as a protective coating where space dust typically impacts a spacecraft at ultrahigh velocities. Second, as can be seen from Figure 4d, the failure strength of a material also depends on the rate of deformation estimated as  $\approx V_0/2a_v$ .  $2a_v$  was used rather than h because the lateral strain of the film defines the average deformation, and thus deformation rate, of the material in these impact tests.<sup>7,11</sup> As this figure shows, the failure strength for PC shows a very weak correlation with an increasing deformation rate. Still, the npPMAs strongly depend on the deformation rate, especially at the higher strain rates achieved using LIPIT. This general trend is consistent with various materials using other high-rate mechanical tests. 6,36,37 However, we find it intriguing that these npPMAs are akin to granular materials, silica nanoparticles separated by a fuzzy corona of PMA. Granular materials, such as corn starch in water, undergo jamming transitions and stiffen significantly when deformed at high rates. While this mechanism remains to be verified, it is exciting to see that research is warranted to understand how soft materials resist size-dependent impacts at ultrahigh deformation rates.

We can also use Figure 4 to predict the performance of graphene. The minimum perforation velocity for SLG is the highest compared with the other materials studied in this work. These materials are nanoscale materials with thicknesses on the order of  $\approx$ 0.3 nm; thus, the enhancement in  $V_0$  cannot be attributed to geometry but is instead related to the high failure stress of these materials. Figure 4c suggests that these materials are below the flaw-sensitive regime, and thus the failure stress appears to approach the intrinsic limit for SLG. As the thickness and projectile size of these materials are increased, i.e., MLG, we find that the minimum perforation velocity decreases slightly due to the reduction in the failure stress. This result is extremely promising, as it suggests that the impact

properties are comparable to those of high-performance polymers such as PC. In relation to macroscale ballistic performance, these results demonstrate that novel materials such as graphene show significant potential for real-world applications.

## 4. CONCLUSION

By combining Buckingham Π dimensional analysis with experimental results from micro- and macroscale projectile tests, we developed a quantitative relationship between the size-scale and material properties that contribute to the highvelocity puncture resistance of impact mitigating materials. We demonstrated that the minimum perforation velocity of a material from the impact by a fast-moving projectile is related to both the geometric and materials properties of the impact. The Buckingham  $\Pi$  analysis yielded a relationship showing that  $V_0$  scales with the size-scale of the experiment and properties over impact energies and momentum values that span several orders of magnitude. The ability to connect the results from the microscale, where the experiments are more time and resource effective, to the macroscale will aid in the development of new materials and deepen our understanding of all impact events, whether the ultimate application is intended for protection against an impact like ballistic resistance or enhancement in penetration such as drug delivery.

## 5. METHODS

Certain instruments and materials are identified in this paper to adequately specify the experimental details. Such identification does not imply a recommendation by the National Institute of Standards and Technology nor does it imply that the materials are necessarily the best available for the purpose.

- **5.1. Ablation Target Preparation.** Ablation targets consisting of microparticles were prepared by first sputter coating 20 nm of gold onto a 20 cm  $\times$  40 cm glass coverslip. Next, polydimethylsiloxane (PDMS, Dow Sylgard 184, Ellsworth Adhesives) was mixed, degassed for 20 min, and subsequently deposited via spin-coating (2500 rpm for 10 min) to form a 20  $\mu$ m thick elastomeric layer. The deposited film was further degassed for another 10 min and finally thermally cured at 120 °C for 15 min. Microparticles (silica and lime glass, microParticles GmbH) were directly deposited onto the prepared substrate. An air gun was used to disperse the particles on the surface evenly and to remove any excess or large aggregates of particles. Inspection via optical microscopy confirmed that the microparticles were a monolayer thick.
- **5.2.** npPMA Film Preparation. Solid npPMA was dissolved in a mixture of toluene and tetrahydrofuran by stirring overnight with a Teflon stir bar to obtain solutions with a concentration of approximately = 120 mg/mL. Thin films of npPMA were fabricated by spin-coating the prepared solutions at 4000 rpm (=418.8 rad/s) for 10 s onto Ultraviolet-Ozone (UVO) cleaned silicon wafers that have been coated with a sacrificial layer of 1% by mass aqueous solution of poly(styrene sulfonate). Film thicknesses were measured with an optical interferometer (F3-NIR Profilometer, Filmetrics, KLA Corporation). Films were then sectioned into  $2 \times 2$  mm coupons with a razor blade and floated off in a water bath. The sectioned films were then transferred directly onto a TEM grid (PELCO, Tabbed 100 mesh, Nickel, Ted Pella, Inc.) that functions as a sample holder for the LIPIT experiment. Before testing, each grid was visually inspected with a video camera (PixeLink). The transferred films were in good contact with the TEM grid and were used without further processing.
- **5.3.** Laser-Induced Projectile Impact Test (LIPIT). A pulsed-diode-pumped solid-state IR laser (Flare NX,  $\lambda = 1030$  nm, pulse length = 1.5 ns, Coherent Inc.) was used as the ablation source to accelerate a single particle at the ablation target. A video camera

<span id="page-7-0"></span>(PixeLink) was used to measure, track, and focus onto individual microprojectiles through a 20× microscope objective (SLMPLN20×, NA = 0.25, Olympus). Each selected microparticle was digitally inspected before launch, with a small uncertainty in particle size potentially being slightly out-of-focus. The ablation target was set 1 mm away from the sample. A stroboscopic imaging technique was employed to determine the kinetic energy transfer of this high-rate impact event. IR pulses ( $\lambda = 1030$  nm, pulse length = 300 fs) from a diode-pumped, variable pulse length laser (Monaco Industrial Laser, Coherent Inc.) were converted into green light ( $\lambda = 515$  nm) using a doubling crystal. The strobe laser has a tunable repetition rate of 200 kHz to 10 MHz. A sCMOS camera (PCO Edge 4.2, PCO) captures the laser strobes and outputs a single image containing the spatial history of the microprojectile. The velocity of the microprojectile can be determined from the optical image by dividing the distance traveled by the time between each laser pulse, which was determined by the laser repetition rate. The error from the pixel resolution was < 1% of the measured interparticle distance. Synchronization of the ablation event, laser strobe, and image acquisition was achieved via digital triggers modulated using a digital waveform generator (NI-9402, National Instruments). At least 20 microprojectile impact tests were conducted for each npPMA system.

5.4. Buckingham  $\Pi$  Analysis: The Ideal Hydrodynamic, **Poncelet, and Tate Models.** To understand how various geometric test conditions and materials properties contribute to defining penetration resistance, we begin by discussing several classic penetration models used to predict the impact performance of materials.<sup>52</sup> These models were developed to predict penetration depth by a projectile, such as a long rod, into a semi-infinite material. One of the earliest models is the hydrodynamic model that was developed for predicting shaped-charge jet penetration by assuming that both the projectile and material are inviscid materials.<sup>20</sup> Based on the conservation of momentum, this model predicts that the penetration depth  $(\delta_t)$  of a projectile with a given length  $(l_n)$  into a semi-infinite target material should scale with the densities of the projectile and material,  $\delta_t/l_p \sim \sqrt{\rho_p/\rho_t}$ . The second is the Poncelet model, based on momentum and energy conservation.<sup>21</sup> This model predicts that the penetration depth into a material for a projectile of a given size to scale with the density and velocity of the projectile as well as the resistance of the material, with units of stress (=  $N/m^2$ ) that relate materials strength. While developed separately, the results of the hydrodynamic model can be derived using Buckingham  $\Pi$ theorem by relating the penetration depth into the material to the following independent variables

$$\delta_t = \delta_t(\nu_i, l_p, \rho_p, \rho_t) \tag{5}$$

where  $v_{ij}$   $l_{pj}$  and  $\rho_{pj}$  are the projectile's impact velocity, size, and density, respectively. The parameter  $\rho_t$  is the density of the material. Since we are assuming that the material is an incompressible fluid with negligible viscosity in the limit of infinite  $v_{ij}$  a resistance term related to the strength of the material is absent. There are n=5 physical variables and j=3 dimensions, and therefore,  $n-j=5-3=2\Pi_i$  groups consisting of the following variables and the corresponding dimensions.

To define  $\Pi_1$ , we choose  $v_i$ ,  $l_p$ ,  $\rho_t$  as the 3 repeating variables and  $\delta_t$  as the dependent variable

Table 2. Geometric and Materials Properties Used to Rederive the Hydrodynamic, Poncelet, and Tate Models<sup>a</sup>

| Parameter  | Description                     | Physical dimensions   |
|------------|---------------------------------|-----------------------|
| $\delta_t$ | penetration depth into material | [L]                   |
| $\nu_i$    | impact velocity                 | [L/T]                 |
| $l_p$      | projectile size                 | [L]                   |
| $\rho_p$   | projectile density              | $[M/L^3]$             |
| $\rho_t$   | material density                | $\lceil M/L^3 \rceil$ |

<sup>&</sup>lt;sup>a</sup>The dimensions L = length, M = mass, and T = time.

$$\Pi_1 = \delta_t v_i^{\alpha} l_p^{\beta} \rho_t^{\gamma} = [L] [L/T]^{\alpha} [L]^{\beta} [M/L^3]^{\gamma}$$
(6)

with the power law coefficients,  $\alpha = 0$ ,  $\beta = -1$ ,  $\gamma = 0$ . Therefore, eq 6 becomes.

$$\Pi_1 = \delta_t l_p^{-1} \tag{7}$$

To define  $\Pi_{2}$ ,  $\nu$ ,  $l_{p}$ ,  $\rho_{t}$  are the 3 repeating variables and  $\rho_{p}$  is the dependent variable

$$\Pi_{2} = \rho_{p} \nu^{\alpha} l_{p}^{\beta} \rho_{t}^{\gamma} = [M/L^{3}] [L/T]^{\alpha} [L]^{\beta} [M/L^{3}] \gamma$$
(8)

with power law coefficients,  $\alpha = 0$ ,  $\beta = 0$ ,  $\gamma = -1$ .Eq 8 becomes

$$\Pi_2 = \rho_p \rho_t^{-1} \tag{9}$$

Buckingham  $\Pi$  Theorem relates the  $\Pi$  as

$$\Pi_1 = \Phi(\Pi_2)$$

$$\frac{\delta_t}{l_p} \sim \left(\frac{\rho_p}{\rho_t}\right)^m \tag{10}$$

Eq 10 indicates that the penetration depth should scale with the ratio of the densities. Buckingham  $\Pi$  analysis does not explicitly define the exponent m, merely that one dimensionless group  $(\Pi_1)$  is related to another one  $(\Pi_2)$ . We do know that the hydrodynamic solution requires that m = 1/2.

$$\frac{\delta_t}{l_p} \sim \left(\frac{\rho_p}{\rho_t}\right)^{1/2} \tag{11}$$

We can also derive the Poncelet model by using Buckingham  $\Pi$  analysis. The model relates the  $\delta_t$  as a function of  $\nu_{\nu}$   $\rho_p$ , and strength of the material  $(\sigma_t)$ ,

$$\delta_t = \delta_t(\nu_i, l_p, \rho_p, \sigma_t) \tag{12}$$

There are 2  $\Pi$  groups. According to Buckingham  $\Pi$ , they can be related to each other as

$$\Pi_1 = \Phi(\Pi_2)$$

$$\frac{\delta_t}{l_p} \sim \frac{\rho_p v_i^2}{\sigma_t} \tag{13}$$

Eq 13 is the Poncelet model, which shows that the penetration depth scales with the impact velocity; i.e., the higher the velocity, the greater the penetration depth. The hydrodynamic model is another approximation that shows that the penetration depth should scale with the density of the material. Combining the dependent variables from these two models, we arrive at the following function

$$\delta_t = \delta_t(\nu_i, l_p, \rho_v, \rho_t, \sigma_t) \tag{14}$$

where  $\sigma_t$  describes the resistance of the material to penetration, and has units of  $[M/LT^2]$ . There are n=6 physical variables and j=3 dimensions. Therefore, there are  $n-j=5-3=3\Pi_i$  groups that are defined as

$$\Pi_1 = \Phi(\Pi_2, \Pi_2)$$

$$\frac{\delta_t}{l_p} = \Phi\left(\frac{\rho_p}{\rho_t}, \frac{\rho_t v_i^2}{\sigma_t}\right) \tag{15}$$

One of the solutions for eq 15 has already been defined based on the

$$\frac{\delta_{t}}{l_{p}} = \frac{\rho_{p}}{\rho_{t}} ln \left( 1 + \frac{\rho_{t} v_{i}^{2}}{2\sigma_{t}} \right) = \frac{\rho_{p}}{\rho_{t}} ln \left( 1 + \frac{v_{i}^{2}}{2\overline{\sigma_{t}}} \right)$$
(16)

<span id="page-8-0"></span>where we have defined  $\overline{\sigma}_t = \sigma_t/\rho_t$  as the specific material property of the target material. We can rearrange eq 16 and simplify it based on the magnitude of the dimensionless quantity  $v_t^2/\overline{\sigma}_t$ 

$$\frac{\rho_t}{\rho_p} \frac{\delta_t}{l_p} = \ln \left( 1 + \frac{\nu_i^2}{2\overline{\sigma}_t} \right) \approx \frac{\nu_i^2}{2\overline{\sigma}_t} - \frac{1}{2} \left( \frac{\nu_i^2}{2\overline{\sigma}_t} \right)^2 + \frac{1}{3} \left( \frac{\nu_i^2}{2\overline{\sigma}_t} \right)^3 - \dots$$

$$for - 1 < \nu_i^2 / \overline{\sigma}_t \le 1$$
(17)

For our problem of interest, which is how does the minimum perforation velocity  $(V_0)$  scales with thickness of the material (h) and radius of the spherical projectile  $(a_p)$ , we can relate  $V_0$  to the following set of independent variables,

$$V_0 = V_0(h, a_p, \rho_p, \rho_t, \sigma_f)$$
(18)

where we define  $\sigma_{j}$  as the failure strength of the material. The relationship between the  $3\Pi_{i}$  groups is,

$$\Pi_1 = \Phi(\Pi_2, \Pi_2)$$

$$\frac{\rho_t V_0^2}{\sigma_f} = \Phi\left(\frac{h}{a_p}, \frac{\rho_p}{\rho_t}\right) \tag{19}$$

which is similar to eq 15. We apply the simplified Tate model as a possible solution to yield the final result,

$$\frac{\rho_t V_0^2}{\sigma_f} = c_1 \frac{\rho_t}{\rho_p} \frac{h}{a_p} \tag{20}$$

## ASSOCIATED CONTENT

## **Solution** Supporting Information

The Supporting Information is available free of charge at https://pubs.acs.org/doi/10.1021/acsami.3c05130.

Plot of energy scales for microprojectile vs macroprojectile impacts, comparing the derivation of the Cunniff model with our model, and tables of projectile impact data used to generate the results in Figures 2–4 (PDF)

## AUTHOR INFORMATION

## **Corresponding Author**

Edwin P. Chan — National Institute of Standards and Technology,, Materials Science and Engineering Division, Gaithersburg, Maryland 20899, United States; orcid.org/0000-0003-4832-6299; Email: edwin.chan@nist.gov

## **Authors**

Katherine M. Evans – National Institute of Standards and Technology,, Materials Science and Engineering Division, Gaithersburg, Maryland 20899, United States

Shawn H. Chen – National Institute of Standards and Technology,, Materials Measurement Sciences Division, Gaithersburg, Maryland 20899, United States

Amanda J. Souna – National Institute of Standards and Technology,, Materials Science and Engineering Division, Gaithersburg, Maryland 20899, United States

Stephan J. Stranick – National Institute of Standards and Technology,, Materials Measurement Sciences Division, Gaithersburg, Maryland 20899, United States

Christopher L. Soles — National Institute of Standards and Technology,, Materials Science and Engineering Division, Gaithersburg, Maryland 20899, United States; orcid.org/0000-0002-1963-6039

Complete contact information is available at:

https://pubs.acs.org/10.1021/acsami.3c05130

#### Note

The authors declare no competing financial interest.

## ACKNOWLEDGMENTS

This work was supported in part by an Interagency Agreement with the U.S. Army Engineer Research and Development Center (ERDC). K.M.E. also acknowledges financial support from the National Research Council (NRC) Postdoctoral Fellowship award. The authors thank Prof. Sanat Kumar for kindly providing the npPMA materials used in this work. The authors also greatly appreciate the insightful discussions with Dr. Joseph Dennis, Dr. Emil Sandoz-Rosado, and Dr. Eric Wetzel. This work is a contribution of NIST, an agency of the U.S. Government, and not subject to U.S. copyright.

#### REFERENCES

- (1) Nüchter, T.; Benoit, M.; Engel, U.; Özbek, S.; Holstein, T. W. Nanosecond-Scale Kinetics of Nematocyst Discharge. *Curr. Biol.* **2006**, *16*, R316–R318.
- (2) Schramm, J.; Mitragotri, S. Transdermal Drug Delivery By Jet Injectors: Energetics of Jet Formation and Penetration. *Pharm. Res.* **2002**, *19*, 1673–1679.
- (3) Phoenix, S. L.; Porwal, P. K. A New Membrane Model for the Ballistic Impact Response and V50 Performance of Multi-Ply Fibrous Systems. *International Journal of Solids and Structures* **2003**, 40, 6723–6765.
- (4) Grossman, E.; Gouzman, I.; Verker, R. Debris/Micrometeoroid Impacts and Synergistic Effects on Spacecraft Materials. *MRS Bulletin* **2010**, 35, 41–47.
- (5) Lee, J.-H.; Veysset, D.; Singer, J. P.; Retsch, M.; Saini, G.; Pezeril, T.; Nelson, K. A.; Thomas, E. L. High Strain Rate Deformation of Layered Nanocomposites. *Nature Communications* **2012**. 3, 1164.
- (6) Veysset, D.; Lee, J.-H.; Hassani, M.; Kooi, S. E.; Thomas, E. L.; Nelson, K. A. High-Velocity Micro-Projectile Impact Testing. *Applied Physics Reviews* **2021**, *8*, 011319.
- (7) Lee, J.-H.; Loya, P. E.; Lou, J.; Thomas, E. L. Dynamic Mechanical Behavior of Multilayer Graphene via Supersonic Projectile Penetration. *Science* **2014**, 346, 1092–1096.
- (8) Zhu, Y.; Giuntoli, A.; Hansoge, N.; Lin, Z.; Keten, S. Scaling for the Inverse Thickness Dependence of Specific Penetration Energy in Polymer Thin Film Impact Tests. *Journal of the Mechanics and Physics of Solids* **2022**, *161*, 104808–104808.
- (9) Chen, S. H.; Souna, A. J.; Stranick, S. J.; Jhalaria, M.; Kumar, S. K.; Soles, C. L.; Chan, E. P. Controlling Toughness of Polymer-Grafted Nanoparticle Composites for Impact Mitigation. *Soft Matter* **2022**, *18*, 256–261.
- (10) Recht, R. F.; Ipson, T. W. Ballistic Perforation Dynamics. *Journal of Applied Mechanics* **1963**, 30, 384–390.
- (11) Chen, S. H.; Souna, A. J.; Soles, C. L.; Stranick, S. J.; Chan, E. P. Using Microprojectiles to Study the Ballistic Limit of Polymer Thin Films. *Soft Matter* **2020**, *16*, 3886–3890.
- (12) Meng, Z.; Singh, A.; Qin, X.; Keten, S. Reduced Ballistic Limit Velocity of Graphene Membranes due to Cone Wave Reflection. *Extreme Mechanics Letters* **2017**, *15*, 70–77.
- (13) Xie, W.; Lee, J.-H. Intrinsic Dynamics and Toughening Mechanism of Multilayer Graphene upon Microbullet Impact. ACS Appl. Nano Mater. 2020, 3, 9185–9191.
- (14) Tsai, C. T.; Mayer, A. Finite Element Analysis of Impact and Penetration of Polycarbonate Plate by a Rigid Spherical Projectile; Cold Regions Research and Engineering Laboratory (U.S.), 1999.
- (15) Nandlall, D.; Chrysler, J. A Numerical Analysis of the Ballistic Performance of a 6.35-mm Transparent Polycarbonate Plate; Technical Reports; Defence Research Establishment, Centre De Recherches Pour La Defense: Valcartier, Quebec, 1998.

- <span id="page-9-0"></span>(16) Wright, S. C.; Huang, Y.; Fleck, N. A. Deep [Penetration](https://doi.org/10.1016/0167-6636(92)90020-E) of [Polycarbonate](https://doi.org/10.1016/0167-6636(92)90020-E) by a Cylindrical Punch. *Mechanics of Materials* 1992, *13*, 277−284.
- (17) Wright, S. C.; Fleck, N. A.; Stronge, W. J. [Ballistic](https://doi.org/10.1016/0734-743X(93)90105-G) Impact of [Polycarbonate](https://doi.org/10.1016/0734-743X(93)90105-G) - An Experimental Investigation. *International Journal of Impact Engineering* 1993, *13*, 1−20.
- (18) Yong, M.; Iannucci, L.; Falzon, B. Efficient [Modelling](https://doi.org/10.1016/j.ijimpeng.2009.07.004) and [Optimisation](https://doi.org/10.1016/j.ijimpeng.2009.07.004) of Hybrid Multilayered Plates Subject to Ballistic [Impact.](https://doi.org/10.1016/j.ijimpeng.2009.07.004) *International Journal of Impact Engineering* 2010, *37*, 605−624.
- (19) Fadhel, B. M. Numerically Study of Ballistic Impact of Polycarbonate. *2011 International Symposium on Humanities, Science and Engineering Research*, Kuala Lumpur, Malaysia, 2011; pp 101− 105.
- (20) Birkhoff, G.; MacDougall, D. P.; Pugh, E. M.; Taylor, G. [Explosives](https://doi.org/10.1063/1.1698173) with Lined Cavities. *J. Appl. Phys.* 1948, *19*, 563−582.
- (21) Allen, W. A.; Mayfield, E. B.; Morrison, H. L. [Dynamics](https://doi.org/10.1063/1.1722750) of a Projectile [Penetrating](https://doi.org/10.1063/1.1722750) Sand. *J. Appl. Phys.* 1957, *28*, 370−376.
- (22) Veysset, D.; Sun, Y.; Lem, J.; Kooi, S. E.; Maznev, A. A.; Cole, S. T.; Mrozek, R. A.; Lenhart, J. L.; Nelson, K. A. [High-Strain-Rate](https://doi.org/10.1007/s11340-020-00639-9) Behavior of a Viscoelastic Gel Under [High-Velocity](https://doi.org/10.1007/s11340-020-00639-9) Microparticle [Impact.](https://doi.org/10.1007/s11340-020-00639-9) *Experimental Mechanics* 2020, *60*, 1179−1186.
- (23) Buckingham, E. On Physically Similar Systems; [Illustrations](https://doi.org/10.1103/PhysRev.4.345) of the Use of [Dimensional](https://doi.org/10.1103/PhysRev.4.345) Equations. *Phys. Rev.* 1914, *4*, 345−376.
- (24) Portela, C. M.; Edwards, B. W.; Veysset, D.; Sun, Y.; Nelson, K. A.; Kochmann, D. M.; Greer, J. R. [Supersonic](https://doi.org/10.1038/s41563-021-01033-z) Impact Resilience of [Nanoarchitected](https://doi.org/10.1038/s41563-021-01033-z) Carbon. *Nat. Mater.* 2021, *20*, 1491.
- (25) Gu, Z.; Cheng, Y.; Xiao, K.; Li, K.; Wu, X.; Li, Q.; Huang, C. Geometrical Scaling Law for Laser-Induced [Micro-Projectile](https://doi.org/10.1016/j.ijmecsci.2022.107289) Impact [Testing.](https://doi.org/10.1016/j.ijmecsci.2022.107289) *International Journal of Mechanical Sciences* 2022, *223*, 107289.
- (26) Fleck, N. A.; Stronge, W. J.; Liu, J. H. High [Strain-Rate](https://doi.org/10.1098/rspa.1990.0069) Shear Response of [Polycarbonate](https://doi.org/10.1098/rspa.1990.0069) and Polymethyl Methacrylate. *Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences* 1990, *429*, 459−479.
- (27) Xu, Y.; Gao, T.; Wang, J.; Zhang, W. [Experimentation](https://doi.org/10.3390/polym8030063) and Modeling of the Tension Behavior of [Polycarbonate](https://doi.org/10.3390/polym8030063) at High Strain [Rates.](https://doi.org/10.3390/polym8030063) *Polymers* 2016, *8*, 63.
- (28) Xie, W.; Lee, J.-H. Dynamics of [Entangled](https://doi.org/10.1021/acs.macromol.9b02265?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) Networks in Ultrafast Perforation of Polystyrene [Nanomembranes.](https://doi.org/10.1021/acs.macromol.9b02265?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) *Macromolecules* 2020, *53*, 1701−1705.
- (29) Callahan, K.; Heard, W. F.; Kundu, S. High Strain Rate [Failure](https://doi.org/10.1021/acs.macromol.2c01151?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) Behavior of [Polycarbonate](https://doi.org/10.1021/acs.macromol.2c01151?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) Plates due to Hypervelocity Impact. *Macromolecules* 2022, *55*, 9640−9649.
- (30) Abtew, M. A.; Boussu, F.; Bruniaux, P.; Loghin, C.; Cristian, I. Ballistic Impact [Mechanisms](https://doi.org/10.1016/j.compstruct.2019.110966) A Review on Textiles and Fibre-Reinforced [Composites](https://doi.org/10.1016/j.compstruct.2019.110966) Impact Responses. *Composite Structures* 2019, *223*, 110966.
- (31) Cunniff, P. Dimensionless Parameters for Optimization of Textile Based Body Armor Systems. *Proceedings of the 18th International Symposium on Ballistics*; 1999; pp 1303−1310.
- (32) Phoenix, S.; Heisserer, U.; van der Werff, H.; van der Jagt-Deutekom, M. Modeling and [Experiments](https://doi.org/10.3390/fib5010008) on Ballistic Impact into UHMWPE Yarns Using Flat and [Saddle-Nosed](https://doi.org/10.3390/fib5010008) Projectiles. *Fibers* 2017, *5*, 8.
- (33) Johnson, T. H.; Freeman, L.; Hester, J.; Bell, J. L. [A](https://doi.org/10.1109/ACCESS.2014.2377633) [Comparison](https://doi.org/10.1109/ACCESS.2014.2377633) of Ballistic Resistance Testing Techniques in the [Department](https://doi.org/10.1109/ACCESS.2014.2377633) of Defense. *IEEE Access* 2014, *2*, 1442−1455.
- (34) Creton, C.; Ciccotti, M. Fracture and [Adhesion](https://doi.org/10.1088/0034-4885/79/4/046601) of Soft [Materials:](https://doi.org/10.1088/0034-4885/79/4/046601) A Review. *Rep. Prog. Phys.* 2016, *79*, 046601.
- (35) Long, R.; Hui, C.-Y.; Gong, J. P.; Bouchbinder, E. The [Fracture](https://doi.org/10.1146/annurev-conmatphys-042020-023937) of Highly [Deformable](https://doi.org/10.1146/annurev-conmatphys-042020-023937) Soft Materials: A Tale of Two Length Scales. *Annual Review of Condensed Matter Physics* 2021, *12*, 71−94.
- (36) Mulliken, A. D.; Boyce, M. C. [Mechanics](https://doi.org/10.1016/j.ijsolstr.2005.04.016) of the Rate-Dependent [ElasticPlastic](https://doi.org/10.1016/j.ijsolstr.2005.04.016) Deformation of Glassy Polymers from Low to High Strain [Rates.](https://doi.org/10.1016/j.ijsolstr.2005.04.016) *International Journal of Solids and Structures* 2006, *43*, 1331−1356.
- (37) Sarva, S. S.; Boyce, M. C. Mechanics of [Polycarbonate](https://doi.org/10.2140/jomms.2007.2.1853) Under [High-Rate](https://doi.org/10.2140/jomms.2007.2.1853) Tension. *Journal of Mechanics of Materials and Structures* 2007, *2*, 1853−1880.

- (38) Akers, B.; Belmonte, A. Impact [Dynamics](https://doi.org/10.1016/j.jnnfm.2006.01.004) of a Solid Sphere Falling into a [Viscoelastic](https://doi.org/10.1016/j.jnnfm.2006.01.004) Micellar Fluid. *Journal of Non-Newtonian Fluid Mechanics* 2006, *135*, 97−108.
- (39) Snider, E. J.; Cornell, L. E.; Acevedo, J. M.; Gross, B.; Edsall, P. R.; Lund, B. J.; Zamora, D. O. Development and [Characterization](https://doi.org/10.1038/s41598-020-61079-y) of a [Benchtop](https://doi.org/10.1038/s41598-020-61079-y) Corneal Puncture Injury Model. *Scientific Reports* 2020, *10*, 4218.
- (40) Ghods, S.; Murcia, S.; Ossa, E. A.; Arola, D. [Designed](https://doi.org/10.1016/j.jmbbm.2018.10.037) for [Resistance](https://doi.org/10.1016/j.jmbbm.2018.10.037) to Puncture: The Dynamic Response of Fish Scales. *Journal of the Mechanical Behavior of Biomedical Materials* 2019, *90*, 451−459.
- (41) Cui, J.; Shi, Y.; Zhang, X.; Huang, W.; Ma, M. [Experimental](https://doi.org/10.1016/j.polymertesting.2020.106863) Study on the Tension and [Puncture](https://doi.org/10.1016/j.polymertesting.2020.106863) Behavior of Spray Polyurea at High [Strain](https://doi.org/10.1016/j.polymertesting.2020.106863) Rates. *Polymer Testing* 2021, *93*, 106863−106863.
- (42) Baxter, J.; Mitragotri, S. [Jet-induced](https://doi.org/10.1016/j.jconrel.2005.05.023) Skin Puncture and its Impact on Needle-Free Jet Injections: [Experimental](https://doi.org/10.1016/j.jconrel.2005.05.023) Studies and a [Predictive](https://doi.org/10.1016/j.jconrel.2005.05.023) Model. *J. Controlled Release* 2005, *106*, 361−373.
- (43) Kendall, M.; Mitchell, T.; Wrighton-Smith, P. [Intradermal](https://doi.org/10.1016/j.jbiomech.2004.01.032) Ballistic Delivery of [Micro-Particles](https://doi.org/10.1016/j.jbiomech.2004.01.032) into Excised Human Skin for [Pharmaceutical](https://doi.org/10.1016/j.jbiomech.2004.01.032) Applications. *Journal of Biomechanics* 2004, *37*, 1733− 1741.
- (44) Anderson, P. S.; Crofts, S. B.; Kim, J. T.; Chamorro, L. P. Taking a Stab at [Quantifying](https://doi.org/10.1093/icb/icz078) the Energetics of Biological Puncture. *Integrative and Comparative Biology* 2019, *59*, 1586−1596.
- (45) Whitford, M. D.; Freymiller, G. A.; Higham, T. E.; Clark, R. W. The Effects of [Temperature](https://doi.org/10.1093/iob/obaa025) on the Kinematics of Rattlesnake Predatory Strikes in Both Captive and Field [Environments.](https://doi.org/10.1093/iob/obaa025) *Integrative Organismal Biology* 2020, *2*, obaa025.
- (46) O'Leary, M. D.; Simone, C.; Washio, T.; Yoshinaka, K.; Okamura, A. M. Robotic Needle [Insertion:](https://doi.org/10.1109/ROBOT.2003.1241851) Effects of Friction and Needle [Geometry.](https://doi.org/10.1109/ROBOT.2003.1241851) *Proceedings - IEEE International Conference on Robotics and Automation* 2003, *2*, 1774−1780.
- (47) Maiorana, C. H.; Jotawar, R. A.; German, G. K. [Biomechanical](https://doi.org/10.1039/D1SM01187A) Fracture Mechanics of [Composite](https://doi.org/10.1039/D1SM01187A) Layered Skin-Like Materials. *Soft Matter* 2022, *18*, 2104−2112.
- (48) Yamaguchi, S.; Tsutsui, K.; Satake, K.; Morikawa, S.; Shirai, Y.; Tanaka, H. T. [Dynamic](https://doi.org/10.1016/j.compbiomed.2014.07.012) Analysis of a Needle Insertion for Soft Materials: Arbitrary [Lagrangian-Eulerian-Based](https://doi.org/10.1016/j.compbiomed.2014.07.012) Three-Dimensional Finite Element [Analysis.](https://doi.org/10.1016/j.compbiomed.2014.07.012) *Computers in Biology and Medicine* 2014, *53*, 42−47.
- (49) Cho, W. K.; Ankrum, J. A.; Guo, D.; Chester, S. A.; Yang, S. Y.; Kashyap, A.; Campbell, G. A.; Wood, R. J.; Rijal, R. K.; Karnik, R.; Langer, R.; Karp, J. M. [Microstructured](https://doi.org/10.1073/pnas.1216441109) Barbs on the North American Porcupine Quill Enable Easy Tissue [Penetration](https://doi.org/10.1073/pnas.1216441109) and Difficult [Removal.](https://doi.org/10.1073/pnas.1216441109) *Proceedings of the National Academy of Sciences* 2012, *109*, 21289−21294.
- (50) Bao, Y. d.; Qu, S. q.; Qi, D. b.; Wei, W. [Investigation](https://doi.org/10.1016/j.jmbbm.2021.104958) on Puncture Mechanical [Performance](https://doi.org/10.1016/j.jmbbm.2021.104958) of Tracheal Tissue. *Journal of the Mechanical Behavior of Biomedical Materials* 2022, *125*, 104958− 104958.
- (51) Leibinger, A.; Oldfield, M. J.; Rodriguez Y Baena, F. [Minimally](https://doi.org/10.1098/rsfs.2015.0107) Disruptive Needle Insertion: A [Biologically](https://doi.org/10.1098/rsfs.2015.0107) Inspired Solution. *Interface Focus* 2016, *6*, 20150107.
- (52) Anderson, C. E., Jr. Analytical Models for [Penetration](https://doi.org/10.1016/j.ijimpeng.2017.03.018) [Mechanics:](https://doi.org/10.1016/j.ijimpeng.2017.03.018) A Review. *International Journal of Impact Engineering* 2017, *108*, 3−26.