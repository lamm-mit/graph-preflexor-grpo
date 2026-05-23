## **Eve Langelier**

PERSEUS, De´partement de ge´nie me´canique, Universite´ de Sherbrooke, Sherbrooke (Que´bec), J1K 2R1, Canada telephone: (819) 821-8000 ext. 2998, fax: (819) 821-7163 e-mail: eve.langelier@usherbrooke.ca

## **Daniel Dupuis**

Faculte´ des sciences et de ge´nie–Dean's office, Universite Laval, Pavillon Adrien-Pouliot, local 1310 Que´bec, Canada G1K 7P4 Telephone: (418) 656-2131 ext. 12468,

e-mail: daniel.dupuis@fsg.ulaval.ca

# **Michel Guillot**

MultiSigma Inc, 736 Avenue Godin, Que´bec, Que´bec, Canada telephone: (418) 688-4000, fax:(418) 688-4000, e-mail: direction@multisigma.com

## **Francine Goulet**

Tissue Engineering Laboratory, Pavillon Notre-Dame, H-401, CHA, Hoˆpital de l'Enfant-Je´sus, 1401 18<sup>e</sup> rue, Que´bec (Que´bec), Canada, G1J 1Z4, telephone: (418) 682-7765, fax:(418) 649-5969 e-mail: chgfgo@hermes.ulaval.ca

# **Denis Rancourt**

PERSEUS, De´partement de ge´nie me´canique, Universite´ de Sherbrooke, Sherbrooke (Que´bec), J1K 2R1, Canada, telephone: (819) 821-8000 ext.1346, fax: (819) 821-7163 e-mail: denis.rancourt@usherbrooke.ca

# **Cross-Sectional Profiles and Volume Reconstructions of Soft Tissues Using Laser Beam Measurements**

*Precise geometric reconstruction is a valuable tool in the study of soft tissues biomechanics. Optical methods have been developed to determine the tissue cross section without mechanical contact with the specimen. An adaptation of the laser micrometer developed by Lee and Woo [ASME J. Biomech. Eng., 110 (2), pp. 110*–*114]. is proposed in which the laser-collimated beam rotates around and moves along a fixed specimen to reconstruct its cross sections and volume. Beam motion is computer controlled to accelerate data acquisition and improve beam positioning accuracy. It minimizes time-dependent shape modifications and increases global reconstruction precision. The technique is also competent for the measurement of immersed collagen matrices.* @DOI: 10.1115/1.1824125#

*Keywords: Laser, Micrometer, Cross Section, Profile, Volume, Algorithm, Soft Tissues, Collagen Matrices*

#### **Introduction**

With the growing interest in tissue engineering, a number of measuring devices must be developed or improved for better in vitro characterization of reconstructed tissues. One valuable device is a system to accurately measure the three-dimensional geometry of reconstructed tissues. Tissue geometry is important for several reasons: it is essential to normalize tissue stress–strain curves, it helps monitor the tissue dynamic behavior such as degradation over time @1#, it can provide some insights on biological phenomena such as tissue contraction occurring in a living soft tissue in vitro @2#, and it provides a method to construct accurate geometrical models of soft tissues for finite element analysis ~FEA!.

Many groups have contributed in the past to improving crosssection measurement systems for ligaments and tendons speci-

Correspondence to: Denis Rancourt.

Contributed by the Bioengineering Division for publication in the JOURNAL OF BIOMECHANICAL ENGINEERING. Manuscript received by the Bioengineering Division April 28, 2003; revision received May 26, 2004. Associate Editor: Philip V. Bayly

mens @3–15#. Optical @4,5,7–10,15#, ultrasound @3,6,11#, and mechanical @12–14# techniques were investigated. Earlier attempts to evaluate the cross-sectional area of soft tissues are reviewed in Ellis @5#. Based on the work of Lee and Woo @8#, we have developed a new computer-controlled apparatus to reconstruct profiles and volumes of engineered soft tissues using a laser-collimated beam technology. Three-dimensional reconstruction is obtained by measuring cross-sectional profiles at different positions along the specimen axis. It is a precise and repeatable noncontact technique which is independent of the operator skills @15#, unlike ultrasound @3#. A noncontact technique is desirable because of the high compliance of soft tissue. Our apparatus was specifically designed to measure the three-dimensional geometry of reconstructed collagen matrices.

Due to the particular mechanical properties of reconstructed collagen matrices, the sagging issue mentioned by Lee and Woo @8# ~i.e., deformation of the specimen due to gravity that creates an undesirable change of specimen shape over time! is highly amplified when the specimen is rotated for cross-section measurements. For native soft tissues, sagging may be prevented by the application of a small preload tension @8,15#. However, we found sagging becomes a crucial issue when working with reconstituted collagen matrices which are very soft and fragile. Sufficient preloading to prevent sagging deforms the specimen and can induce tissue damage. This problem is avoided if the specimen is maintained fixed in space while the laser beam rotates about it.

#### **Materials and Methods**

**Apparatus.** The laser micrometer is shown in Fig. 1. A microprocessor controlled laser-collimated beam of 10  $\mu$ m resolution (Z4LC-S28, Omron Electronics Inc, Schaumburg, IL, U.S.A.) is used to measure the specimen width (SW) at different angles and different positions along the collagen matrix longitudinal axis. Rotation of the beam around the specimen is generated by a stepping motor (HT23-396, Applied Motion Products Inc, Watsonville, CA, U.S.A.) and the angular position is measured via an encoder (Dynapar M21 series, Danaher Controls, Gurnee, IL, U.S.A.). A second stepping motor is used along with a lead screw to displace the beam along the specimen (z axis). The motor angular position is measured via a second encoder. Servo-control is achieved by a dual axis stepper motor control system (MAX-410/420, AMS Inc, Nashua, NH, U.S.A.) and data acquisition is performed by MS PROFILE, a Visual C++ software.

**Profile and Volume Reconstruction Algorithm.** The algorithm for the reconstruction of the specimen cross-sectional profiles and volume is illustrated in the flowchart shown in Fig. 2. First, global and local Cartesian reference systems are defined. To that end, one needs to locate the center of rotation (COR) of the rotating collimated laser beam relative to the beam itself. Thereafter, data (i.e., specimen edges) are acquired with the laser beam oriented at a number of discrete angles around the specimen (a 0 to 180 deg range is sufficient since the two opposite specimen edges are simultaneously acquired at each measurement, thus covering the whole 0 to 360 deg range) and at a number of z discrete positions along the specimen (Fig. 1). The cross-sectional profiles at each z location and the specimen volume are reconstructed from the specimen edges and profile z positions. The detail reconstruction procedure is described below.

1 Defining the Global and Local Reference Frames. The ordinate Y of the global reference system passes through the laser beam COR. The global abscissa X is defined as the lower beam edge (0.00 mm) for a rotation angle  $\theta = 0$  deg (Fig. 3). The local UV reference system is similarly defined, except that it rotates with the laser beam. The COR location of the rotating collimated laser beam is obtained from a calibration specimen's lower and upper edges measured at  $\theta = 0$  deg and  $\theta = 180$  deg (Fig. 4). Since the COR projection on the beam receiver remains constant for all angles, symmetry of the specimen edges projections is observed on each side of the COR for  $\theta = 0$  deg and  $\theta = 180$  deg. Therefore, the COR location is given by

Fig. 1 Photograph of the laser micrometer. The laser beam sensor rotates around and translates along the specimen to measure its width at different angles and different z positions.

Fig. 2 Flowchart illustrating the reconstruction algorithm. CP stands for "confining polygon" and PA for "parallelepiped area"

COR = Average 
$$\left[ \frac{(V_0^L + V_{180}^U)}{2}; \frac{(V_0^U + V_{180}^L)}{2} \right]$$
 (1)

where  $V^L$  and  $V^U$  represent the positions of the first and second edges projections detected in the local reference frame (Fig. 3).

Fig. 3 The origins of the global and local Cartesian reference systems are set by the lower edge of the laser beam (0.00 mm) for a rotation angle  $\theta$ =0 deg (x axis) and by its COR (y axis). Note that the global system is fixed in space, while the local system rotates with the laser beam. The measured edges of the specimen are referred to as  $V^L$  and  $V^U$ ,  $V^L$  being closer to the lower beam edge (0.00 mm), and  $V^U$  to the upper beam edge (28.00 mm).

Fig. 4 The location of the COR is determined from the data measured at  $\theta$ =0 and  $\theta$ =180 deg. Note the reflection of the specimen projection each side of the COR at  $\theta$ =0 and  $\theta$ =180 deg. The localization formula is described in the text.

It is to be noted that the COR position may differ from one z position to another due to misalignment of the main axis of the calibration specimen with the laser beam z-displacement axis. The COR must then be determined over the whole z-displacement range. These steps are part of the machine precalibration and are not conducted thereafter for each specimen.

- 2 Data Acquisition. At each z position along the specimen,  $V^L$  and  $V^U$  data are recorded at  $\Delta \theta$  increments between 0 and 180 deg. They represent the positions of the first and second edges projections detected in the local reference frame (Fig. 3). The longitudinal position z for each measurement is also recorded.
- 3 Reconstruction of a Cross-Sectional Profile at a Given z Location. Any reconstructed cross-sectional profile is a discretization of the true specimen profile. This discretized profile is called a profile polygon (PP). The PP corresponds to the intersection of all parallelepiped areas (PAs) delimited by the lower and upper edge projections of the specimen ( $V^L$  and  $V^U$ ) measured at  $\Delta\theta$  increments between 0 and 180 deg (Fig. 5). The intersection of the first two PAs ( $\theta$ =0 and  $\theta$ = $\Delta\theta$  deg) defines a four-sided polygon called the initial confining polygon (CP). This polygon is progressively reduced to the PP by intersecting it with all subsequent PAs obtained at  $\Delta\theta$  increments between  $2\Delta\theta$  and 180 deg. Mathematically, the CP at an angle  $\theta$  is defined by

$$CP_{\theta} = CP_{\theta - \Delta\theta} \cap PA_{\theta} \tag{2}$$

This iterative process is illustrated in Fig. 6.

3.1 Transposition of the data from the local to the global reference system. The CP at a given angle  $\theta$  is truncated to the specimen width (SW) by the upper and lower edges of the specimen measured at this angle  $\theta$  ( $V_{\theta}^{L}$  and  $V_{\theta}^{U}$ ) in the local reference frame

Fig. 5 The reconstructed cross-sectional profile is a polygon (PP) corresponding to the intersection of PAs obtained at  $\Delta\theta$  increments between 0 and 180 deg.

Fig. 6 Illustration of three consecutive steps of the iterative process used to obtain the final cross-sectional profile. (a) The initial CP is bounded by the sectioning lines (SLs) obtained at  $\theta{=}0$  and  $\theta{=}\Delta\,\theta$  deg. (b) and (c) Dashed lines are derived from the precedent confining polygon (CP $_{\theta{-}\Delta\,\theta}$ ), while continuous lines represent the PA $_{\theta}$ . Circles identify the CP $_{\theta{-}\Delta\,\theta}$  vertices, while filled circles identify new intersection points. Only the points lying within the PA $_{\theta}$  and the CP $_{\theta{-}\Delta\,\theta}$  define the CP $_{\theta}$ .

$$SW_{\theta} = V_{\theta}^{U} - V_{\theta}^{L} \tag{3}$$

Definition of these edges in the global reference frame is essential to the reconstruction algorithm.  $V_{\theta}^{L}$  and  $V_{\theta}^{U}$  data define two straight lines referred to as sectioning lines (SLs) in the global reference frame (Figs. 6, 7) given by

$$Y_{\theta}^{L} = M_{\theta}^{L} X_{\theta} + B_{\theta}^{L} \tag{4a}$$

and 
$$Y_{\theta}^U = M_{\theta}^U X_{\theta} + B_{\theta}^U$$
 (4b)

Fig. 7 Data transposition from the local to the global reference frame. For an angle  $\theta$ , the profile width is represented by two SLs.  $B^L$  and  $B^U$  correspond to the Y intercepts, and M, to the slope of these SLs.

where  $M_{\theta}^L$  and  $M_{\theta}^U$  represent the respective slopes, and  $B_{\theta}^L$  and  $B_{\theta}^U$ , the Y intercepts. They are given by

$$M_{\theta}^{L} = M_{\theta}^{U} = \tan \theta \tag{5}$$

$$B_{\theta}^{L} = \text{COR} + \left(\frac{V_{\theta}^{L} - \text{COR}}{\cos \theta}\right) \tag{6a}$$

and 
$$B_{\theta}^{U} = B_{\theta}^{L} + \left(\frac{SW_{\theta}}{\cos \theta}\right)$$
 (6b)

Figure 8 illustrates the SLs obtained from  $V_{\theta}^{L}$  and  $V_{\theta}^{U}$  data for a 6 mm hex key measured at 10 deg increments.

3.2 Definition of the initial CP. In the proposed algorithm, the CP that changes from the initial CP to the PP is always defined by the points located at its vertices. For the initial CP, the vertices correspond to the intersections between the two SLs obtained at  $\theta$ =0 deg and the two SLs obtained at  $\theta$ = $\Delta \theta$  deg [Fig. 6(a)]. Four intersection points ( $X_{\rm inter}$ ,  $Y_{\rm inter}$ ) define the initial CP. They are given by

$$X_{\text{inter}_i} = \frac{(B_{i+1} - B_i)}{(M_i - M_{i+1})} \tag{7}$$

Fig. 8 SLs obtained from  $V^L$  and  $V^U$  data for an Allen key measured at 10-deg intervals.

$$Y_{\text{inter}_i} = M_i X_{\text{inter}_i} + B_i \tag{8}$$

for i = 1, ..., 4.

- 3.3 Stepwise reduction of the CP. As mentioned earlier, the initial CP is progressively reduced to the PP by intersecting the PAs obtained at  $\Delta\theta$  increments between  $2\Delta\theta$  and 180 deg. This is performed, at each angle  $\theta$ , by determining the intersections between the SLs delimiting the PA $_{\theta}$  and the line segments delimiting the CP $_{\theta-\Delta\theta}$ . The new CP $_{\theta}$  is solely defined by the intersection points included within the PA $_{\theta}$  and the CP $_{\theta-\Delta\theta}$  (Fig. 6).
- 3.3.1 Definition of the  $PA_{\theta}$ . For the purpose of the algorithm, the PAs must be of finite size. One simple solution is to use PAs whose heights are defined by the two SLs, and widths, by  $X_{\min}$  and  $X_{\max}$  values given by

$$X_{\min} = \text{COR-UpperBeamEdge}$$
 (9)

$$X_{\text{max}} = \text{COR-LowerBeamEdge}$$
 (10)

3.3.2 Redefinition of the  $CP_{\theta-\Delta\theta}$ . A CP is initially defined by its vertices  $(X_i, Y_i)$ . In order to calculate the intersection between the CP and the PA, one needs to redefine the CP based on its sides, i.e., line segments. They are expressed in straight-line equations of the form Y = MX + B, where the slopes and Y intercepts are calculated by

$$M_i = \frac{(Y_{i+1} - Y_i)}{(X_{i+1} - X_i)} \tag{11}$$

$$B_i = Y_i - M_i X_i \tag{12}$$

for  $i = 1, \ldots$ , number of points + 1.

3.3.3 Calculation of the new intersection points. The initial CP is iteratively trimmed down by intersecting it with the subsequent PAs. To that end, intersections between the CP line segments and the SLs delimiting the PAs are determined. For each line segment composing the  $\mathrm{CP}_{\theta-\Delta\theta}$ , intersections with the two  $\mathrm{SLs}_{\theta}$  are evaluated as

$$X_{\text{inter}_{ij}} = \frac{(B_i - B_j)}{(M_i - M_i)} \tag{13}$$

$$Y_{\text{inter}_i} = M_j X_{\text{inter}_i} + B_j \tag{14}$$

for  $i=1,\ldots,2$ , for  $j=1,\ldots$ , number of line segments composing the  $\mathrm{CP}_{\theta-\Delta\theta}$ .

- 3.3.4 Selection of the intersection points limiting the  $CP_{\theta}$ . Only the new intersection points lying on the line segments composing the  $CP_{\theta-\Delta\theta}$  are kept. These points are identified by comparing  $X_{\text{inter}}$  and  $Y_{\text{inter}}$  with the limit coordinates of the line segments, i.e., the CP vertices. As for the points defining the  $CP_{\theta-\Delta\theta}$ , only those lying within the PA $_{\theta}$  are kept. The MATLAB (version 5.3, The MathWorks inc, Natick, MA, U.S.A.) function inpolygon can easily accomplish this task. In order to correctly define the  $CP_{\theta}$ , the selected intersection points need to be sorted out. They must be ordered in a clockwise or anti-clockwise manner. For a convex section, ordering can be performed by the MATLAB function convhull.
- 3.4 Area calculation. The PP area is calculated with the MAT-LAB command *polyarea*.
- 4 Volume Reconstruction. Once all the PPs are completed along the z axis, they can be examined in a three-dimensional space [16] using the plot3 function in MATLAB. In addition, the 3D volume can be visualized by positioning the cross sections along the z axis and using the surf function in MATLAB (Fig. 9). Finally, geometric 3D models can be generated with many commercial computer assisted design (CAD) software [e.g., sweep function in

**Fig. 9 Volume visualization of a ligament-like cell-seeded collagen matrice produced in vitro. Cross-sectional profiles were reconstructed from data measured at 10-deg intervals, and the volume, from 9 sections obtained at 4-mm intervals along the specimen.**

ProEngineer ~PTC, Needham, MA, U.S.A.!#. It is worth noting that these software generally require an equal number of points for each section and, often, a uniform angular distribution of the points around the cross section. Therefore, resampling may be necessary using, for instance, the *interp1q* function in MATLAB. As a matter of fact, although the laser beam system provides the same number of measurements per profile, the reconstruction algorithm proposed does not always maintain the same number of points per profile, that issue being more important when the profile contains acute angles. This problem is intimately associated with the noise inherent in the measurements.

### **Validation**

The system performance was evaluated using three steel bars, i.e., a 12.68 mm diameter cylinder, a 3.18 mm diameter cylinder, and a 9.28 mm side square bar whose faces were aligned about parallel to the global reference system. Lower and upper edges of the calibration bars were measured at angle increments D<sup>u</sup> of 10, 5, 3, and 1 deg in order to investigate their influence on reconstruction accuracy. For each bar and each angle increment, the profile at *z*515 mm was measured and reconstructed 10 times. Accuracy was evaluated by comparing the variations between the reconstructed and mechanically measured areas, using a hand-held caliper

$$Error = \left(\frac{\text{reconstructed area}}{\text{measured area}} - 1\right) \times 100\% \tag{15}$$

The capacity of the system to measure the 3D geometry of reconstructed collagen matrices was assessed using acellular collagen matrices @17#, acellular dehydrated/rehydrated collagen matrices @17#, and cell-seeded collagen matrices @18#. Briefly, acellular collagen matrices were produced from a Dulbecco's Modified Eagle Medium ~DMEM! and type I collagen solution. Polymerization of the collagen occurred between two bone posts fixed at the top and the bottom of a plastic tube. Acellular dehydrated/rehydrated collagen matrices were produced similarly, but were further completely dehydrated under vacuum at room temperature and rehydrated. Cell-seeded matrices were produced from a solution containing DMEM, type I collagen, fetal calf serum, and living human ligament fibroblasts.

#### **Results and Discussion**

The laser micrometer @8# is possibly the most precise and repeatable method to evaluate cross-sectional areas of soft connective tissues in vitro @15# since it is contactless and operator independent. For tissue engineered ligament- or tendonlike structures, which usually exhibit convex profiles, the laser micrometer is adequate. However, the system and the proposed reconstruction algorithm cannot handle indentations. Other techniques are best suited to detect concavities @3,4,6,7,9,11–14# which, however, have their own limitations: contact with the specimen, time consuming, lower accuracy, operator dependent precision, or not adapted to volume reconstruction.

In this paper, we present a new computer-controlled laser micrometer based on the work of Lee and Woo @8#. The most important improvement carried out by our apparatus is the rotation of the laser beam about the specimen. When a specimen is rotated through 180 deg as in Lee and Woo's system, it experiences sagging due to gravity. This situation leads to an undesirable change of specimen shape and thus in profile shape and location for each angle increment Du. Combined with the very small size of our dehydrated and cell-seeded specimens ~1 to 3 mm diameter!, this change of shape would affect the COR localization ~within or outside the specimen—cf. Fig. 2 of Lee and Woo @8#! and jeopardize the cross-sectional profile reconstruction. In the newly adapted laser micrometer, the specimen remains fixed in space while the laser beam rotates about it. Sagging is still present, but is no more a concern for profile measurement.

An alternative approach to avoid sagging is to position the specimen vertically. However, reconstructed collagen matrices are very soft and fragile, and have a high water content. Therefore, under certain conditions, vertically oriented collagen matrices could damage more easily at the top extremity, i.e., where their weight is supported, compared to a horizontal orientation, where the load is shared by both anchors. This is not true, however, if the tissue orientation at the anchor is smaller than 30 deg from the horizontal. A more important reason to avoid a vertical orientation is the effect of gravity. Due to gravity, water will accumulate at the bottom extremity, inducing tissue shape deformations that are more significant than those occuring with a horizontal orientation of the specimen.

Another improvement of interest is the automation of the longitudinal (*z*) displacement of the laser beam, which allows precise positioning of the cross-sectional profiles, and thus precise threedimensional reconstruction. As a matter of fact, measurement time depends on the number of readings and relies mainly on displacement and positioning time. The laser micrometer needs about 1 min to measure one profile at 10 deg ~Du! increments, but more than 45 min to measure five profiles at 1 deg increments. For collagen matrices, increasing measurement time results in increased creep, and may therefore lead to observable sagging and shape variations over time.

Two valuable reconstruction methods are proposed, each presenting specific advantages and disadvantages as described below. The method developed by Lee and Woo @8# uses different algorithms whether the COR is located inside or outside the specimen profile ~see the Appendix!. When the COR is located outside the specimen, it must be shifted inside the specimen profile by an offsetting distance ~OS!. However, this OS varies between profiles obtained at different *z* positions so that, before performing volume reconstruction, one must accurately reposition the reconstructed profiles in the *X*– *Y* plane. The method we propose is independent of the COR location and does not affect the reconstructed profiles positioning. Furthermore, it delimits the reconstructed profiles by a minimum of points, i.e., the polygon's vertices. However, as mentioned earlier, resampling may be necessary for particular volume reconstruction situations, for instance if the number of vertices varies from one reconstructed profile to the other.

Another reconstruction algorithm taking advantage of the profile convexity was initially tested. Straight lines (*Y*5*MX*1*B*)

Fig. 10 Reconstructed profile of an Allen key obtained with the initial tested algorithm. Encircled is a magnification of the star-like artifact resulting from the laser beam precision.

obtained from  $V^L$  and  $V^U$  data envelop a convex specimen in an anti-clockwise manner (from  $\theta = 0$  to 360 deg) as seen in Fig. 7. As a result, the profile can theoretically be defined by the intersections calculated between consecutive lines. The resulting profile of a 6 mm Allen key is shown in Fig. 10. A problem occurred for small radii of curvature where the errors in the measured data (laser beam precision) modified the appearance order of the intersection points. It resulted in a star-like geometry at each corner of the Allen key profile. This problem was easily avoided for angle increments larger than 5 deg. However, for smaller increments, the algorithm often diverged. The algorithm, exploiting the intersections of parallelepipeds, was preferred for its simplicity, accuracy, and robustness.

Using this new system, the cross-sectional profiles of the calibration bars were suitably reconstructed. When  $\Delta\theta=1$  deg, the differences between the reconstructed and measured areas for the large diameter cylinder, the small diameter cylinder, and the square bar were, respectively, 0.5%, 0.9%, and 1.4% (cf. Table 1). For cylinder bars, the error decreases with increasing bar size. This logical error variation depends on the laser beam resolution, which is equal for both cases. For a cylinder bar of intermediate diameter, Lee and Woo [8] obtained a smaller error. This can be explained by the difference of laser beam resolutions (Lee and Woo: 1  $\mu$ m; this current study: 10  $\mu$ m). Because of its shape, the square bar cross-sectional profile evaluation should be more precise than that of a circle (which is overestimated to a polygon). It should be noted, however, that the measurement error for the

Table 1 Table showing the measured and reconstructed cross-sectional areas of the calibration bars for angular increments  $\Delta\theta$  of 1 deg

| Geometric shape             | Mechanically<br>measured area<br>(mm <sup>2</sup> ) | Reconstructed area (mm <sup>2</sup> ) | Standard<br>deviation | Error (%) |
|-----------------------------|-----------------------------------------------------|---------------------------------------|-----------------------|-----------|
| Large Diameter<br>Cylinder  | 126.3                                               | 126.9                                 | 0.017                 | 0.47      |
| Small Diameter<br>Cylinder  | 7.9                                                 | 8.0                                   | 0.005                 | 0.82      |
| Square Bar                  | 86.1                                                | 87.5                                  | 0.065                 | 1.44      |
| Lee and Woo<br>Cylinder [8] | 78.5                                                | 78.6                                  | •••                   | 0.13      |

Fig. 11 Illustration of the nomenclature used by Lee and Woo [8] adapted here to the new apparatus. The offset (OS) and offsetting angle  $(\gamma)$  are obtained from data measured at rotation angles of 0 deg (A) and 90 deg (B). They are used to calculate  $R_1$  and  $R_2$  when the COR is located outside of the specimen profile.

square bar case depends on its orientation. The larger error obtained for the square bar can be mathematically explained by a minimal misalignment (<0.5 deg) of the square bar faces with the global reference system. When  $\Delta\,\theta\!=\!5$  deg, reconstruction differences were still below 5% for all three bars. At low resolution, the sharp corners may produce large errors. Fortunately, tissue engineered soft tissues generally exhibit ellipse-like contour or curved edges.

The cross-sectional profiles of acellular dehydrated/rehydrated and cell seeded collagen matrices were successfully reproduced (Fig. 9). Nevertheless, we experienced problems when measuring acellular reconstituted collagen matrices. We initially believed that the laser refracted because of the high water content of the collagen matrices. However, immersion of these matrices in physiologic solution during the measurement did not solve the problem. The collagen density in the acellular reconstituted collagen matrices appears to be too low to interrupt the laser beam. Matrices that were allowed to dry at room temperature and atmospheric pressure for only 2 min could be measured both in open air as well as immersed in a physiologic solution. They had reduced diameters and showed more opacity. We suggest that the diameter of the acellular reconstituted collagen matrices should instead be measured with a calibrated camera system.

As mentioned by Woo et al. [15], the variations in shape and area add to our current knowledge of soft tissues. For example, they could be correlated with the tissue's ultrastructure [15] or be used to calculate maximal stress, mean stress, and stress at the site of failure. Time changes in the overall tissue reconstructed geometry could also be key information for studies investigating structural changes induced by microdamaging or tissue contraction in cell-seeded collagen matrices. Furthermore, precise three-dimensional reconstruction generates accurate geometric models for FEA. Since the laser micrometer can operate on immersed tissues, it leaves the way open to new applications. For example, it could be used to evaluate changes of the specimen's width over long culture periods. However, it is to be noted that immersed

measurements are very challenging because any small bubbles, fragments, fibers, or scratches in the laser path can jeopardize the measurements. Further investigations are necessary to develop an adequate container allowing multiple scans around the specimen.

#### Acknowledgment

This work was supported by IRSC Grant No. 49478, by NSERC as a postdoctoral fellowship to Eve Langelier, and by FCAR as a scholarship to Daniel Dupuis.

#### **Appendix**

The reconstruction method developed by Lee and Woo [8] proposes two algorithms depending on whether the COR is located inside or outside the specimen profile. When the COR is located outside the specimen, the offsetting distance (OS) to shift the COR inside the specimen profile must be evaluated (Fig. 11). With the new apparatus presented in this paper, the equations proposed to calculate the OS and the profile edges become inappropriate (see Eq. (4)–(5) and Fig. 5 of Lee and Woo [8]). The following equations should be used instead:

$$DS = \sqrt{\left(RD(0) + \frac{PW(0)}{2} - COR\right)^2 + \left(RD(90) + \frac{PW(90)}{2} - COR\right)^2}$$
(A1)

$$\gamma = \tan^{-1} \left( \frac{RD(0) + \frac{PW(0)}{2} - \text{COR}}{RD(90) + \frac{PW(90)}{2} - \text{COR}} \right)$$
(A2)

$$R_1(\theta) = |C - RD(\theta) + OS^* \sin(\gamma - \theta)| \tag{A3}$$

$$R_2(\theta) = PW(\theta) - R_1(\theta) \tag{A4}$$

#### References

- Lanir, Y., Salant, E. L., and Foux, A., 1988, "Physico-Chemical and Microstructural Changes in Collagen Fiber Bundles Following Stretch in-Vitro," Biorheology, 25(4), pp. 591–603.
- [2] López Valle, C. A., Auger, F. A., Rompré, P., Bouvard, V., and Germain, L., 1992, "Peripheral Anchorage of Dermal Equivalents," Br. J. Dermatol., 127(4), pp. 365–371.
- [3] Basset, O., Gimenez, G., Mestas, J. L., Cathignol, D., and Devonec, M., 1991, "Volume Measurement by Ultrasonic Transverse or Sagital Cross-Sectional Scanning," Ultrasound Med. Biol., 17(3), pp. 291–296.
- [4] Chan, S. S., Livesay, G. A., Morrow, D. A., and Woo, S. L.-Y., 1995, "The Development of a Low-Cost Laser Reflectance System to Determine the Cross-Sectional Shape and Area of Soft Tissues," 1995 Advances in Bioengineering, ASME, New York, 31, pp. 123–124.
- [5] Ellis, D. G., 1969, "Cross-Sectional Area Measurements for Tendon Specimens: A Comparison of Several Methods," J. Biomech., 2, pp. 175–186.
- [6] Gillis, C., Sharkey, N., Stover, S. M., Pool, R. R., Meagher, D. M., and Willits, N., 1995, "Ultrasonography as a Method to Determine Tendon Cross-sectional Area," Am. J. Vet. Res., 56(10), pp. 1270–1274.
- [7] Iaconis, F., Steindler, R., and Marinozzi, G., 1987, "Measurements of Cross-Sectional Area of Collagen Structures (Knee Ligaments) by Means of Optical Method," J. Biomech., 20(10), pp. 1003–1010.

- [8] Lee, T. Q., and Woo, S. L.-Y., 1988, "A New Method for Determining Cross-Sectional Shape and Area of Soft Tissues," ASME J. Biomech. Eng., 110(2), pp. 110–114.
- [9] Love, C. L., Korvick, D. L., Lanctot, D. R., Agrawal, C. M., and Athanasiou, K. A., 1998, "Design, Validation, and Application of a System to Measure Cross-Sectional Area in the Rat Achilles Tendons," 17th Southern Biomedical Engineering Conference, pp. 25.
- [10] Njus, G. O., and Njus, N. M., 1986, "A Noncontact Method for Determining Cross-Sectional Area of Soft Tissues," Transactions of the 32nd Meeting, Orthopaedic Research Society, 11, pp. 126.
- [11] Noguchi, M., Kitaura, T., Ikoma, K., and Kusaka, Y., 2002, "A Method of In-Vitro Measurement of the Cross-Sectional Area of Soft Tissues, Using Ultrasonography," J. Orthop. Sci., 7(2), pp. 247–251.
- [12] Race, A., and Amis, A. A., 1996, "Cross-Sectional Area Measurement of Soft Tissue. A New Casting Method," J. Biomech., 29(9), pp. 1207–1212.
- [13] Shrive, N. G., Lam, T. C., Damson, E., and Frank, C. B., 1988, "A New Method of Measuring the Cross-Sectional Area of Connective Tissue Structures," ASME J. Biomech. Eng., 110(2), pp. 104-109.
- [14] Vanderby, Jr., R., Masters, G. P., Bowers, J. R., and Graf, B. K., 1991, "A Device to Measure the Cross-Sectional Area of Soft Connective Tissues," IEEE Trans. Biomed. Eng., 38(9), pp. 1040–1042.
- [15] Woo, S. L.-Y., Danto, M. I., Ohland, K. J., Lee, T. Q., and Newton, P. O., 1990, "The Use of a Laser Micrometer System to Determine the Cross-Sectional Shape and Area of Ligaments: A Comparative Study With two Existing Methods," ASME J. Biomech. Eng., 112(4), pp 426–431.
- [16] Harner, C. D., Livesay, S., Kashiwaguchi, H., Fujie, H., Choi, N. Y., and Woo, S. L.-Y., 1995, "Comparative Study of the Size and Shape of Human Anterior and Posterior Cruciate Ligaments," J. Orthop. Res., 13, pp. 429–434.
- [17] Dupuis, D., 2002, "Déshydratation de matrices collagéniques reconstruites in vitro: Effets sur les propriétés mécaniques et histologiques," Ph.D. thesis, Université Laval. Québec.
- [18] Goulet, F., Rancourt, D., Cloutier, R., Germain, L., Poole, A. R., and Auger, F. A., 2000, *Principles of Tissue Engineering*, Academic Press, San Diego, Chap. 50, pp. 711–722.