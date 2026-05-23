#### **RESEARCH PAPER**

# Topology optimization of tension-only cable nets under finite deformations

Emily D. Sanders 1 · Adeildo S. Ramos Jr. 2 · Glaucio H. Paulino 1 10

Received: 12 August 2019 / Revised: 6 December 2019 / Accepted: 17 January 2020 / Published online: 8 April 2020 © Springer-Verlag GmbH Germany, part of Springer Nature 2020

#### **Abstract**

Structures containing tension-only members, i.e., cables, are widely used in engineered structures (e.g., suspension and cable-stayed bridges, tents, and bicycle wheels) and are also found in nature (e.g., spider webs). We seek to use the ground structure method to obtain optimal cable network configurations. The structures are modeled using principles of nonlinear elasticity that allow for large displacements, i.e., global configuration changes, and large deformations. The material is characterized by a hyperelastic constitutive relation in which the strain energy is nonzero only when the axial stretch of a member is greater than or equal to one (i.e., tension-only behavior). We maximize the stationary potential energy of the equilibrated system, which avoids the need for an additional adjoint equation in computing the derivatives needed for the solution of the optimization problem. Several examples demonstrate the capabilities of the proposed formulation for topology optimization of cable networks. Motivated by nature, a spider web—inspired cable net is designed.

**Keywords** Tension-only cable nets · Topology optimization · Ground structure method · Finite deformations

## 1 Introduction and approach

Frei Otto's visionary use of tensile components (e.g., cable nets and membranes) pioneered design and construction of lightweight structures and continues to influence minimal design today (Otto and Trostel 1967; Otto and Schleyer 1969; Glaeser 1972; Otto and Rasch 1995; Nerdinger 2005). Inspired by Otto's work, we present a formulation for topology optimization of structures composed of tension-only members, i.e., cable structures, that may undergo large displacements and deformations. Seeking cable networks of maximum stiffness and limited total material volume,

Responsible Editor: Xu Guo

Dedicated to the memory of Dr. Frei Otto (1925-2015)

- ☐ Glaucio H. Paulino glaucio.paulino@ce.gatech.edu
- School of Civil and Environmental Engineering, Georgia Institute of Technology, 790 Atlantic Drive, Atlanta, GA 30332, USA
- <sup>2</sup> Laboratory of Scientific Computing and Visualization Technology Center, Federal Univerity of Alagoas, Maceió, AL 57092-970, Brazil

we maximize the stationary potential energy subject to a volume constraint (Klarbring and Strömberg 2012):

<span id="page-0-0"></span>
$$\min_{\mathbf{A}} f(\mathbf{A}) = -\Pi_{\min}(\mathbf{A}, \mathbf{u}(\mathbf{A}))$$
 (1)

s.t. 
$$g(\mathbf{A}) = \mathbf{L}^T \mathbf{A} - V^{\max} \le 0$$
 (2)

$$0 \le A_i \le A_i^{\max} \tag{3}$$

with 
$$\mathbf{u}(\mathbf{A}) = \underset{\mathbf{u}}{\operatorname{arg min}} \Pi(\mathbf{A}, \mathbf{u}(\mathbf{A}))$$
 (4)

In (1)–(4),  $\bf A$  is the vector of design variables representing the cross-sectional areas of the cable members in the undeformed configuration,  $\bf u$  is the vector of nodal displacements,  $\bf \Pi$  is the total potential energy of the system,  $\bf \Pi_{min}$  is the stationary potential energy of the system,  $\bf L$  is the vector of cable member lengths in the undeformed configuration,  $V^{max}$  is a limit on the total volume of the cable network in the undeformed configuration, and  $A_i^{max}$  is the upper bound on the undeformed cross-sectional area of member i. Note that the cable cross-sectional areas are allowed to reduce to zero and the resulting singular system of equilibrium equations is solved using a damped Newton method (Madsen and Nielsen 2010).

The formulation in (1)–(4) is applicable to structures with both linear and nonlinear elastic material behavior as well as both small and large displacements and deformations. In this

<span id="page-1-1"></span>Fig. 1 Hyperelastic constitutive model for cable members. a Strain energy density function. b Corresponding axial stress-strain curve for a linear strain measure

work, we consider large displacement and large deformation kinematics and distinguish tension-only behavior through the selected hyperelastic constitutive model with strain energy density function:

<span id="page-1-0"></span>
$$\Psi_i = \begin{cases} \frac{E_i}{2} \left[ \varepsilon_i \left( \lambda_i \right) \right]^2 & \text{if } \lambda_i \ge 1\\ 0 & \text{otherwise} \end{cases}$$
 (5)

In (5),  $\Psi_i$  is the stored strain energy per unit volume of cable member i (see Fig. 1),  $E_i$  is a material parameter relating stress to strain for cable member i,  $\varepsilon_i$  is the axial strain of cable member i, and  $\lambda_i$  is the axial stretch of cable member i. When coupled with a linear strain measure and assuming uniaxial strain (i.e., no transverse deformation), the hyperelastic strain energy density function selected here leads to a piecewise linear relation between Cauchy stress and strain. The hyperelastic strain energy density function and stress-strain curves are illustrated in Fig. 1a and b, respectively.

The proposed formulation often leads to nonintuitive results. For example, the expected solution for a problem considering a pair of self-equilibrated compression loads (Fig. 2a) is a single bar in compression (Fig. 2b). Using our tension-only material model, this compression-only solution is not a feasible design. However, the single-member, self-equilibrated topology is in the feasible space, and in fact, is a

solution in the case of a tension-only design space (Fig. 2c). A similar example is explored further in Section 8.1.

## 2 Motivation and background

The ground structure method, developed by Dorn et al. (1964), is a numerical technique that uses mathematical programming to extract optimal trusses, i.e., trusses that approximate Michell solutions (Michell 1904), from dense truss networks. Although shown to be extremely efficient for finding minimum volume trusses with bounded member stresses (Gilbert and Tyas 2003; Sokół 2011, 2015; Zegard and Paulino 2014, 2015), the plastic formulation, which enforces only nodal equilibrium, is limited in the scope of design problems that it can address (e.g., material and geometric nonlinearities cannot be handled). Using the elastic formulation (see, e.g., Christensen and Klarbring (2008)), which considers compatibility and constitutive relations in addition to equilibrium, Ramos Jr. and Paulino (2015), Zhang et al. (2017), and Zhang et al. (2018) used the ground structure method to design stiff truss systems composed of nonlinear materials.

Adopting the bi-linear material model used by Zhang et al. (2017) for design of optimal trusses with different stiffnesses in the tension and compression members, we

<span id="page-1-2"></span>Fig. 2 Self-equilibrated solutions. a Domain, ground structure, and boundary conditions. b Self-equilibrated compression structure obtained using a linear model. c Self-equilibrated tension structure

(displayed in the undeformed configuration) obtained using our nonlinear model with tension-only constitutive relation. (color online)

seek to design structures with members that have a finite stiffness in tension and zero stiffness in compression (i.e., cable structures). However, in order to ensure that we obtain equilibrated structures when no members can take compression forces, we consider finite displacements and deformations, in addition to the nonlinear material model. For example, in Fig. [3,](#page-2-0) we design a simply supported beam subjected to a midspan point load on its top surface. When considering the cable material model illustrated in Fig. [1,](#page-1-1) but only small displacements and deformations, we cannot obtain any solution since a tensiononly configuration does not exist for the given ground structure and boundary conditions (Fig. [3b](#page-2-0)). In contrast, if we consider the cable material model and include finite displacements and deformations, we obtain a tension-only structure that reaches an equilibrium configuration in the deformed shape (Fig. [3c](#page-2-0) and d). This example provides the motivation for our work.

A number of researchers have explored topology optimization with nonlinear elastic material behavior and finite displacements and/or finite deformations, but most have focused their efforts in the continuum setting (Neves et al. [1995;](#page-20-13) Jog [1996;](#page-19-5) Buhl et al. [2000;](#page-19-6) Sekimoto and Noguchi [2001;](#page-20-14) Gea and Luo [2001;](#page-19-7) Jung and Gea [2004;](#page-19-8) Yoon and Kim [2005;](#page-20-15) Kemmler et al. [2005;](#page-19-9) Kawamoto [2009;](#page-19-10) Klarbring and Stromberg ¨ [2013;](#page-20-16) Wang et al. [2014;](#page-20-17) Gomes and Senne [2014;](#page-19-11) van Dijk et al. [2014;](#page-20-18) Luo et al. [2015;](#page-20-19) Luo and Tong [2016\)](#page-20-20). In contrast, we seek to optimize the layout of cable networks. Nevertheless, our problem shares some of the same challenges faced in the continuum setting, while others are inherently avoided. One challenge that arises when nonlinearities are included in the analysis is how to define the objective function for maximum stiffness structures. Buhl et al. [\(2000\)](#page-19-6) confirmed that minimizing the end-compliance for a fixed load leads to structures that are inefficient for different load magnitudes. As a remedy, they minimized the weighted sum of end-compliance for multiple loads and ultimately found that minimizing the complementary elastic work was the most effective way to ensure that the structure could withstand all loads traversed by the load-displacement curve. In addition to these objectives, Kemmler et al. [\(2005\)](#page-19-9) also minimize the strain energy of the final structure and maximize end-stiffness, which corresponds to the tangent of the load-displacement diagram. All of these objective functions require solution of an extra adjoint equation. Du et al. [\(2019\)](#page-19-12) recently derived the sensitivities for compliance considering bi-linear materials that satisfy a scaling property without the need for an extra adjoint solve. Further, Klarbring and Stromberg ¨ [\(2013\)](#page-20-16) show that no extra adjoint equation is needed in the case of maximizing the total stationary potential energy, which is equivalent to minimizing compliance for linear problems, minimizing complementary energy for material nonlinear problems (Ramos Jr. and Paulino [2015\)](#page-20-10), and minimizing complementary energy for cable networks under finite deformations (Kanno and Ohsaki [2003\)](#page-19-13). For this reason, the total potential energy objective was adopted for optimization of trusses with nonlinear material behavior by Ramos Jr. and Paulino [\(2015\)](#page-20-10) and Zhang et al. [\(2017\)](#page-20-11) and Zhang et al. [\(2018\)](#page-20-12), and is also adopted in this work for optimal layout design of tension-only (cable) structures under finite displacements and deformations.

<span id="page-2-0"></span>**Fig. 3** Illustration of the need for large displacement kinematics with the use of the cable material model. **a** Domain, ground structure, and boundary conditions. **b** Unbounded solution for the cable material model with small displacement kinematics. **c** Undeformed and **d**

deformed solution for the cable material model with large displacement kinematics in which equilibrium is achieved in the deformed configuration. (color online)

Another challenge in topology optimization problems that consider large displacements and deformations is the possibility of critical points in the equilibrium path (e.g., buckling, snap-through, and snap-back behavior), which require special techniques (e.g., arc-length methods) to traverse the full nonlinear load-displacement curve (Sekimoto and Noguchi 2001; Kemmler et al. 2005; Leon et al. 2011). However, tension-only structures designed using our cable topology optimization formulation inherently avoid these situations since, by construction, the selected strain energy density function, which does not allow compression, prevents limit points in the equilibrium path. Thus, standard Newton-Raphson iterations are sufficient to capture the desired behavior. Another difficulty avoided in our problem is handling the numerical difficulties (non-convergence) caused by excessive deformations in low-density regions of the domain, which may lead to an indefinite or negative definite tangent stiffness matrix (Buhl et al. 2000). Many authors (Neves et al. 1995; Yoon and Kim 2005; van Dijk et al. 2014; Wang et al. 2014; Luo et al. 2015; Luo and Tong 2016) have circumvented the problem in various ways. For example, Wang et al. (2014) interpolate between the nonlinear and linear strain energy density functions so that the low-density elements behave linearly and avoid numerical issues. Our formulation naturally avoids this problem since excessive deformations of 1D elements with the selected strain energy density function does not pose a concern. In fact, we show that the tangent stiffness matrix is guaranteed to be positive semi-definite, which, in addition to preventing numerical difficulties in solution of the nonlinear equilibrium equations, also has implications on the convexity of the optimization problem (Ramos Jr. and Paulino 2015; Zhang et al. 2017).

In the following, we detail our ground structure-based topology optimization formulation for cable structures undergoing possibly finite displacements and deformations. Large deformation kinematics for the case of 1D truss members is presented in Section 3 and the piecewise linear, tension-only constitutive model used to describe the cable members is presented in Section 4. In Section 5, we derive the nonlinear and linearized equilibrium equations describing our system and provide an expression for the tangent stiffness matrix. The optimization formulation is detailed in Section 6 along with a derivation of the sensitivities and a brief discussion on convexity and optimality conditions for our problem. In Section 7, we discuss some aspects related to the numerical implementation, including solving the possibly singular system of equilibrium equations using a damped Newton solution scheme with line search, the design variable update, and the maximum end filter used to remove unnecessary thin members remaining at convergence. Finally, in Section 8, we present four numerical examples that highlight the key features of the cable formulation. Additionally, Appendix A includes a full derivation of the element tangent stiffness matrix, Appendix B elaborates on positive semi-definiteness of the element tangent stiffness matrix, Appendix C provides the full damped Newton and line search algorithms, Appendix D includes additional details on the optimality criteria design variable update scheme, Appendix E investigates the potential use of other strain measures, and Appendix F defines the nomenclature used throughout the manuscript.

## <span id="page-3-0"></span>3 Large deformation kinematics

The kinematics considered here is similar to that of the uniaxial, large displacement, large deformation, pin-jointed truss member provided by Bonet and Wood (2008) (see Fig. 4). In what follows, we use the terminology "large deformations" to encompass both large displacements and finite strains. We adopt the notation in which upper-case letters are used to describe the undeformed configuration and lower-case letters are used to describe the deformed configuration. As such, the positions of cable member *i* are described for the undeformed and deformed configurations, respectively, as

$$\mathbf{X}_{i} = \sum_{j=1}^{n_{d}} X_{i}^{j} \mathbf{E}^{j} ; \mathbf{x}_{i} = \sum_{j=1}^{n_{d}} x_{i}^{j} \mathbf{e}^{j}$$
 (6)

where  $n_d$  is the number of spatial dimensions and we have defined the coordinate frames for the undeformed  $(\mathbf{E}^j, j = 1, \dots, n_d)$  and deformed  $(\mathbf{e}^j, j = 1, \dots, n_d)$  configurations to coincide. Also, in Fig. 4, we have introduced the displacement,  $\mathbf{u}_i = \sum_{j=1}^{n_d} u_j^j \mathbf{E}^j$ , of member i. For later

<span id="page-3-1"></span>**Fig. 4** Kinematics for 3D cable member i (Bonet and Wood 2008)

use in our solution scheme, we also define incremental displacements  $\Delta \mathbf{u}_{i}^{p}$  and  $\Delta \mathbf{u}_{i}^{q}$  at ends p and q of deformed member i.

Based on the coordinates of ends P(p) and Q(q) of the member, the lengths of member i are computed as

$$L_{i} = \left[ \left( \mathbf{X}_{i}^{Q} - \mathbf{X}_{i}^{P} \right) \cdot \left( \mathbf{X}_{i}^{Q} - \mathbf{X}_{i}^{P} \right) \right]^{1/2};$$

$$\ell_{i} = \left[ \left( \mathbf{x}_{i}^{q} - \mathbf{x}_{i}^{p} \right) \cdot \left( \mathbf{x}_{i}^{q} - \mathbf{x}_{i}^{p} \right) \right]^{1/2}$$
(7)

where  $L_i$  and  $\ell_i$  denote the lengths of member i in the undeformed and deformed configurations, respectively. Similarly, the orientations of member i in the undeformed and deformed configurations, respectively, are defined by unit vectors along their axes:

$$\mathbf{N}_{i} = \frac{\mathbf{X}_{i}^{Q} - \mathbf{X}_{i}^{P}}{L_{i}}; \mathbf{n}_{i} = \frac{\mathbf{x}_{i}^{q} - \mathbf{x}_{i}^{P}}{l_{i}}$$
(8)

We assume uniaxial strain such that the transverse principal stretches,  $\lambda_i^2$  and  $\lambda_i^3$ , are unity and the fundamental measure of deformation in each cable member is the axial stretch,  $\lambda_i^1 = \lambda_i = \ell_i/L_i$ . The Jacobian,  $J_i = \lambda_i \lambda_i^2 \lambda_i^3 = \lambda_i$ , gives the ratio of the volume in the deformed configuration to the volume in the undeformed configuration:

$$dv_i = J_i dV_i \to J_i = \frac{v_i}{V_i} = \frac{a_i \ell_i}{A_i L_i} \tag{9}$$

where the cross-sectional areas of member i are denoted  $A_i$  and  $a_i$  in the undeformed and deformed configurations, respectively, and the volumes of member i are denoted  $V_i = A_i L_i$  and  $v_i = a_i \ell_i$  in the undeformed and deformed configurations, respectively. Note that since  $J_i = \lambda_i$ , the deformed and undeformed cross-sectional areas are equal (i.e.,  $a_i = A_i$ ).

We define a measure of instantaneous strain in member *i* as the ratio of the instantaneous change in member length to the original member length:

$$d\varepsilon_i = \frac{d\ell_i}{L_i} \tag{10}$$

from which we derive our linear strain measure<sup>1</sup> by integrating over the entire deformation from  $L_i$  to  $\ell_i$ :

$$\varepsilon_i = \int_{L_i}^{\ell_i} d\varepsilon_i = \lambda_i - 1 \tag{11}$$

## <span id="page-4-0"></span>4 Cable constitutive model

Defining  $t_i = \sigma_i a_i$  as the magnitude of the internal force in member i, where  $\sigma_i$  is the axial component of Cauchy stress

in member i, we can compute the stored strain energy per unit volume in member i as

$$\Psi_{i} = \frac{1}{V_{i}} \int_{L_{i}}^{\ell_{i}} t_{i} dl_{i} = \int_{0}^{\varepsilon_{i}} \sigma_{i} \frac{v_{i}}{V_{i}} \frac{L_{i}}{\ell_{i}} d\varepsilon_{i} = \int_{0}^{\varepsilon_{i}} \sigma_{i} \frac{J_{i}}{\lambda_{i}} d\varepsilon_{i}$$

$$\tag{12}$$

<span id="page-4-4"></span>which defines our constitutive relationship to be

$$\sigma_i = \frac{\lambda_i}{J_i} \frac{\partial \Psi_i}{\partial \lambda_i} = \frac{\partial \Psi_i}{\partial \lambda_i} \tag{13}$$

where we have used the facts that the axial strain defined in (11) is a linear function of the axial stretch and  $J_i = \lambda_i$ .

Introducing a material parameter,  $E_i$ , which turns out to be the standard Young's modulus of linear elasticity, we choose the following strain energy density function:

$$\Psi_i = \begin{cases} \frac{E_i}{2} (\lambda_i - 1)^2 & \text{if } \lambda_i \ge 1\\ 0 & \text{otherwise} \end{cases}$$
 (14)

<span id="page-4-5"></span>It is noted that  $\Psi_i$  is a function of both  $L_i$  and  $\ell_i$ , and thus, is a path-independent hyperelastic constitutive model. Then, according to (13), the Cauchy stress is expressed as

$$\sigma_i = \begin{cases} E_i \ (\lambda_i - 1) & \text{if } \lambda_i \ge 1\\ 0 & \text{otherwise} \end{cases}$$
 (15)

Although the constitutive relation in (15) is non-smooth, in general, it does not cause problems in the optimization of tension-only cable structures (Klarbring and Rönnqvist 1995).

### <span id="page-4-1"></span>5 Nonlinear equilibrium equations

Equilibrium of the cable network is enforced by requiring the total potential energy,  $\Pi(\mathbf{u})$ , to be stationary. We write  $\Pi(\mathbf{u})$  as the sum of internal strain energy,  $U(\mathbf{u})$ , and potential of externally applied loads,  $\Omega(\mathbf{u})$ , where

<span id="page-4-6"></span>
$$U\left(\mathbf{u}\right) = \sum_{i=1}^{N} \int_{V_i} \Psi_i\left(\mathbf{u}\right) dV = \sum_{i=1}^{N} V_i \Psi_i\left(\mathbf{u}\right)$$
 (16)

<span id="page-4-7"></span>and

<span id="page-4-3"></span>
$$\Omega\left(\mathbf{u}\right) = -\mathbf{F}^{T}\mathbf{u} \tag{17}$$

In (16) and (17), N is the number of cable members in the model and  $\mathbf{F}$  is the vector of external loads. Then,

<span id="page-4-8"></span>
$$\frac{\partial \Pi \left(\mathbf{u}\right)}{\partial \mathbf{u}} = 0 = \frac{\partial U \left(\mathbf{u}\right)}{\partial \mathbf{u}} + \frac{\partial \Omega \left(\mathbf{u}\right)}{\partial \mathbf{u}}$$

$$= \sum_{i=1}^{N} V_{i} \frac{\partial \Psi_{i} \left(\mathbf{u}\right)}{\partial \mathbf{u}} - \frac{\partial \left(\mathbf{F}^{T} \mathbf{u}\right)}{\partial \mathbf{u}}$$

$$= \sum_{i=1}^{N} V_{i} \frac{\partial \Psi_{i} \left(\mathbf{u}\right)}{\partial \lambda_{i}} \frac{\partial \lambda_{i}}{\partial \mathbf{u}} - \mathbf{F}$$
(18)

<span id="page-4-2"></span><sup>&</sup>lt;sup>1</sup>Other strain measures are explored in Appendix E

Using (13), the derivative of the stored strain energy function in (18) can be written in terms of the Cauchy stress and we only need to compute

<span id="page-5-1"></span>
$$\frac{\partial \lambda_{i}}{\partial \mathbf{u}} = \frac{1}{L_{i}} \frac{\partial \ell_{i}}{\partial \mathbf{u}}$$

$$= \frac{1}{L_{i}} \frac{\partial \ell_{i}}{\partial \mathbf{x}} \frac{\partial \mathbf{x}}{\partial \mathbf{u}}$$

$$= \frac{1}{L_{i}} \frac{\partial \ell_{i}}{\partial \mathbf{x}} \frac{\partial (\mathbf{X} + \mathbf{u})}{\partial \mathbf{u}}$$

$$= \frac{1}{L_{i}} \mathbf{b}_{i} \tag{19}$$

where

$$\mathbf{b}_{i} = \left\{ \dots \frac{\partial \ell_{i}}{\partial \mathbf{x}_{p}} \dots \frac{\partial \ell_{i}}{\partial \mathbf{x}_{q}} \dots \right\}^{T} = \left\{ \dots - \mathbf{n}_{i} \dots \mathbf{n}_{i} \dots \right\}^{T}$$
(20)

and the dots indicate zeros. Plugging (15) and (19) into (18), we write the stationary condition of the potential energy as

<span id="page-5-2"></span>
$$\frac{\partial \Pi \left(\mathbf{u}\right)}{\partial \mathbf{u}} = 0 = \sum_{i=1}^{N} A_{i} \frac{J_{i}}{\lambda_{i}} \sigma_{i} \mathbf{b}_{i} - \mathbf{F}$$

$$= \sum_{i=1}^{N} A_{i} \frac{v_{i}}{V_{i}} \frac{L_{i}}{\ell_{i}} \sigma_{i} \mathbf{b}_{i} - \mathbf{F}$$

$$= \sum_{i=1}^{N} a_{i} \sigma_{i} \mathbf{b}_{i} - \mathbf{F}$$

$$0 = \mathbf{T} \left(\mathbf{u}\right) - \mathbf{F} = \mathbf{R} \left(\mathbf{u}\right)$$
(21)

where we have noted that the magnitude of the internal force in member i is  $t_i = a_i \sigma_i$  and we have defined the vector of member internal forces,  $\mathbf{T}(\mathbf{u})$ , and the vector of residual nodal forces,  $\mathbf{R}(\mathbf{u})$ . Note that  $\mathbf{T}(\mathbf{u})$  is a function of the deformed configuration, i.e., the equilibrium equations in (21) are nonlinear and thus, need to be linearized and solved iteratively.

#### <span id="page-5-6"></span>5.1 Linearization

Given a solution to (21),  $\mathbf{u}_k$  at iteration k, a new value,  $\mathbf{u}_{k+1} = \mathbf{u}_k + \Delta \mathbf{u}_k$ , is obtained in terms of an increment,  $\Delta \mathbf{u}_k$ , by establishing a linear approximation of the residual

$$\mathbf{R}(\mathbf{u}_{k+1}) = \mathbf{R}(\mathbf{u}_k) + \mathcal{D}\mathbf{R}(\mathbf{u}_k) [\Delta \mathbf{u}_k] = \mathbf{0}$$
 (22)

where the directional derivative is determined using the chain rule

<span id="page-5-3"></span>
$$\mathcal{D}\mathbf{R}\left(\mathbf{u}_{k}\right)\left[\Delta\mathbf{u}_{k}\right] = \frac{d}{d\epsilon}\left[\mathbf{R}\left(\mathbf{u}_{k} + \epsilon\Delta\mathbf{u}_{k}\right)\right]\Big|_{\epsilon=0}$$

$$= \frac{\partial\mathbf{R}}{\partial\mathbf{u}}\Big|_{\mathbf{u}_{k}}\frac{\partial(\mathbf{u}_{k} + \epsilon\Delta\mathbf{u}_{k})}{\partial\epsilon}\Big|_{\epsilon=0}$$

$$= \mathbf{K}^{t}\left(\mathbf{u}_{k}\right)\Delta\mathbf{u}_{k} \tag{23}$$

In (23), we have defined the global tangent stiffness matrix,  $\mathbf{K}^t$ , as the derivative of the unbalanced forces with respect

<span id="page-5-7"></span>to the displacement field. Now, the linear set of equations to solve at each iteration k is

$$\mathbf{K}^{t}\left(\mathbf{u}_{k}\right)\Delta\mathbf{u}_{k}=-\mathbf{R}\left(\mathbf{u}_{k}\right)=\mathbf{F}-\mathbf{T}\left(\mathbf{u}_{k}\right)\tag{24}$$

## 5.2 Tangent stiffness matrix

Since the external force vector is not a function of the deformed configuration,  $\mathbf{K}^t$  is derived as the derivative of the internal member forces with respect to the displacement field:

$$\mathbf{K}^{t} = \frac{\partial \mathbf{T}}{\partial \mathbf{u}} \bigg|_{\mathbf{u}_{k}} \tag{25}$$

<span id="page-5-4"></span>and can be assembled from the element tangent stiffness matrices:

$$\mathbf{k}_{i}^{t}(\mathbf{u}_{k}) = \begin{bmatrix} \left(\partial \mathbf{t}_{i}^{p} / \partial \mathbf{u}_{i}^{p}\right) \middle|_{\mathbf{u}_{k}} \left(\partial \mathbf{t}_{i}^{p} / \partial \mathbf{u}_{i}^{q}\right) \middle|_{\mathbf{u}_{k}} \\ \left(\partial \mathbf{t}_{i}^{q} / \partial \mathbf{u}_{i}^{p}\right) \middle|_{\mathbf{u}_{k}} \left(\partial \mathbf{t}_{i}^{q} / \partial \mathbf{u}_{i}^{q}\right) \middle|_{\mathbf{u}_{k}} \end{bmatrix} = \begin{bmatrix} \mathbf{k}_{i}^{pp} & \mathbf{k}_{i}^{pq} \\ \mathbf{k}_{i}^{qp} & \mathbf{k}_{i}^{qq} \end{bmatrix}$$
(26)

where  $\mathbf{k}_{i}^{t}(\mathbf{u}_{k})$  denotes the element tangent stiffness matrix of element *i* and the element internal force vector and element displacement vector are, respectively,

$$\mathbf{t}_{i}\left(\mathbf{u}_{i}\right) = \begin{Bmatrix} \mathbf{t}_{i}^{p} \\ \mathbf{t}_{i}^{q} \end{Bmatrix} = t_{i} \begin{Bmatrix} -\mathbf{n}_{i} \\ \mathbf{n}_{i} \end{Bmatrix} ; \mathbf{u}_{i} = \begin{Bmatrix} \mathbf{u}_{i}^{p} \\ \mathbf{u}_{i}^{q} \end{Bmatrix}$$
(27)

for member i with end nodes p and q. In (26), it is noted that  $\mathbf{k}_i^{pp} = \mathbf{k}_i^{qq} = -\mathbf{k}_i^{pq} = -\mathbf{k}_i^{qp}$ , and therefore, we can derive  $\mathbf{k}_i^t(\mathbf{u}_k)$  using only one of the four partitions (see Appendix A for the full derivation):

<span id="page-5-5"></span>
$$\mathbf{k}_{i}^{qq} = \frac{\partial \mathbf{t}_{i}^{q}}{\partial \mathbf{u}_{i}^{q}} \bigg|_{\mathbf{n}_{i}} = \frac{a_{i}}{L_{i}} \frac{\partial^{2} \Psi_{i}}{\partial \lambda_{i}^{2}} \mathbf{n}_{i} \mathbf{n}_{i}^{T} + \frac{t_{i}}{\ell_{i}} \left( \mathbf{I} - \mathbf{n}_{i} \mathbf{n}_{i}^{T} \right)$$
(28)

where **I** is an  $n_d \times n_d$  identity matrix.

## <span id="page-5-0"></span>**6 Topology optimization formulation**

We seek to maximize the structural stiffness of cable networks. To do so, we choose to maximize the stationary total potential energy (see (1)), which has been shown equivalent to minimizing end-compliance for linear problems (Klarbring and Strömberg 2012), complementary energy for nonlinear problems with small strains (Ramos Jr. and Paulino 2015), and complementary energy for cable networks under large deformations (Kanno and Ohsaki 2003). Next, we derive the sensitivities of this objective function without the need for an adjoint vector, make some remarks regarding convexity of the formulation, and state the optimality conditions.

## 6.1 Sensitivity analysis

The sensitivities of the linear constraint in (2) with respect to the design variables are

$$\frac{\partial g}{\partial A_i} = L_i, \ i = 1, \dots, N \tag{29}$$

and the sensitivities of the objective function in (1) with respect to the design variables are

<span id="page-6-1"></span>
$$\frac{\partial f(\mathbf{A})}{\partial A_{i}} = -\frac{\partial \Pi_{\min}(\mathbf{A}, \mathbf{u}(\mathbf{A}))}{\partial A_{i}} - \frac{\partial \Pi_{\min}(\mathbf{A}, \mathbf{u}(\mathbf{A}))}{\partial \mathbf{u}} \frac{\partial \mathbf{u}(\mathbf{A})}{\partial A_{i}}, i = 1, \dots, N (30)$$

Due to the equilibrium conditions, the second term in (30) goes to zero. Writing  $\Pi_{\min}$  as the sum of the internal strain energy,  $U(\mathbf{A}, \mathbf{u}(\mathbf{u}))$ , and potential of externally applied loads,  $\Omega(\mathbf{u}(\mathbf{A}))$ , and noting that  $\Omega(\mathbf{u}(\mathbf{A}))$  is (explicitly) independent of  $A_i$  (Klarbring and Strömberg 2012), we write the sensitivity of the objective as

$$\frac{\partial f(\mathbf{A})}{\partial A_{i}} = -\frac{\partial U(\mathbf{A}, \mathbf{u}(\mathbf{A}))}{\partial A_{i}} = -L_{i}\Psi_{i}(\mathbf{u}(\mathbf{A})) i = 1, \dots, N$$
(31)

Note that there is no need to calculate an adjoint vector for the selected objective function. Additionally,  $L_i\Psi_i\left(\mathbf{u}\left(\mathbf{A}\right)\right)\geq 0$ , i.e., the sensitivities of the objective are always non-positive, an observation that demonstrates a clear parallel between the current formulation for maximum stationary potential energy and that of minimum end-compliance. Note also that although the constitutive model is discontinuous at zero strain (see Fig. 1b), the sensitivity of the objective function is continuous everywhere since it depends only on the continuous strain energy density function (see Fig. 1a) and not on its derivative.

## <span id="page-6-4"></span>6.2 Convexity

In their work focusing on material nonlinearities, Ramos Jr. and Paulino (2015) proved convexity of the objective function in (1) for a positive definite tangent stiffness matrix and Zhang et al. (2017) generalized the proof to include positive semi-definite tangent stiffness matrices. Noting that the global tangent stiffness matrix is guaranteed to be positive semi-definite if the element tangent stiffness matrices from which it is assembled are positive semi-definite, they analyzed the element tangent stiffness matrix and determined that the optimization problem in (1) - (4) is convex for (non-strictly) convex hyperelastic material models. Since our tangent stiffness matrix contains a geometric stiffness term that did not exist in the previous small deformation cases, we need to investigate the criteria

needed for a positive semi-definite tangent stiffness matrix in the case of finite deformation kinematics.

Again, we study only the element tangent stiffness matrix, which is a partitioned matrix (see (26)) that satisfies some conditions that allow us to check only the lower left partition,  $\mathbf{k}_{i}^{qq}$ , for positive semi-definiteness (Kreindler and Jameson 1972). In (28),  $\mathbf{k}_{i}^{qq}$  is a sum of two terms. The first term is analogous to the small deformation case and is positive semi-definite given that strain energy density function has non-negative curvature, i.e.,  $\partial^2 \Psi / \partial \lambda_i^2 \geq 0$ . Additionally, in the second term, the matrix,  $(\mathbf{I} - \mathbf{n}_i \mathbf{n}_i^T)$ , can be shown to be positive semi-definite using the principal minors test (Strang 2006) (see details in Appendix B). Thus, for tension-only structures in which  $t_i/\ell_i \geq 0$ , the stiffness matrix is guaranteed to be positive semi-definite and we arrive at the same criterion for convexity as in the small deformation case, i.e., the optimization problem in (1) – (4) is convex for (non-strictly) convex hyperelastic material models. This criterion is satisfied for the selected strain energy density function, linear strain measure<sup>2</sup>, and uniaxial strain assumption.

## 6.3 Optimality conditions

The KKT optimality conditions for the convex optimization problem in (1)–(4) have been derived previously and are stated here for completeness (Ramos Jr. and Paulino 2015; Zhang et al. 2017):

$$\Psi_i\left(\mathbf{u}\left(\mathbf{A}^*\right)\right) \ge \phi^* \text{ if } A_i^* = A_i^{\text{max}} \tag{32}$$

<span id="page-6-3"></span>
$$\Psi_i\left(\mathbf{u}\left(\mathbf{A}^*\right)\right) = \phi^* \text{ if } 0 < A_i^* < A_i^{\text{max}}$$
(33)

$$\Psi_i\left(\mathbf{u}\left(\mathbf{A}^*\right)\right) \le \phi^* \text{ if } A_i^* = 0 \tag{34}$$

where  $\mathbf{A}^*$  and  $\phi^*$  are the solution and Lagrange multiplier, respectively, at the optimum. Note that when the box constraints are inactive at the optimal point, all cable members have equal strain energy, analogously to full stressed design in the linear case with end-compliance objective.

## <span id="page-6-0"></span>7 Details of the numerical implementation

In this section, we provide some details related to implementation of the topology optimization formulation for structures composed of tension-only members. Specifically, we review the damped Newton algorithm with line search used for efficient solution of the possibly singular nonlinear equilibrium equations, summarize the design variable

<span id="page-6-2"></span><sup>&</sup>lt;sup>2</sup>Convexity of other strain measures is explored in Appendix E

update scheme, detail how we reduce the size of the problem during the optimization, and discuss a maximum end filter used to extract clean solutions (after convergence) that satisfy global equilibrium.

## 7.1 Damped Newton with line search

Typically, Newton-Raphson iterations are used to solve the nonlinear equilibrium equations according to the linearization scheme in Section 5.1; however, due to the zero lower bound in (3), the tangent stiffness matrix may become singular, preventing us from solving (24). Thus, we adopt a damped Newton method (Madsen and Nielsen 2010) so that the linearized equilibrium equation of the nonlinear system becomes

<span id="page-7-0"></span>
$$\mathbf{K}^{t,\eta} \Delta \mathbf{u}_k = -\mathbf{R} (\mathbf{u}_k) = \mathbf{F} - \mathbf{T} (\mathbf{u}_k)$$
(35)

where  $\mathbf{K}^{t,\eta} = \mathbf{K}^t + \eta \mathbf{I}$  and  $\eta$  is the damped Newton parameter defined as  $\eta_0 \approx 10^{-12}$  to  $10^{-8}$  multiplied by the mean of the diagonal of  $\mathbf{K}^t$  (a similar regularization scheme is adopted by Ramos Jr. and Paulino (2016), Zhang et al. (2017) and Sanders et al. (2017)). The damped Newton scheme using (35) and considering a load control approach is provided in Algorithm 1 of Appendix C. It is noted that Newton methods only converge locally and depending on the constitutive model, the algorithm may diverge (Madsen and Nielsen 2010). Although our constitutive model will not lead to divergence, we seek to improve the convergence of the first several iterations of the damped Newton algorithm by taking strategically sized steps such that our update becomes (see e.g., Ascher and Greif (2011), Wriggers (2008), and Wright and Nocedal (1999)):

$$\mathbf{u}_{k+1} = \mathbf{u}_k + \xi_k \Delta \mathbf{u}_k \tag{36}$$

We use backtracking line search with an Armijo condition (inexact line search) to find the line search parameter,  $\xi_k$ , such that the Wolfe conditions are satisfied. The specific line search algorithm used in our implementation, taken from (Ascher and Greif 2011), is provided in Algorithm 2 of Appendix C.

### <span id="page-7-1"></span>7.2 Design variable update

The design variables are updated in each optimization iteration using the optimality criteria (OC) method, which is detailed in Appendix D. The OC method is characterized by a recursive update derived using Lagrangian duality of truncated Taylor approximated subproblems evaluated at intervening variable  $y_i = A_i^{\alpha_i}$ ,  $\alpha_i < 0$  (Bendsøe and Sigmund 2003; Groenwold and Etman 2008). The approximate subproblems are only accurate in a small neighborhood of the current design; thus, we impose a move limit, M, on the change in the design variables in a given

<span id="page-7-2"></span>
$$M = \gamma A_0 \tag{37}$$

where  $A_0$  is the initial cross-sectional area of each member (Ramos Jr. and Paulino 2016). Note that the recursive nature of the OC method does not allow reappearance of zero-area members (see (51) in Appendix D).

## 7.3 Convergence criterion

Convergence of the optimization problem is determined based on the change in the design variables. Specifically, the optimization algorithm is aborted when

$$\max\left(\frac{|A_i^{k+1} - A_i^k|}{1 + A_i^k}\right) \le tol \tag{38}$$

#### 7.4 Reduced order model

As noted in Section 7.2, the OC update scheme used in this work does not allow zero-area members to reappear. Thus, we reduce the design space in each iteration by removing the zero-area members (Ramos Jr. and Paulino 2016; Sanders et al. 2017; Zhang et al. 2017). The following mapping matrix **Q** is constructed, such that

$$\mathbf{u} = \mathbf{Q}\mathbf{u}^{\mathrm{Top}} \tag{39}$$

where **u**<sup>Top</sup> is the vector of nodal displacements considering only the degrees of freedom associated with the topology, i.e., the set of members in the initial ground structure that have non-null member area. Based on this mapping, we can also define the global tangent stiffness matrix, external load vector, and internal load vector associated with the topology, respectively, as

$$\mathbf{K}^{t,Top} = \mathbf{Q}^{T} \mathbf{K}^{t} \mathbf{Q} ; \mathbf{F}^{Top} = \mathbf{Q}^{T} \mathbf{F} ; \mathbf{T}^{Top} \left( \mathbf{u}_{k}^{Top} \right) = \mathbf{Q}^{T} \mathbf{T} \left( \mathbf{u}_{k} \right)$$
(40)

Then the associated linearized equilibrium equation of the nonlinear system becomes

$$\left(\mathbf{K}^{t,\text{Top}} + \eta \mathbf{I}\right) \Delta \mathbf{u}_{k}^{\text{Top}} = \mathbf{R}^{\text{Top}} \left(\mathbf{u}_{k}\right) = \mathbf{F}^{\text{Top}} - \mathbf{T}^{\text{Top}} \left(\mathbf{u}_{k}^{\text{Top}}\right)$$
(41)

with update:

$$\mathbf{u}_{k+1}^{\text{Top}} = \mathbf{u}_{k}^{\text{Top}} + \xi_{k} \Delta \mathbf{u}_{k}^{\text{Top}} \tag{42}$$

## 7.5 End filter

Although we enforce a zero lower bound, the solutions often contain thin members that can be removed with negligible effect on the structural behavior. To clean up the final

design, we adopt the maximum end filter, proposed by Sanders et al. (2017), which sets cross-sectional areas equal to zero according to the following:

$$A_{i} = \text{Filter}\left(\mathbf{A}, \alpha_{f}\right) = \begin{cases} 0 & \text{if } \frac{A_{i}}{\max(\mathbf{A})} < \alpha_{f} \\ A_{i} & \text{otherwise} \end{cases}$$
(43)

where  $\alpha_f$  is the filter value selected using a bisection algorithm to ensure that, after filtering, the final design satisfies global equilibrium and the increase in the objective is controlled.

To determine whether to accept the filtered structure for a given  $\alpha_f$ , we first check that the global equilibrium error of the filtered structure is within a tolerance,  $\rho$ , which is typically taken to be  $10^{-4}$ :

<span id="page-8-2"></span>
$$\frac{||\mathbf{R}^{\text{Top}}(\mathbf{u}_k)||}{||\mathbf{F}^{\text{Top}}||} \le \rho \tag{44}$$

<span id="page-8-3"></span>Additionally, we check that the objective value obtained after filtering,  $f_{\text{filtered}}$ , is within a margin of that obtained at convergence,  $f_{\text{converged}}$ :

$$\Delta f = \frac{\left(f_{\text{filtered}} - f_{\text{converged}}\right)}{f_{\text{converged}}} \le f^{\text{tol}} \tag{45}$$

where  $f^{\text{tol}}$  is a user-prescribed tolerance. If either of (44) or (45) is not satisfied by a filter value,  $\alpha_f$ , greater than  $A_i / \max{(\mathbf{A})} \forall i$ , then no end filter is applied.

## <span id="page-8-1"></span>8 A few examples

In this section, we present four numerical examples to illustrate the capabilities of the proposed formulation for obtaining maximum stiffness cable networks under possibly finite displacements and deformations. The first example highlights the effect of load magnitude on the final design when considering large deformation kinematics and includes a case similar to Fig. 2 in which the linear and nonlinear (cable) formulation lead to the same topology, but with very different mechanical behavior. The second example shows that when the design variable upper bound,  $A_i^{\text{max}}$ , is active, the final topology does not have constant stress or strain energy, in accordance with the KKT optimality conditions. The third example demonstrates that the cable topology optimization formulation may lead to topologies that would be meaningless in the case of small displacement kinematics, but that are well defined in the context of the current nonlinear model. Finally, we use the formulation to design spider web-inspired cable nets that have similarities to an orb-web (Vollrath and Mohren 1985). Since the problems provided are relatively small scale, we do not consider symmetry reduction, but note that convexity of the formulation implies that symmetry reduction can be used for problems with symmetric domain and boundary conditions (Guo et al. 2013; Du and Guo 2016).

The ground structure "level" reported for each example is based on the definition proposed by Zegard and Paulino (2014). As such, the ground structure is generated on a base mesh in which neighboring nodes are defined as nodes that belong to the same element in the base mesh. Then, a level 1 ground structure contains connectivity between all neighboring nodes, level 2 contains connectivity up to the neighbors of the neighbors, level 3 contains connectivity up to the neighbors of the neighbors, and so on. A full-level ground structure contains connectivity between all nodes in the base mesh. In all examples, the longer of two overlapping members in the initial ground structure is not considered. Line thicknesses in the topology plots indicate the diameter of the member normalized to the maximum member diameter, assuming a circular crosssection. Note that, in general, line thicknesses cannot be compared between structures, unless indicated otherwise. In all presented results, blue and red indicate members in tension and compression, respectively.

## <span id="page-8-0"></span>8.1 Clamped beam with equal and opposite point loads

Here, we consider a clamped beam subjected to equal and opposite compressive point loads on the top and bottom faces at midspan of the beam. A full-level ground structure consisting of 251 members with Young's modulus,  $E_i=7.5$  GPa, is generated based on a  $6\times 3$  orthogonal base mesh. The ground structure and boundary conditions are provided in Fig. 5a. In this example, an end filter with  $f^{\rm tol}=0.01$  is used to remove thin members remaining at convergence. All other optimization parameters used for this problem are provided in Table 1.

Considering small deformation kinematics and a linear material model, the expected solution, shown in Fig. 5b, is a single vertical member in compression, self-equilibrated by the two equal and opposite loads, where members connected by aligned nodes have been replaced by a single long member (see Sanders et al. 2017). In this case, the magnitude of the load P is irrelevant to obtaining the optimal design in Fig. 5b.

Three different optimal designs obtained considering large deformation kinematics with the cable material model are provided in Fig. 5c, d, and e. We show the specific sizing and deformed shapes for  $P=10~\rm kN$ ,  $P=1000~\rm kN$ , and  $P=2000~\rm kN$ , but note that the critical load that causes a transition between the topologies in Fig. 5c and d, and Fig. 5d and e is about 400 kN and 1655 kN, respectively. In this case, the topology and sizing of the stiffest structure is dependent on the magnitude of the load. Note also that all of the members in these designs are in tension, as clearly

<span id="page-9-0"></span>**Fig. 5** Clamped beam with equal and opposite compressive point loads. **a** Domain, ground structure, and boundary conditions. **b** Optimal design considering small deformation kinematics and linear material model (aligned nodes are removed in post-processing).

Undeformed (left) and deformed (right) topologies considering large deformation kinematics and the cable material model, with **c** *P* = 10 kN, **d** *P* = 1000 kN, and **e** *P* = 2000 kN. (color online)

indicated in the plots of the deformed shapes (right side of Fig. [5c](#page-9-0), d, and e). The maximum strains and stresses for the three (fully stressed) designs are provided in Table [2.](#page-10-1)

This example is similar to the one provided in the Introduction and we finalize the discussion about it here. As noted previously, depending on the load magnitude, we

<span id="page-10-0"></span>**Table 1** Optimization input parameters for the clamped beam

| Volume limit, V <sup>max</sup>        | $2.000 \times 10^{-2} \text{ m}^3$ |
|---------------------------------------|------------------------------------|
| Initial area, $A_0$                   | $3.198 \times 10^{-6} \text{ m}^2$ |
| Maximum area, $A_i^{\text{max}}$      | $1.600 \times 10^{-3} \text{ m}^2$ |
| Move parameter, $\gamma$              | $1.000 \times 10^{3}$              |
| Convergence tolerance, tol            | $1.000 \times 10^{-9}$             |
| Objective tolerance, $f^{\text{tol}}$ | $1.000 \times 10^{-2}$             |
| Damped Newton parameter, $\eta$       | $1.000 \times 10^{-8}$             |
|                                       |                                    |

can sometimes obtain an identical topology to that obtained using small deformation kinematics and a linear material model (compare the topologies in Fig. 5b and e). The difference is that in the case of finite deformations with the cable material model, the structure goes through a large configuration change to obtain its equilibrium position. Notice that in the deformed shape, the location of loaded nodes A and B flip so that tension is induced in the structure.

## 8.2 Pin-supported beam with midspan load

In this example, we study a pin-supported beam with midspan point load (100 kN) at the top surface. The domain, boundary conditions, and full-level ground structure consisting of 251 members with Young's modulus,  $E_i = 170$  GPa, are provided in Fig. 6a. Here we investigate the solutions based on two different design variable upper bounds,  $A_i^{max} = 1.444 \times 10^{-3} \text{ m}^2$  and  $A_i^{max} = 1.155 \times 10^{-4} \text{ m}^2$  for all i. An end filter with  $f^{tol} = 0.01$  is used to remove thin members remaining at convergence. All other optimization parameters used for this problem are provided in Table 3.

The solution considering small deformation kinematics and a linear material model is provided in Fig. 6b, where members connected by aligned nodes have been replaced by a single long member (see Sanders et al. 2017). The results considering large deformation kinematics and the cable material model with load P = 100 kN are provided in Fig. 6 c and d for  $A_i^{\text{max}} = 1.444 \times 10^{-3} \text{ m}^2$  and  $A_i^{\text{max}} = 1.155 \times 10^{-4} \text{ m}^2$ , respectively, for all i. Notice that the result

<span id="page-10-1"></span>**Table 2** Clamped beam maximum strains and member stresses for varying applied load magnitudes

| Load (kN) | Maximum strain (%) | Member stress (MPa) <sup>1</sup> |
|-----------|--------------------|----------------------------------|
| 10        | 2.02               | 152                              |
| 1000      | 25                 | 1875                             |
| 2000      | 26.6               | 2000                             |

<sup>&</sup>lt;sup>1</sup>All three structures are fully stressed

in Fig. 6c has the same topology as that obtained from a linear model; however, the internal forces are tension forces rather than compression forces. Also note that the cross-sectional area of each member in Fig. 6c is below the upper bound and, in agreement with the optimality conditions in (33), all members have the same strain energy (fully stressed). In contrast, when we reduce the upper bound to  $A_i^{\text{max}} = 1.155 \times 10^{-4} \, \text{m}^2$ , a different topology, shown in Fig. 6d, is obtained and the area of each member coincides with the upper bound, leading to a design without constant strain energy (non-fully stressed). Convergence plots for both cases are provided in Fig. 7.

## 8.3 Tangentially loaded donut

Here, we investigate the centrally supported, tangentially loaded, donut-shaped domain available with download of GRAND and shown in Fig. 8a. Also shown is the base mesh used by GRAND to generate a level 4 ground structure composed of 69,400 members (Zegard and Paulino 2014). A Young's modulus of  $E_i = 170$  GPa is assigned to all members. As in the previous examples, an end filter with  $f^{\text{tol}} = 0.01$  is used to remove thin members remaining at convergence. All other optimization parameters are provided in Table 4.

Considering small deformation kinematics and the linear material model, we expect to obtain an assembly of five structures resembling Michell's solution for a cantilever with circular support (Zegard and Paulino 2014). This solution is repeated here in Fig. 8b with the tension and compression members indicated in blue and red, respectively.

Considering large deformation kinematics and the cable material model with load  $P=100~\rm kN$ , we obtain the structure in Fig. 8c in which all members are in tension. Note that in the deformed configuration, the members making up each tension strand become collinear with the direction of the load, putting the structure in equilibrium. In this case, the configuration found considering finite deformations with the cable material model is different from and totally meaningless for the case of a small deformation kinematics. The objective function plotted in Fig. 9 shows smooth convergence.

### 8.4 Spider web inspired cable net

Inspired by the elegant and efficient cable systems found in nature, we design a spider web—inspired cable net using the proposed formulation. Cranford et al. (2012) provide empirically parameterized stress-strain curves for two types of spider silks: radial (dragline) silk and spiral (viscid) silk.

<span id="page-11-0"></span>**Fig. 6** Pin-supported beam with midspan point load. **a** Domain, ground structure, and boundary conditions. **b** Optimal design considering small deformation kinematics and linear material model (aligned nodes are removed in post-processing). Undeformed (top) and

deformed (bottom) topologies considering large deformation kinematics and the cable material model, with  $\mathbf{c}\ A_i^{max}=1.444\times 10^{-3}\ \mathrm{m}^2$  for all i and  $\mathbf{d}\ A_i^{max}=1.155\times 10^{-4}\ \mathrm{m}^2$  for all i. (color online)

<span id="page-12-0"></span>**Table 3** Optimization input parameters for the pin-supported beam

| Volume limit, V <sup>max</sup>        | $2.000 \times 10^{-3} \; m^3$        |
|---------------------------------------|--------------------------------------|
| Initial area, $A_0$                   | $2.888 \times 10^{-6} \; \text{m}^2$ |
| Move parameter, $\gamma$              | $1.000 \times 10^{3}$                |
| Convergence tolerance, tol            | $1.000 \times 10^{-9}$               |
| Objective tolerance, $f^{\text{tol}}$ | $1.000 \times 10^{-2}$               |
| Damped Newton parameter, $\eta$       | $1.000 \times 10^{-8}$               |
|                                       |                                      |

Although spider silk mechanical properties are dependent on a variety of factors (e.g., type of spider, type of silk, spinning conditions), the spiral (viscid) silk reported by Cranford et al. (2012) seems to fit well with the constitutive model proposed here as it reaches strains of around 250% and has a stress-strain response with positive curvature (i.e., it satisfies the convexity requirement discussed in Section 6.2). Additional information regarding spider silk can be found in the review paper by Omenetto and Kaplan (2010).

The goal of this example is not to exactly match the material properties and boundary conditions of a real spider web, but instead, to use the general characteristics of spider webs as inspiration for conceptual design of a cable net. Noting that the design depends on the ratio of the applied load to the stiffness of the system, we consider dimensionless parameters in our design and investigate the effect of varying the design variable upper bound,  $A_i^{\max}$ , for a given volume limit,  $V^{\max}$ .

<span id="page-12-1"></span>**Fig. 7** Convergence of the objective function for the pin-supported beam considering large deformation kinematics and the cable material model for the two different design variable upper bounds considered (color online)

With an orb-web in mind, we consider simplified boundary conditions on a 2D circular domain of radius,  $r_0 = 1$ . Fully fixed supports are placed at 8 equally spaced locations around the circumference and a single out-of-plane point load of magnitude 1 is applied at the center of the domain. We use GRAND (Zegard and Paulino 2014) to generate a level 3 ground structure on a polar grid with 16 circumferential and 16 radial divisions. Inside a central hole of radius,  $r_i = 0.2$ , we only allow radial members, i.e., a restriction zone is defined in GRAND such that additional members are not generated in that region. The initial ground structure contains 5056 members. A summary of the domain, boundary conditions, base mesh, and initial ground structure is provided in Fig. 10. The volume limit is defined to be 1% of the total in-plane area of the design domain (i.e.,  $V^{\text{max}} = 0.01\pi r^2$ ). Additionally, we assign a Young's modulus of  $E_i = 1000$  to all members. An end filter with  $f^{\text{tol}} = 0.001$  is used to remove thin members remaining at convergence. All other optimization parameters are provided in Table 5.

Results considering large deformation kinematics and the cable material model are provided in Fig. 11a and b considering design variable upper bounds,  $A_{:}^{\text{max}}$ 0.005 and 0.0012, respectively, for all i. In Fig. 11a, the design variable upper bound is not active and in the very simple final design containing 8 radial members with crosssectional area,  $A_i = 0.0039$  for all i, all elements have the same strain energy (fully stressed). In the other case, the design variable upper bound is active such that some members have cross-sectional area,  $A_i = A_i^{\text{max}} = 0.0012$ , and additional radial and "circumferential" members are included in the final design to add additional stiffness. A side view showing the deflection of each design confirms that the case with inactive upper bound is stiffer. It is also interesting to note that although the initial ground structure contains many crossing members, the optimal solution prefers radial and nearly circumferential members. In fact, the design in Fig. 11b is reminiscent of an orb-web with only radial and spiral members (Fig. 12a). Furthermore, the design clearly prefers higher stiffness for the radial members, which mimics the distinction in real spider webs between the radial (dragline) silk, which is a few orders of magnitude stiffer than the spiral (viscid) silk (Cranford et al. 2012).

Unlike the designs in Fig. 11, spider webs in nature are imperfect. Notice in Fig. 12a that neither the radial nor circumferential strands are equally spaced and some of the circumferential members even intersect each other. In an effort to achieve a spider web–inspired cable net that is imperfect, like those found in nature, we redesign the spider web using an initial ground structure with perturbed

<span id="page-13-0"></span>**Fig. 8** Centrally supported, tangentially loaded donut. **a** Domain, boundary conditions and mesh used to generate the level four ground structure. **b** Optimal design considering small deformation kinematics

and linear material model (nodes omitted for clarity). **c** Undeformed (left) and deformed (right) topologies considering large deformation kinematics and the cable material model. (color online)

nodal positions. Starting with the same nodal mesh used to define the initial ground structure of the previous spider web designs (Fig. 10), we modify the potential spacing of

Table 4 Optimization input parameters for the tangentially loaded donut

<span id="page-13-1"></span>

| Volume limit, V <sup>max</sup>        | $2.000 \times 10^{-3} \text{ m}^3$ |
|---------------------------------------|------------------------------------|
| Initial area, $A_0$                   | $2.157 \times 10^{-7} \text{ m}^2$ |
| Maximum area, $A_i^{\text{max}}$      | $2.157 \times 10^{-3} \text{ m}^2$ |
| Move parameter, $\gamma$              | $1.000 \times 10^4$                |
| Convergence tolerance, tol            | $1.000 \times 10^{-9}$             |
| Objective tolerance, $f^{\text{tol}}$ | $1.000 \times 10^{-2}$             |
| Damped Newton parameter, $\eta$       | $1.000 \times 10^{-8}$             |
|                                       |                                    |

the radial members by randomly perturbing each of the 16 sets of radial nodes (each radial set of nodes is defined by the same polar angle) by a uniformly distributed random perturbation in the range  $[-7.2^{\circ}, 7.2^{\circ}]$ . Additionally, we randomly select 1% of the nodes to perturb in the radial direction by a uniformly distributed random perturbation in the range [-0.005 units, 0.005 units]. The level 3 ground structure containing only radial members within a radius of  $r_i = 0.2$  is regenerated on this perturbed nodal mesh. This time, due to an increased number of overlapping members, the initial ground structure contains a total of 5045 members. The imperfect spider web, designed

<span id="page-14-0"></span>**Fig. 9** Convergence of the objective function for the tangentially loaded donut considering large deformation kinematics and the cable material model

considering all the same input parameters as the previous spider web designs (Table [5\)](#page-14-2) and *A*max *<sup>i</sup>* = 0.0012, is provided in Fig. [12c](#page-15-1).

<span id="page-14-2"></span>**Table 5** Optimization input parameters for spider web–inspired cable net

| Volume limit, V max        | 0.01π        |
|----------------------------|--------------|
| Initial area, A0           | 1.223 × 10−5 |
| Move parameter, γ          | 1.000 × 102  |
| Convergence tolerance, tol | 1.000 × 10−9 |
| Objective tolerance, f tol | 1.000 × 10−3 |
| Damped Newton parameter, η | 1.000 × 10−8 |
|                            |              |

## **9 Conclusion**

We proposed a topology optimization formulation for conceptual design of tension-only cable networks of maximum stiffness by maximizing the stationary potential energy of the system. This objective function was selected because of its elegant sensitivities, which can be computed without solving an adjoint system of equations. With the goal of finding optimal cable network configurations for conceptual design, we assumed uniaxial strain in all members for simplicity. In order to promote tension-only designs, we prescribed a tension-only constitutive relation

<span id="page-14-1"></span>**Fig. 10** Domain, boundary conditions, base mesh, and initial ground structure used for the spider web–inspired cable net

<span id="page-15-0"></span>Fig. 11 Spider web-inspired cable net: optimal design considering large deformation kinematics, the cable material model, and the unperturbed nodal mesh with

**a**  $A_i^{\text{max}} = 0.005$  for all i and **b**  $A_i^{\text{max}} = 0.0012$  for all i

<span id="page-15-1"></span>Fig. 12 Comparison of a an orb spider web found in nature (https://inchemistry.acs.org/content/inchemistry/en/atomic-news/spider-webs.html) and our spider web—inspired cable nets designed using topology optimization on a b unperturbed nodal mesh and c perturbed nodal mesh

such that the members have zero stiffness in compression. Additionally, in order to induce tension in the members, finite displacement and deformation kinematics were considered such that the structures are allowed to undergo large configurational changes as well as large strains. We assumed a strain energy density function, which is shown to meet a curvature requirement that ensures our optimization problem is convex for an appropriately selected axial strain measure (for simplicity, we chose a linear strain measure). Several simple 2D numerical examples were provided to demonstrate that the cable results are 1) distinct from those obtained using linear mechanics, 2) are dependent on the load magnitude for a given material, and (3) are dependent on large configurational changes in order to induce tension in the members. Finally, the formulation is used to design a 3D spider web-inspired cable net, where a very simple topology was obtained when the box constraints were relaxed, while a more complex design with many similarities to an orb-web was obtained when the box constraints were set to become active in the final design.

**Acknowledgments** This paper is dedicated to the memory of Dr. Frei Otto (1925–2015).

Funding information GHP and EDS acknowledge the financial support from the US National Science Foundation under projects #1559594 and #1663244 and the endowment provided by the Raymond Allen Jones Chair at the Georgia Institute of Technology. ASR Jr. appreciates the financial support from the Brazilian National Council for Research and Development (CNPq) and from the Laboratory of Scientific Computing and Visualization (LCCV) at the Federal University of Alagoas (UFAL). The information in this paper is the sole opinion of the authors and does not necessarily reflect the views of the sponsoring agencies.

#### Compliance with Ethical Standards

Conflict of interests The authors declare that they have no conflict of interest.

**Replication of results** The paper includes details of the numerical implementation and all input parameters for the numerical examples are provided to facilitate replication of the results.

## <span id="page-16-0"></span>Appendix A: Derivation of element tangent stiffness matrix

We derive the element tangent stiffness matrix, focusing on the partition,  $\mathbf{k}_{i}^{qq}$ :

<span id="page-16-2"></span>
$$\begin{aligned} \mathbf{k}_{i}^{qq} &= \frac{\partial \mathbf{t}_{i}^{q}}{\partial \mathbf{u}_{i}^{q}} \bigg|_{\mathbf{u}_{k}} = \frac{\partial (\sigma_{i} a_{i} \mathbf{n}_{i})}{\partial \mathbf{u}_{i}^{q}} \\ &= a_{i} \frac{\partial \sigma_{i}}{\partial \mathbf{u}_{i}^{q}} \bigg|_{\mathbf{u}_{k}} \mathbf{n}_{i} + a_{i} \sigma_{i} \frac{\partial \mathbf{n}_{i}}{\partial \mathbf{u}_{i}^{q}} \bigg|_{\mathbf{u}_{k}} \\ &= a_{i} \frac{\partial \sigma_{i}}{\partial \lambda_{i}} \frac{\partial \lambda_{i}}{\partial \ell_{i}} \frac{\partial \ell_{i}}{\partial \mathbf{x}_{i}^{q}} \frac{\partial \mathbf{x}_{i}^{q}}{\partial \mathbf{u}_{i}^{q}} \bigg|_{\mathbf{u}_{k}} \mathbf{n}_{i} \\ &+ a_{i} \sigma_{i} \left( \frac{\partial \left(\frac{1}{\ell_{i}}\right)}{\partial \mathbf{u}_{i}^{q}} \bigg|_{\mathbf{u}_{k}} \left( \mathbf{x}_{i}^{q} - \mathbf{x}_{i}^{p} \right) + \frac{\partial \left( \mathbf{x}_{i}^{q} - \mathbf{x}_{i}^{p} \right)}{\partial \mathbf{u}_{i}^{q}} \bigg|_{\mathbf{u}_{k}} \frac{1}{\ell_{i}} \right) \\ &= \frac{a_{i}}{L_{i}} \frac{\partial \sigma_{i}}{\partial \lambda_{i}} \mathbf{n}_{i} \mathbf{n}_{i}^{T} + a_{i} \sigma_{i} \left( -\frac{\left( \mathbf{x}_{i}^{q} - \mathbf{x}_{i}^{p} \right)}{\ell_{i}^{2}} \mathbf{n}_{i} + \frac{1}{\ell_{i}} \mathbf{I} \right) \\ &= \frac{a_{i}}{L_{i}} \frac{\partial \sigma_{i}}{\partial \lambda_{i}} \mathbf{n}_{i} \mathbf{n}_{i}^{T} - \frac{a_{i} \sigma_{i}}{\ell_{i}} \mathbf{n}_{i} \mathbf{n}_{i}^{T} + \frac{a_{i} \sigma_{i}}{\ell_{i}} \mathbf{I} \\ &= \frac{a_{i}}{L_{i}} \frac{\partial^{2} \Psi_{i}}{\partial \lambda_{i}^{2}} \mathbf{n}_{i} \mathbf{n}_{i}^{T} + \frac{t_{i}}{\ell_{i}} \left( \mathbf{I} - \mathbf{n}_{i} \mathbf{n}_{i}^{T} \right) \end{aligned} \tag{46}$$

In the second line of (46), we use the observation that since  $J_i = \lambda_i$ ,  $a_i = A_i$ ; in the last line of (46), we substitute the expression for internal force,  $t_i = a_i \sigma_i$ , and use the relationship between  $\Psi_i$  and  $\sigma_i$  from (13). Additionally, in (46), we use the following derivatives:

$$\frac{\partial l_i}{\partial \mathbf{x}_i^q} = \frac{\mathbf{x}_i^q - \mathbf{x}_i^p}{\ell_i} = \mathbf{n}_i \tag{47}$$

$$\frac{\partial (\mathbf{x}_{i}^{q} - \mathbf{x}_{i}^{p})}{\partial \mathbf{u}_{i}^{q}} = \frac{\partial \mathbf{x}_{i}^{q}}{\partial \mathbf{u}_{i}^{q}} = \frac{\partial (\mathbf{X}_{i}^{Q} + \mathbf{u}_{i}^{q})}{\partial \mathbf{u}_{i}^{q}} = \mathbf{I}$$
(48)

$$\frac{\partial \left(\frac{1}{\ell_i}\right)}{\partial \mathbf{u}_i^q} \bigg|_{\mathbf{u}_k} = -\frac{\mathbf{x}_i^q - \mathbf{x}_i^p}{\ell_i^3} = -\frac{1}{\ell_i^2} \mathbf{n}_i \tag{49}$$

## <span id="page-16-1"></span>Appendix B: Positive semi-definiteness of the element tangent stiffness matrix

As stated in Section 6.2, to show that the lower partition of the element tangent stiffness matrix in (28) is positive semi-definite, we need to show that the matrix,  $(\mathbf{I} - \mathbf{n}_i \mathbf{n}_i^T)$ , is positive semi-definite. To do so, we can easily confirm that the determinants of all principal minors of the following matrix are non-negative (Strang 2006):

$$\left(\mathbf{I} - \mathbf{n}_{i} \mathbf{n}_{i}^{T}\right) = \begin{bmatrix} 1 - \cos^{2} \alpha & \cos \beta \cos \alpha & \cos \gamma \cos \alpha \\ 1 - \cos^{2} \beta & \cos \gamma \cos \beta \\ \text{symm.} & 1 - \cos^{2} \gamma \end{bmatrix}$$
(50)

where we have noted that the unit vector along member i's axis is  $\mathbf{n}_i = [\cos \alpha, \cos \beta, \cos \gamma]^T$  for some angles,  $0 \le \alpha, \beta, \gamma \le 2\pi$ , defining the orientation of member i in Cartesian space.

## <span id="page-17-0"></span>Appendix C: Damped Newton and line search algorithms

The damped Newton algorithm used to solve the nonlinear equilibrium equations is provided in Algorithm 1. In line 8 of the algorithm, line search is used to compute the step length for the solution update. The line search algorithm is provided in Algorithm 2, where the Armijo condition is stated on line 4 and the backtracking parameter is computed using a quadratic interpolant on line 5. Additional detail on the Newton and damped Newton methods can be found in textbooks such as Wriggers (2008), Bonet and Wood (2008), and Madsen and Nielsen (2010); additional detail on the specific line search algorithm used here is detailed by Ascher and Greif (2011) and used for topology optimization considering various nonlinear mechanics models by Zhang et al. (2017) and Zhao et al. (2019).

## Algorithm 1 Damped Newton algorithm.

```
1: assume \mathbf{u}_0 is the displacement at the previous optimiza-
      tion iteration
 2: set maxIter = 30
 3: for k = 0 to maxIter do
            compute T(\mathbf{u}_k)
 4:
            compute \mathbf{R}(\mathbf{u}_k) = \mathbf{T}(\mathbf{u}_k) - \mathbf{F}
 5:
            compute \mathbf{K}^{t,\eta}(\mathbf{u}_k)
 6:
            compute \Delta \mathbf{u}_k = -\mathbf{K}^{t,\eta} (\mathbf{u}_k)^{-1} \mathbf{R} (\mathbf{u}_k)
 7:
            find line search parameter, 0 < \xi_{min} \le \xi_k \le 1
 8:
            update \mathbf{u}_{k+1} = \mathbf{u}_k + \xi_k \Delta \mathbf{u}_k
 9:
            if ||\mathbf{R}(\mathbf{u}_k)||/||\mathbf{F}|| < \text{tol or } ||\Delta \mathbf{u}_k||/(1+||\mathbf{u}_{k+1}||)
10:
            < tol then
                  break
11:
            end if
12:
13: end for
```

#### Algorithm 2 Line search algorithm.

```
1: input \mathbf{u}_k, \Delta \mathbf{u}_k
 2: set j = 0, \tau = 10^{-4}, \xi_{\min} = 10^{-6}, \xi_{\max} = 1, \xi_k^0 = \xi_{\max}
 3: compute \mathbf{u}_{k+1} = \mathbf{u}_k + \xi_k^0 \Delta \mathbf{u}_k
 4: while \Pi(\mathbf{u}_{k+1}) > \Pi(\mathbf{u}_k) + \tau \xi_k^j \nabla \Pi(\mathbf{u}_k)^T \Delta \mathbf{u}_k do
               compute \mu = -0.5\nabla\Pi \left(\mathbf{u}_{k}\right)^{T} \Delta\mathbf{u}_{k}\xi_{k}^{j}/\left(\Pi \left(\mathbf{u}_{k+1}\right) - \frac{1}{2}\right)
               \Pi\left(\mathbf{u}_{k}\right) - \nabla\Pi\left(\mathbf{u}_{k}\right)^{T} \Delta\mathbf{u}_{k} \xi_{k}^{j}
               if \mu < 0.1 or \nabla \Pi (\mathbf{u}_k)^T \Delta \mathbf{u}_k \geq 0 then
 6:
                       \mu = 0.5
 7:
 8:
               update \xi_k^{j+1} = \mu \xi_k^j
 9:
               update \mathbf{u}_{k+1} = \mathbf{u}_k + \xi_k^{j+1} \Delta \mathbf{u}_k
10:
11:
12: end while
13: Return \xi_k = \xi_k^J
```

## <span id="page-17-1"></span>Appendix D: Optimality Criteria design variable update scheme

The optimality criteria (OC) method is characterized by a recursive update derived using Lagrangian duality of truncated Taylor approximated subproblems evaluated at intervening variable  $y_i = A_i^{\alpha_i}$ ,  $\alpha_i < 0$  (Bendsøe and Sigmund 2003; Groenwold and Etman 2008). The update for iteration k + 1 is as follows:

<span id="page-17-3"></span>
$$A_i^{k+1} = \begin{cases} \underline{A}_i^k & \text{if } A_i^k B_i^k \le \underline{A}_i^k \\ A_i^k B_i^k & \text{if } \underline{A}_i^k \le A_i^k B_i^k \le \overline{A}_i^k \\ \overline{A}_i^k & \text{if } A_i^k B_i^k \ge \overline{A}_i^k \end{cases}$$
(51)

<span id="page-17-4"></span>where

$$B_i^k = \left( -\frac{\frac{\partial f}{\partial A_i} \Big|_{\mathbf{A} = \mathbf{A}^k}}{\phi \frac{\partial g}{\partial A_i} \Big|_{\mathbf{A} = \mathbf{A}^k}} \right)^{\frac{1}{1 - \alpha_i}}$$
(52)

and the bounds,  $\underline{A}_i^k$  and  $\overline{A}_i^k$ , are defined by a move limit, M, (see (37)) such that

$$\underline{A}_{i}^{k} = \max \left\{ \begin{array}{ll} A_{i}^{k} - M; & \overline{A}_{i}^{k} = \min \left\{ \begin{array}{ll} A_{i}^{k} + M \\ A_{i}^{\max} \end{array} \right. \right.$$
 (53)

In (52),  $\phi$  is the Lagrange multiplier and the quantity  $1/(1-\alpha_i)$  is a damping factor corresponding to a reciprocal approximation when  $\alpha_i = -1$ . Here, we determine  $\alpha_i$  using a two-point approximation such that the derivatives of the reciprocal approximation at iteration k match the derivatives of the function at iteration k-1 (Fadel et al. 1990; Groenwold and Etman 2008):

<span id="page-17-5"></span>
$$\alpha_i^k = 1 + \frac{\ln\left(\frac{\partial f}{\partial A_i}\Big|_{\mathbf{A} = \mathbf{A}^{k-1}} / \frac{\partial f}{\partial A_i}\Big|_{\mathbf{A} = \mathbf{A}^k}\right)}{\ln\left(A_i^{k-1} / A_i^k\right)}$$
(54)

In the first iteration,  $\alpha_i = -1$  and in subsequent iterations,  $\alpha_i$  is computed based on (54) with bounds,  $-15 \le \alpha_i \le -0.1$ .

## <span id="page-17-2"></span>**Appendix E: Other strain measures**

In the main body of the paper, a linear strain measure is selected for simplicity; however, many other strain measures can be defined and used with the proposed formulation. In this Appendix, we explore three common strain measures, linear strain (used here), logarithmic strain, and Green-Lagrange strain, and investigate how each one affects convexity of the optimization problem. Table 6 defines the three strain measures and shows the strain energy density

<span id="page-18-1"></span>Table 6 Definition of three common strain measures and the strain energy density function as a function of stretch for each one

| Strain measure | Instantaneous strain, $d\varepsilon_i$ | Strain, $\varepsilon_i$                   | Strain energy density, $\Psi_i$                  | $\frac{\partial \Psi_i}{\partial \lambda_i}$       | $\frac{\partial^2 \Psi_i}{\partial \lambda_i^2}$           |
|----------------|----------------------------------------|-------------------------------------------|--------------------------------------------------|----------------------------------------------------|------------------------------------------------------------|
| Linear         | $\frac{d\ell_i}{L_i}$                  | $\lambda_i - 1$                           | $\frac{E_i}{2} \left( \lambda_i - 1 \right)^2$   | $E_i (\lambda_i - 1)$                              | $E_i$                                                      |
| Logarithmic    | $\frac{d\ell_i}{\ell_i}$               | $\ln \lambda_i$                           | $\frac{E_i}{2} \left( \ln \lambda_i \right)^2$   | $\frac{E_i}{\lambda_i} \ln \lambda_i$              | $\frac{E_i}{\lambda_i^2} \left( 1 - \ln \lambda_i \right)$ |
| Green-Lagrange | $\frac{\ell_i d\ell_i}{L_i^2}$         | $\frac{1}{2}\left(\lambda_i^2 - 1\right)$ | $\frac{E_i}{8} \left( \lambda_i^2 - 1 \right)^2$ | $\frac{E_i\lambda_i}{2}\left(\lambda_i^2-1\right)$ | $\frac{E_i}{2}\left(3\lambda_i^2-1\right)$                 |

as a function of stretch for each one. Noting that the strain energy density of member i can be expressed as

$$\Psi_i = \frac{1}{V_i} \int_{L_i}^{\ell_i} \sigma_i a_i d\ell_i \tag{55}$$

we can re-write the integral for each case in terms of strain, i.e., with limits of integration from 0 to  $\varepsilon_i$  and with the appropriate instantaneous strain measure substituted for  $d\ell_i$ . Then, the integrand is equal to  $\partial \Psi_i/\partial \lambda_i$ , and we can obtain the expressions for strain energy density of the three strain measures provided in Table 6. From the last column of Table 6, it is clear that the convexity requirement,  $\partial^2 \Psi_i/\partial \lambda_i^2 \geq 0$ , is always satisfied for the cases of linear and Green-Lagrange strain, and is only satisfied for logarithmic strain when  $\lambda_i \leq e$ .

To further illustrate the differences between the three strain measures, the strain energy density function is plotted against axial strain and axial stretch in Fig. 13a and b, respectively. For the case of logarithmic strain, the strain energy density function is not a convex function of axial stretch and thus does not meet the curvature requirement needed for a positive semi-definite element tangent stiffness matrix. Although they are different, the general trend of the linear and Green-Lagrange strain measures are similar, and for the conceptual designs pursued in this paper, these two strain measures can be expected to lead to similar designs.

<span id="page-18-2"></span>Fig. 13 Strain energy density function versus **a** axial strain,  $\varepsilon_i$ , and **b** axial stretch,  $\lambda_i$ , for three common strain measures

## <span id="page-18-0"></span>**Appendix F: Nomenclature**

### Nomenclature

f objective function

g volume constraint

 $V^{\text{max}}$  upper limit on structural volume

 $A_i$  cross-sectional area of member i in undeformed configuration

 $A_i^{\text{max}}$  upper bound on the cross-sectional area of member i in undeformed configuration

 $\Pi$  total potential energy of the system

 $\Pi_{min}$  stationary potential energy of the system

u vector of nodal displacements

 $\eta$  damped Newton parameter

 $\Psi_i$  stored strain energy per unit volume of member i

 $E_i$  material parameter for member i

 $\varepsilon_i$  axial strain in member i

 $\lambda_i$  axial stretch of member i

 $X_i$  position of member i in undeformed configuration

 $\mathbf{E}^{j}$  coordinate frame describing the undeformed configuration

 $\mathbf{x}_i$  position of member i in deformed configuration

 e<sup>j</sup> coordinate frame describing the deformed configuration

 $n_d$  number of spatial dimensions

 $\Delta \mathbf{u}_{i}^{p}$  incremental displacement at end p of member i

 $\Delta \mathbf{u}_{i}^{q}$  incremental displacement at end q of member i

b

- $L_i$  length of member i in undeformed configuration
- $\ell_i$  length of member i in deformed configuration
- $N_i$  unit vector along the length of member i in undeformed configuration
- $\mathbf{n}_i$  unit vector along the length of member i in deformed configuration
- $\lambda_i^2, \lambda_i^3$  transverse principal stretches
  - $J_i$  Jacobian of member i
  - $a_i$  cross-sectional area of member i in deformed configuration
  - $V_i$  volume of member i in undeformed configuration
  - $v_i$  volume of member i in deformed configuration
  - $t_i$  magnitude of internal force in member i
  - $\sigma_i$  axial component of Cauchy stress in member i
  - U internal strain energy
  - N number of cable members in the model
  - $\Omega$  potential of external loads
  - F external load vector
  - T member internal force vector
  - R residual force vector
  - $\mathbf{u}_k$  vector of nodal displacements at iteration k
  - $\Delta \mathbf{u}_k$  incremental displacement at Newton-Raphson iteration k
  - $\mathbf{K}^t$  global tangent stiffness matrix
  - $\mathbf{k}_{i}^{t}$  tangent stiffness matrix of member i
  - $\mathbf{t}_i$  internal force vector of member i
  - $\mathbf{u}_i$  vector of nodal displacements for member i
  - **A**\* optimal solution
  - $\phi^*$  Lagrange multiplier at optimum
  - $\xi_k$  line search parameter at Newton-Raphson iteration
  - ν move parameter
  - tol convergence tolerance
- **u**<sup>Top</sup> nodal displacement vector for dof in the topology
- $\mathbf{K}^{t,\text{Top}}$  global tangent stiffness matrix for dof in the topology
  - $\mathbf{F}^{\text{Top}}$  external load vector for dof in the topology
  - T<sup>Top</sup> internal force vector for dof in the topology
- **R**<sup>Top</sup> residual force vector for dof in the topology
  - **Q** mapping matrix between ground structure and topology
- $\Delta \mathbf{u}_k^{\text{Top}}$  incremental displacement vector for the topology at Newton-Raphson iteration k
  - $\rho$  global equilibrium tolerance
  - f<sup>tol</sup> tolerance on increases in objective due to end filter
  - $\Delta f$  change in objective due to end filter
  - $\alpha_i$  OC parameter
  - $B_i^k$  recursive OC multiplier for member i in iteration optimization iteration k
  - M OC move limit
  - $\underline{A}_{i}^{k}$  lower bound on change in design variables
  - $A_i^{\kappa}$  upper bound on change in design variables

#### References

- <span id="page-19-16"></span>Ascher UM, Greif C (2011) A first course on numerical methods. SIAM
- <span id="page-19-17"></span>Bendsøe MP, Sigmund O (2003) Topology optimization: theory, methods, and applications. Springer
- <span id="page-19-14"></span>Bonet J, Wood RD (2008) Nonlinear continuum mechanics for finite element analysis. Cambridge University Press, 2nd edn
- <span id="page-19-6"></span>Buhl T, Pedersen CBW, Sigmund O (2000) Stiffness design of geometrically nonlinear structures using topology optimization. Struct Multidiscip Optim 19(2):93–104
- <span id="page-19-4"></span>Christensen PW, Klarbring A (2008) An introduction to structural optimization. vol 153, Springer Science & Business Media
- <span id="page-19-21"></span>Cranford SW, Tarakanova A, Pugno NM, Buehler MJ (2012) Nonlinear material behaviour of spider silk yields robust webs. Nature 482(7383):72
- <span id="page-19-2"></span>Dorn WS, Gomory RE, Greenberg HJ (1964) Automatic design of optimal structures. J de Mech 3:25–52
- <span id="page-19-20"></span>Du Z, Guo X (2016) Symmetry analysis for structural optimization problems involving reliability measure and bi-modulus materials. Struct Multidiscip Optim 53(5):973–984
- <span id="page-19-12"></span>Du Z, Zhang W, Zhang Y, Xue R, Guo X (2019) Structural topology optimization involving bi-modulus materials with asymmetric properties in tension and compression. Comput Mech 63(2):335– 363
- <span id="page-19-22"></span>Fadel GM, Riley MF, Barthelemy JM (1990) Two point exponential approximation method for structural optimization. Struct Multi-discip Optim 2(2):117–124
- <span id="page-19-7"></span>Gea HC, Luo J (2001) Topology optimization of structures with geometrical nonlinearities. Comput Struct 79(20):1977–1985
- <span id="page-19-3"></span>Gilbert M, Tyas A (2003) Layout optimization of large-scale pinjointed frames. Eng Comput 20(8):1044–1064
- <span id="page-19-0"></span>Glaeser L (1972) The work of Frei Otto. The Museum of Modern Art, New York
- <span id="page-19-11"></span>Gomes FA, Senne TA (2014) An algorithm for the topology optimization of geometrically nonlinear structures. Int J Numer Methods Eng 99(6):391–409
- <span id="page-19-18"></span>Groenwold AA, Etman LFP (2008) On the equivalence of optimality criterion and sequential approximate optimization methods in the classical topology layout problem. Int J Numer Methods Eng 73(3):297–316
- <span id="page-19-19"></span>Guo X, Du Z, Cheng G, Ni C (2013) Symmetry properties in structural optimization: some extensions. Struct Multidiscip Optim 47(6):783–794
- <span id="page-19-5"></span>Jog C (1996) Distributed-parameter optimization and topology design for non-linear thermoelasticity. Comput Methods Appl Mech Eng 132(1-2):117–134
- <span id="page-19-8"></span>Jung D, Gea HC (2004) Topology optimization of nonlinear structures. Finite Elem Anal Des 40(11):1417–1427
- <span id="page-19-13"></span>Kanno Y, Ohsaki M (2003) Minimum principle of complementary energy of cable networks by using second-order cone programming. Int J Solids Struct 40(17):4437–4460
- <span id="page-19-10"></span>Kawamoto A (2009) Stabilization of geometrically nonlinear topology optimization by the levenberg–Marquardt method. Struct Multidiscip Optim 37(4):429–433
- <span id="page-19-9"></span>Kemmler R, Lipka A, Ramm E (2005) Large deformations and stability in topology optimization. Struct Multidiscip Optim 30(6):459–476
- <span id="page-19-15"></span>Klarbring A, Rönnqvist M (1995) Nested approach to structural optimization in nonsmooth mechanics. Struct Multidiscip Optim 10(2):79–86
- <span id="page-19-1"></span>Klarbring A, Strömberg N (2012) A note on the min-max formulation of stiffness optimization including non-zero prescribed displacements. Struct Multidiscip Optim 45(1):147–149

- <span id="page-20-16"></span>Klarbring A, Stromberg N (2013) Topology optimization of hypere- ¨ lastic bodies including non-zero prescribed displacements. Struct Multidiscip Optim 47(1):37–48
- <span id="page-20-22"></span>Kreindler E, Jameson A (1972) Conditions for nonnegativeness of partitioned matrices. IEEE Trans Autom Control 17(1):147–148
- <span id="page-20-21"></span>Leon SE, Paulino GH, Pereira A, Menezes IF, Lages EN (2011) A unified library of nonlinear solution schemes. Appl Mech Rev 64(4):040803
- <span id="page-20-20"></span>Luo Q, Tong L (2016) An algorithm for eradicating the effects of void elements on structural topology optimization for nonlinear compliance. Struct Multidiscip Optim 53(4):695–714
- <span id="page-20-19"></span>Luo Y, Wang MY, Kang Z (2015) Topology optimization of geometrically nonlinear structures based on an additive hyperelasticity technique. Comput Methods Appl Mech Eng 286:422–441
- <span id="page-20-4"></span>Madsen K, Nielsen HB (2010) Introduction to optimization and data fitting. Informatics and Mathematical Modelling, Technical University of Denmark
- <span id="page-20-5"></span>Michell AG (1904) The limits of economy of material in frame structures. Phil Mag 8(6):589–597
- <span id="page-20-3"></span>Nerdinger W (ed) (2005) Frei Otto complete works: lightweight construction, natural design. Birkhauser, Switzerland
- <span id="page-20-13"></span>Neves MM, Rodrigues H, Guedes JM (1995) Generalized topology design of structures with a buckling load criterion. Struct Multidiscip Optim 10(2):71–78
- <span id="page-20-29"></span>Omenetto FG, Kaplan DL (2010) New opportunities for an ancient material. Science 329(5991):528–531
- <span id="page-20-2"></span>Otto F, Rasch B (1995) Finding form: towards an architecture of the minimal. Axel Menges, Stuttgard
- <span id="page-20-1"></span>Otto F, Schleyer F-K (1969) Tensile structures: design, structure, and calculation of buildings, cables, nets, and membranes, volume 2. The M.I.T. Press
- <span id="page-20-0"></span>Otto F, Trostel R (1967) Tensile structures: design, structure, and calculation of buildings, cables, nets, and membranes, volume 1. The M.I.T. Press
- <span id="page-20-10"></span>Ramos Jr. AS, Paulino GH (2015) Convex topology optimization for hyperelastic trusses based on the ground-structure approach. Struct Multidiscip Opt 51(2):287–304
- <span id="page-20-24"></span>Ramos Jr. AS, Paulino GH (2016) Filtering structures out of ground structures – a discrete filtering tool for structural design optimization. Struct Multidiscip Optim 54(1):95–116
- <span id="page-20-25"></span>Sanders ED, Ramos Jr. AS, Paulino GH (2017) A maximum filter for the ground structure method an optimization tool to harness multiple structural designs. Eng Struct 151:235–252
- <span id="page-20-14"></span>Sekimoto T, Noguchi H (2001) Homologous topology optimization in large displacement and buckling problems. JSME International Journal Series A Solid Mechanics and Material Engineering 44(4):616–622

- <span id="page-20-6"></span>Sokoł T (2011) A 99 line code for discretized Michell truss ´ optimization written in Mathematica. Struct Multidiscip Optim 43(2):181–190
- <span id="page-20-7"></span>Sokoł T (2015) Multi-load truss topology optimization using the ´ adaptive ground structure approach. In: Łodygowski T, Rakowski J, Litewka P (eds) Recent Advances in Computational Mechanics, pp 9–16
- <span id="page-20-23"></span>Strang G (2006) Linear algebra and its applications. Brooks/Cole, Thomson
- <span id="page-20-18"></span>van Dijk NP, Langelaar M, van Keulen F (2014) Element deformation scaling for robust geometrically nonlinear analyses in topology optimization. Struct Multidiscip Optim 50(4):537–560
- <span id="page-20-28"></span>Vollrath F, Mohren W (1985) Spiral geometry in the garden spider's orb web. Naturwissenschaften 72(12):666–667
- <span id="page-20-17"></span>Wang F, Lazarov BS, Sigmund O, Jensen JS (2014) Interpolation scheme for fictitious domain techniques and topology optimization of finite strain elastic problems. Comput Methods Appl Mech Eng 276:453–472
- <span id="page-20-26"></span>Wriggers P (2008) Nonlinear finite element methods. Springer Science & Business Media
- <span id="page-20-27"></span>Wright S, Nocedal J (1999) Numerical optimization. Springer Sci 35(67-68):7
- <span id="page-20-15"></span>Yoon GH, Kim YY (2005) Element connectivity parameterization for topology optimization of geometrically nonlinear structures. Int J Solids Struct 42(7):1983–2009
- <span id="page-20-8"></span>Zegard T, Paulino GH (2014) GRAND – ground structure based topology optimization for arbitrary 2D domains using MATLAB. Struct Multidiscip Optim 50(5):861–882
- <span id="page-20-9"></span>Zegard T, Paulino GH (2015) GRAND3 – ground structure based topology optimization for arbitrary 3D domains using MATLAB. Struct Multidiscip Optim 52(6):1161–1184
- <span id="page-20-11"></span>Zhang X, Ramos Jr. AS, Paulino GH (2017) Material nonlinear topology optimization using the ground structure method with a discrete filtering scheme. Struct Multidiscip Optim 55(6):2045–2072
- <span id="page-20-12"></span>Zhang XS, Paulino GH, Ramos Jr. AS (2018) Multi-material topology optimization with multiple volume constraints: a general approach applied to ground structures with material nonlinearity. Struct Multidiscip Optim 57:161–182
- <span id="page-20-30"></span>Zhao T, Ramos Jr. AS, Paulino GH (2019) Material nonlinear topology optimization considering the von Mises criterion through an asymptotic approach Max strain energy and max load factor formulations. Int J Numer Methods Eng 118(13):804–828

**Publisher's note** Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.