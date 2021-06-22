# AE/VAE-SOM


## Goal for the project:

1. Latent space exploration of SOM.
2. Representation learning of AE/VAE/GAN. 
3. Anomaly detection for time-serials data.
    * 1) Spatial anomaly detection.
    * 2) Temporal anomaly detection.
    * 3) Generalization and robustness of the Algorithms.
4. Quantization of uncertainty.
    * 1) Spatial uncertainty.
    * 2) Temporal uncertainty.
___
### Updated May 27th:

* Common Trajectory:

![image](https://user-images.githubusercontent.com/34424773/120104020-b6021180-c152-11eb-8750-93d58f7def63.png){:height="50%" width="50%"}


* Faulty Trajectory:

![image](https://user-images.githubusercontent.com/34424773/120104023-c0bca680-c152-11eb-83fa-c387a7cacca0.png){:height="50%" width="50%"}

### To do list after May 27th update:

1. Investigate correlation between the common input vs. faulty input
2. How to process the biodirectional arrow in the Common trajectroy?
3. <del> How to down sample the $z_e$? </del>
4. <del> Markov Model </del>
___
### Updated June 4th:

* Fixed up the Markov Chain. Need to be validated.
    * *Note: The Markov Chain is not the only method to optimize the temporal model.*
* Set up a new block for spatial anomaly detection.
     * $P_{anomaly}(v) = 1- exp(-(error_q{v}/{\sigma})) (1)$. 
        <br/>$v$ refers to input data.
        <br/>$\sigma$ refers to $\frac{1}{2} \cdot ||w_{bmu}(v)-w_{bmu}^{'}(v)||$
        <br/>$error_q{v}$ refers to $||v-bmu(u_i)||$
* <del>$\sigma$ means the divergence of the data. sigma is larger, the results are more dispersed. $\sigma$ should be around $0.5 \rightarrow 1.5$ $Learning\ rate $ should be around $0.5 \rightarrow 2.5$. The map size should not more than $(10,10).$</del>
*  <br/> The detail about hyper parameters can be found here [A simple demo for Hyper parameters](https://share.streamlit.io/justglowing/minisom/dashboard/dashboard.py)
* An example:
 <br/>![image](https://github.com/qhxxsb/AE-VAE-SOM/blob/Sympathyzzk-patch/Figure/An_example.png){:height="50%" width="50%"}
* Loss
 <br/>![image](https://github.com/qhxxsb/AE-VAE-SOM/blob/Sympathyzzk-patch/Figure/Loss_AE.png){:height="50%" width="50%"}

### To do list after June 4th's update

1. <del>Tidy the source code. </del>
2. Change the dataset. 
    * <del>*Socketsense* </del>
    * *Itrust*
3. <del>Add a new block for anomaly detecion in Spatial space. It needs to test more to avoid FP.</del> 
4. <del>The hyper-parameters needs to be clarify in order to get a better results in SOM.</del>
5. Temporal anomaly detection.
    * Investigation in Markov Chain
    * <del>Investigation in Affinity Matrix.(Continue to abstract for the SOM neuron)</del>.
6. Confirm the Spatial anomaly detection and V&V results.
___

### Updated June 5th
1. The mapsize should be calibrated with the $\sigma$. In other words, when map size is larger, the sigma should be larger. **Question: How we should interpret these two hyper parameters?** 
2. The Sin function has been replaced by SocketSense.
3. Function update:
    * Add a new block to check the FP.
    * Add a new block to generate the affinity(similarity) matrx, which could be treated as *Truth table*.
### To do list after June 5th's update
1. <del> Normaliztion the Affinity matrix, which should either indicate the probability or similarity.</del>
2. Continue to work on the termporal anomaly detection.
___

### Updated June 8th

1. Get the Affinity probability of each *Path state.*
2. Visualization each Affinity Probability according to 
    * $P_{affinity} = exp(-distance(w_{i}, w_{j})/{\delta}) (2)$
    <br/> $w_{i}$ refers to the path state, $w_{j}$ refer to the other state
    <br/> I set a threshold $\delta$, to select the affinity. When $P_{affinity} > \delta$, we treat $w_j$ as *affinity state.*
3. Map the input to SOM.
    * For example, in our case, we have four pressures, these four pressure's sum will map with the Som. When the Pressure is larger, the color is deeper. 
    <br/>![image](https://github.com/qhxxsb/AE-VAE-SOM/blob/Sympathyzzk-patch/Figure/Heat_Map.png){:height="50%" width="50%"}
4. Overview results.
    * Temporal path-state trajectory.
    <br/>![image](https://github.com/qhxxsb/AE-VAE-SOM/blob/Sympathyzzk-patch/Figure/Temporal_Trajectory.png){:height="50%" width="50%"}
    * Visualization of affinity and path states.
    <br/>![image](https://github.com/qhxxsb/AE-VAE-SOM/blob/Sympathyzzk-patch/Figure/Visualization_Velocity.png){:height="50%" width="50%"}
### To do list after June 8th's update
1. <del> Reschematic Markov Chain.</del> 
    * <del>The Markov Chain should save as a table (dataframe) such as following:</del>
  

        |            |      S1       |  S2     |   S3       | ....... |
        | :--------: | :-------------: | :-------: | :----------: | :-- --: |
        | S1       |  $\times$     | $P_{12}$ |    $P_{13}$ | ....... |
        | S2       |   $P_{21}$    | $\times$ |  $P_{23}$   | ....... |
        | S3       | $P_{31}$      | $P_{32}$ | $\times$    | ....... |


2. Add text (probability) in the temporal trajectory figures.
3. Write a Problog program for the figures.

### Discussion ###
1. How can we prove that *Formula (2)* is reliable?
    * Usually, we should use $(w_{i}-w_{j})^2$ to compute the affinity matrix. But in this case, if we use square, the amount of affinity state will increase a lot. Is *Formula (2)* reasonable ?
    * How should we understand $\delta$?
2. How can we improve/calibrate the method used to map the input data to SOM?
3. Validate our data.....(This may meet a lot of problems)

___
### Updated June 12h ###
1. The sampling problem can be checked from the following link (the classical methods):
    * [Engineering statistics handbook.](https://www.itl.nist.gov/div898/handbook/pmc/section2/pmc22.htm)
2. Fix up the datatype of Markov Transition Matrix.
3. Fix up some bugs.

### Discussion ###
1. How can we understand **Average Sample Number ** (ASN)?
<br/> How we understand the $\beta = 0.029$ and which value should be $p_2$ in [Double Sampling Plan](https://www.itl.nist.gov/div898/handbook/pmc/section2/pmc24.htm)?
2. In the  [Double Sampling Plan](https://www.itl.nist.gov/div898/handbook/pmc/section2/pmc24.htm), they should define the $P_1$. 
<br/> For example, As I understand, $P_1 = 1- \alpha + \beta$, although $\alpha$ and $\beta$ belong to different threshold $<n_1,c_1>$ and $<n_2,c_2>$.
3. Should Markov Chain have a sample step(frequency) as sampling data? Otherwise, the tranistion probability would equal to 0 if we use the upsampling states, which shows as below:

![image](https://github.com/qhxxsb/AE-VAE-SOM/blob/Sympathyzzk-patch/Figure/Markov_Wrong.png){:height="50%" width="50%"}

___
### Updated June 15h ###
1. Add a flowgraph to illustrate the workbench.
 <br/>![image](https://github.com/qhxxsb/AE-VAE-SOM/blob/Sympathyzzk-patch/Figure/VAE-SOM%20toolchain.png){:height="50%" width="50%"}

*Note: the sampling problem has not added into the flowgraph. It is a critical problem we should consider now.*
### Discussion ###
1. <del>How should we sample the data?</del> 
2. <del>How can we merge the slide windows in our code </del>?

___

### Updated June 22rd ###
1. Fulfill the simlarity and temporal graph to show the dependency.
 <br/>!![image](https://github.com/qhxxsb/AE-VAE-SOM/blob/Sympathyzzk-patch/Figure/Temporal_Similarity.png){:height="50%" width="50%"}

2. Some summary for the potential publication.
    * 1) What's the purpose of SOM used in latent space?

        SOM can be understood as the measurement of latent space's Euclidean distance.

    * 2) Spatial modeling
            
        $\Bbb{E}_{PD(x)}[\log{p_{\theta}(x)}] \ge {\Bbb{E}_{PD(x)}[\Bbb{E}_{q{\phi}(z|x)}[\log{p_{\theta}(x|z)}]-\Bbb{KL}(q_{\phi}(z|x)||p(z))] } $
        
        $H_0 \ and \ H_1$

        $g(z_i,z_j) =    exp(-distance(w_{i}, w_{j})/{\delta}$

        $ g(z_i,z_j) \ compare \ to \ H_0  \ and \ H_1 $

        $  \mathcal{L}_{vae}  = {\Bbb{E}_{PD(x)}[\Bbb{E}_{q{\phi}(z|x)}[\log{p_{\theta}(x|z)}]-\Bbb{KL}(q_{\phi}(z|x)||p(z))] } $

        $   \mathcal{L}_{vae-som}(\phi, \theta, x_i,x_j)  = \mathcal{L}_{vae}(\phi, \theta, x_i,x_j) +g(z_i,z_j) $


    * 3) Temporal modeling

        How we get the threshold of temporal modeling?

3. Add a function for sliding window.