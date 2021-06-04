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

![image](https://user-images.githubusercontent.com/34424773/120104020-b6021180-c152-11eb-8750-93d58f7def63.png)


* Faulty Trajectory:

![image](https://user-images.githubusercontent.com/34424773/120104023-c0bca680-c152-11eb-83fa-c387a7cacca0.png)

### To do list on May 27th:

1. Investigate correlation between the common input vs. faulty input
2. How to process the biodirectional arrow in the Common trajectroy?
3. <del> How to down sample the z_e? </del>
4. <del> Markov Model </del>


___
### Updated June 4th:

* Fixed up the Markov Chain. Need to be validated.
    * *Note: The Markov Chain is not the only method to optimize the temporal model.*
* Set up a new block for spatial anomaly detection.
     * $P_{anomaly}(v) = 1- exp(-(error_q{v}/{\sigma}))$. 
        <br/>$v$ refers to input data.
        <br/>$\sigma$ refers to $\frac{1}{2} \cdot ||w_{bmu}(v)-w_{bmu}^{'}(v)||$
        <br/>$error_q{v}$ refers to $||v-bmu(u_i)||$
* $\sigma$ means the divergence of the data. sigma is larger, the results are more dispersed. $\sigma$ should be around $0.5 \rightarrow 1.5$ $Learning\ rate $ should be around $0.5 \rightarrow 2.5$. The map size should not more than $(10,10).$ <br/> The detail about hyper parameters can be found here [A simple demo for Hyper parameters](https://share.streamlit.io/justglowing/minisom/dashboard/dashboard.py, "demo")


### To do list on June 4th

1. <del>Tidy the source code. </del>
2. Change the dataset. 
    * *Socketsense*
    * *Itrust*
3. <del>Add a new block for anomaly detecion in Spatial space. </del> It needs to test more to avoid FP.
4. <del>The hyper-parameters needs to be clarify in order to get a better results in SOM.</del>
5. Temporal anomaly detection.
    * Investigation in Markov Chain
    * Investigation in Affinity Matrix.(Continue to abstract for the SOM neuron).
6. Confirm the Spatial anomaly detection and V&V results.

___




