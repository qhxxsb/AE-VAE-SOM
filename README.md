# AE-VAE-SOM
# TODO：
- [ ] Learn how to optimize the learning process. (We can take inspiration from the package minisom)
- [ ] Calculate the transition matrix with the training data


---Updated May 27th:

Common Trajectory:

![image](https://user-images.githubusercontent.com/34424773/120104020-b6021180-c152-11eb-8750-93d58f7def63.png)


Faulty Trajectory:

![image](https://user-images.githubusercontent.com/34424773/120104023-c0bca680-c152-11eb-83fa-c387a7cacca0.png)

To Do:

Investigate correlation between the common input vs. faulty input

How to process the biodirectional arrow in the Common trajectroy?

How to down sample the z_e?

Markov Model 


---Updated June 4th:

Fixed up the Markov Chain. Need to be validated.

Note: The Markov Chain is not the only method to optimize the temporal model.

I will continue to work for this part after building the framework.

To do list:

Tidy the source code. 

Change the dataset. 
