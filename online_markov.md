1. Make sure to download all the packages and dependencies.
2. Examples of how to use the class are in the test function

The code is constructed as a class so the user needs to initiate the class as:
instance_name = MCTransitionMatrix(num_of_possible_states, U, AC, states).
In practice if the class is initiated without a previous simulation only instance_name = MCTransitionMatrix(num_of_possible_states) is needed.

From there everytime there is a state transition the user should call instance_name.update_transition(i, j). This can of course be done inside a loop.

If a the user wants to save a simulation, get_full_transition_matrix() should be used. get_full_transition_matrix() returns an N0xN0 matrix, U, AC, and a state transition history.

If the user want to use the transition matrix 
get_actual_transition_matrix() should be used. get_actual_transition_matrix() will return a reduced matrix P and all the states within P.

If the user wants to prune transient states from the transition matrix, prune(threshold_ratio) should be used. prune(threshold_ratio) will remove states that have been visited at least once and their count divided by the total amount of states is less than or equal to threshold_ratio. It is up to the user to set the threshold_ratio.

If the user wants to get the stable distribution, 
get_stable_distribution() should be used. get_stable_distribution() calculates the stable distribution with QR-decomposition.

If the user wants to see if the stable distibution is periodic, the results from get_stable_dist_brute() should be compared to 
get_stable_dist(). get_stable_dist_brute() will show the stable distibution within the period while get_stable_dist() will show what it converges to. So if the results of get_stable_dist_brute() and get_stable_dist() are not equal, that might indicate a periodic distribution.



