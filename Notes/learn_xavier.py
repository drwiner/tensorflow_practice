"""
Xavier Initialization

Used to set the standard deviation when randomly initializing variables

Let n_i be the number of connections coming from the layer (input),
	and n_0 be the output of connections out of the layer (output)
	
	e.g. n_i = 784 (image pixels) and n_0 = 10 (labels)
	
	xavier (sigma) = sqrt (2   / (n_i + n_o))
	sqrt(2/794) ~ 0.0502
	

Variance = expected value of the squared difference between value and mean
E[X] = probabilistic average of possible values = Sum_x (p(X = x) * x
e.g. six-sided die E[roll] = sum (1/6 * i for i in range(6) = 3.5
"""