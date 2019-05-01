"""
tf.tensordot(A, B, [[1], [0]])

A dot B via dimension 1 of A and dimension 0 of B

If A is 2 by 3 and b is 3 by 2, then classically, result is 2 by 2 (1 is columns, 0 is rows)

If A is (2 by "4" by 4) and B is ("4" by 3) then the result is (2 by 4 by 3) since the "4" dimension is removed


tf.tensordot(A, B, [[1,2], [0,1]])

A is [2, 4, 4], B is [4, 4], then result is dimension [2] (removing the 4's)

"""