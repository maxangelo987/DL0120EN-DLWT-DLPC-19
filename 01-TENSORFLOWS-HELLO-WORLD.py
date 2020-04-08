import tensorflow as tf

###############################################################
#BUILDING-A-GRAPH
###############################################################

graph1 = tf.Graph()

with graph1.as_default():
    a = tf.constant([2,7], name = 'constant_a')
    b = tf.constant([3], name = 'constant_b')

# Printing the value of a
sess = tf.Session(graph = graph1)
result = sess.run(a)
print(result)

sess.close()


with graph1.as_default():
    c = tf.add(a, b)
    #c = a + b is also a way to define the sum of the terms
sess = tf.Session(graph = graph1)

result = sess.run(c)
print(result)

sess.close()

with tf.Session(graph = graph1) as sess:
    result = sess.run(c)
    print(result)

###############################################################
#MULTIDIMENSIONAL-ARRAY-USING-TENSORFLOW
###############################################################
    
graph2 = tf.Graph()
with graph2.as_default():
    Scalar = tf.constant(2)
    Vector = tf.constant([5,6,2])
    Matrix = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
    Tensor = tf.constant( [ [[1,2,3],[2,3,4],[3,4,5]] , [[4,5,6],[5,6,7],[6,7,8]] , [[7,8,9],[8,9,10],[9,10,11]] ] )
with tf.Session(graph = graph2) as sess:
    result = sess.run(Scalar)
    print ("Scalar (1 entry):\n %s \n" % result)
    result = sess.run(Vector)
    print ("Vector (3 entries) :\n %s \n" % result)
    result = sess.run(Matrix)
    print ("Matrix (3x3 entries):\n %s \n" % result)
    result = sess.run(Tensor)
    print ("Tensor (3x3x3 entries) :\n %s \n" % result)

Scalar.shape
Tensor.shape

graph3 = tf.Graph()
with graph3.as_default():
    Matrix_one = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
    Matrix_two = tf.constant([[2,2,2],[2,2,2],[2,2,2]])

    add_1_operation = tf.add(Matrix_one, Matrix_two)
    add_2_operation = Matrix_one + Matrix_two

with tf.Session(graph =graph3) as sess:
    result = sess.run(add_1_operation)
    print ("Defined using tensorflow function :")
    print(result)
    result = sess.run(add_2_operation)
    print ("Defined using normal expressions :")
    print(result)

graph4 = tf.Graph()
with graph4.as_default():
    Matrix_one = tf.constant([[2,3],[3,4]])
    Matrix_two = tf.constant([[2,3],[3,4]])

    mul_operation = tf.matmul(Matrix_one, Matrix_two)

with tf.Session(graph = graph4) as sess:
    result = sess.run(mul_operation)
    print ("Defined using tensorflow function :")
    print(result)
    
###############################################################
#VARIABLES
###############################################################
    
v = tf.Variable(0)

