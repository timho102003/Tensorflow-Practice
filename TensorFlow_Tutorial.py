import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image

# tf.assign
xx = tf.placeholder(tf.float32, name='xx')
Wi = tf.Variable([0.5], name='Weight')
b = tf.Variable([1.2], name='bias')

LinearModel = Wi*xx+b

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(LinearModel, feed_dict={xx:[3.5, 4.0, 7.2, 5.3, 6.1]}))

yy = tf.placeholder(tf.float32, name='yy')
cost = tf.reduce_sum(tf.square(LinearModel-yy))

sess.run(tf.global_variables_initializer())
print(sess.run(cost, feed_dict={xx:[3.5, 4.0, 7.2, 5.3, 6.1], yy : [1.0, 2.0, 3.0, 4.0, 5.0]}))

f_Wi = tf.assign(Wi, [0.75])
f_b = tf.assign(b, [0.004])
sess.run([f_Wi, f_b])
print(sess.run(cost, feed_dict={xx:[3.5, 4.0, 7.2, 5.3, 6.1], yy : [1.0, 2.0, 3.0, 4.0, 5.0]}))

#writer = tf.summary.FileWriter('log/Practice_Log', tf.get_default_graph())
#writer.close()

'''
# Placeholder
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = tf.add(x, y)

with tf.Session() as sess:
    print(sess.run(z, feed_dict={x:5, y:6}))
'''
'''
# Import three dimensional images
img = image.imread('Tim.png')
print(img.ndim)
print(img.shape)

plt.imshow(img)
plt.show()

img_tf = tf.Variable(img, name = 'img_tf')
model = tf.global_variables_initializer()
print(img_tf.get_shape().as_list())

with tf.Session() as sess:
    img_tf = tf.transpose(img_tf, perm=[1,0,2])
    sess.run(model)
    result = sess.run(img_tf)

plt.imshow(result)
plt.show()
'''
'''
# Seed and uniform distribution
X_Seed = tf.random_uniform([1], seed=2)
X_NoSeed = tf.random_uniform([1])

# 有Seed的會重複數字，相反的沒有Seed就不會
with tf.Session() as Fisrt_Sess:
    print('With Seed = 2 --> {}'.format(Fisrt_Sess.run(X_Seed)))
    print('With Seed = 2 --> {}'.format(X_Seed.eval()))
    print('No Seed --> {}'.format(X_NoSeed.eval()))
    print('No Seed --> {}'.format(X_NoSeed.eval()))

with tf.Session() as Second_Sess:
    print('With Seed = 2 --> {}'.format(X_Seed.eval()))
    print('With Seed = 2 --> {}'.format(X_Seed.eval()))
    print('No Seed --> {}'.format(X_NoSeed.eval()))
    print('No Seed --> {}'.format(X_NoSeed.eval()))


x = tf.random_uniform([200], minval=0, maxval=1, dtype=tf.float64)
sess = tf.InteractiveSession()
plt.hist(x.eval(), normed=True)
plt.show()
#with tf.Session() as sess:
#     plt.hist(sess.run(x), normed=True)
#     plt.show()
'''
'''
# Gradient
x = tf.placeholder(tf.float16)
y = x**2
gradient = tf.gradients(y,x)
with tf.Session() as sess:
    print(sess.run(gradient, feed_dict={x:5}))
'''

'''
Amatrix = np.array([(1.0,1.0,1.0), (1.0,1.0,1.0), (1.0,1.0,1.0)])
Bmatrix = np.array([(3.0,3.0,3.0), (3.0,3.0,3.0), (3.0,3.0,3.0)])

TF_A = tf.constant(Amatrix)
TF_B = tf.constant(Bmatrix)

print(TF_A)
print(TF_B)

ABMulti = tf.matmul(TF_A, TF_B)
ABSum = tf.add(TF_A, TF_B)
ABDet = tf.matrix_determinant(TF_A)

print(ABMulti)
print(ABSum)
print(ABDet)

with tf.Session() as sess:
    AResult = sess.run(ABMulti)
    BResult = sess.run(ABSum)
    CResult = sess.run(ABDet)

print(AResult)
print(BResult)
print(CResult)
'''
