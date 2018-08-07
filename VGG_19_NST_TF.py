#Import Dependencies
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import os
import sys
import tensorflow as tf
from PIL import Image
from nst_utils import *

%matplotlib inline

#Load parameters of VGG_19 model in order to apply transfer learning on it.
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
print(model)

#The Model is stored in a python dictionary
#To assign an Image to the model use model["input"].assign(image)
#To get the activation of a particular layer use sess.run(model["conv4_2"])

content_image = scipy.misc.imread("folder/file.jpg")
imshow(content_image)

#Technically we don't need to convert the 3D representation of hidden layer into 2D for Content Cost but it is required for Style cost.
#We have done 3D -> 2D just to know it before we start calculating style cost.

#Compute Content Cost
def compute_content_cost(a_C, a_G):
  m, n_H, n_W, n_C = a_G.get_shape().as_list()
  
  a_C_unrolled = tf.reshape(a_C, [m, -1, n_C])
  a_G_unrolled = tf.reshape(a_G, [m, -1, n_C])
  
  J_content = 1 / (4 * n_H * n_W * n_C) * tf.reduce_sum(tf.squared_difference(a_C_unrolled, a_G_unrolled))
  
  return J_content
  
tf.reset_default_graph()
with tf.Session() as sess:
  a_C = tf.random_normal([1, 4, 4, 3], mean = 1, stddev = 4)
  a_G = tf.random_normal([1, 4, 4, 3], mean = 1, stddev = 4)
  J_content = compute_content_cost(a_C, a_G)
  print("J_content = " + str(J_content.eval()))

style_image = scipy.misc.imread("folder/file.jpg")
imshow(style_image)

#One important part of the gram matrix is that the diagonal elements such as $G_{ii}$ also measures how active filter $i$ is. For example, suppose filter $i$ is detecting vertical textures in the image. Then $G_{ii}$ measures how common vertical textures are in the image as a whole: If $G_{ii}$ is large, this means that the image has a lot of vertical texture.
#By capturing the prevalence of different types of features ($G_{ii}$), as well as how much different features occur together ($G_{ij}$), the Style matrix $G$ measures the style of an image.
#Style Matrix or Gram Matrix
def gram_matrix(A):
  GA = tf.matmul(A, tf.transpose(A))
  
  return GA

tf.reset_default_graph()
with tf.Session() as sess:
  A = tf.random_normal([3, 2 * 1], mean = 1, stddev = 4)
  GA = gram_matrix(A)
  
  print("GA = " + str(GA.eval()))

#Style Cost for a single layer
def compute_layer_style_cost(a_S, a_G):
  m, n_H, n_W, n_C = a_G.get_shape().as_list()
  
  a_S = tf.transpose(tf.reshape(a_S, [n_H * n_W, n_C]))
  a_G = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))
  
  GS = gram_matrix(a_S)
  GG = gram_matrix(a_G)
  
  J_style_layer = 1 / (4 * n_C**2 * n_H**2 * n_W**2) * tf.reduce_sum(tf.squared_difference(GS, GG))
  
  return J_style_layer
  
tf.reset_default_graph()
with tf.Session() as test:
  a_S = tf.random_normal([1, 4, 4, 3], mean = 1, stddev = 4)
  a_G = tf.random_normal([1, 4, 4, 3], mean = 1, stddev = 4)
  J_style_layer = compute_layer_style_cost(a_S, a_G)
  
  print("J_style_layer = " + str(J_style_layer.eval()))
  
#Style Weights
STYLE_LAYERS = [
  ("conv1_1", 0.1),
  ("conv2_1", 0.2),
  ("conv3_1", 0.4),
  ("conv4_1", 0.2),
  ("conv5_1", 0.1)]

#Weighted Style Cost involving many layers
def compute_style_cost(model, STYLE_LAYERS):
  J_style = 0
  
  for layer_name, coeff in STYLE_LAYERS:
    out = model[layer_name]
    a_S = sess.run(out)
    a_G = out
    J_style_layer = compute_layer_style_cost(a_S, a_G)
    J_style += coeff * J_style_layer
    
  return J_style
#Note: In the for-loop above, a_G is a tensor and hasn't been evaluated yet. It will be evaluated and updated at each iteration when we run the TensorFlow graph in model_nn() below.
#The style of an image can be represented using the Gram matrix of a hidden layer's activations. However, we get even better results combining this representation from multiple different layers. This is in contrast to the content representation, where usually using just a single hidden layer is sufficient.
#Minimizing the style cost will cause the image $G$ to follow the style of the image $S$.

#Total Cost
def total_cost(J_content, J_style, alpha = 10, beta = 40):
  J = alpha * J_content + beta * J_style
  
  return J
  
tf.reset_default_graph()
with tf.Session() as sess:
  J_content = np.random.randn()
  J_style = np.random.randn()
  J = total_cost(J_content, J_style)
  print("J = " + str(J))

tf.reset_default_graph()
sess = tf.InteractiveSession()

content_image = scipy.misc.imread("folder/file.jpg")
content_image = reshape_and_normalize_image(content_image)

style_image = scipy.misc.imread("folder/file.jpg")
style_image = reshape_and_normalize_image(content_image)

#By initializing the pixels of the generated image to be mostly noise but still slightly correlated with the content image, this will help the content of the "generated" image more rapidly match the content of the "content" image.
generated_image = generate_noise_image(content_image)
imshow(generated_image[0])

#Load the model
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

sess.run(model["input"].assign(content_image))
out = model["conv4_2"]
a_C = sess.run(out)
a_G = out
J_content = compute_content_cost(a_C, a_G)

sess.run(model["input"].assign(style_image))
J_style = compute_style_cost(model, STYLE_LAYERS)

J = total_cost(J_content, J_style)

optimizer = tf.train.Adamoptimizer(2.0)

train_step = optimizer.minize(J)

#Model VGG-19
def model_nn(sess, input_image, num_iterations = 200):
  sess.run(tf.global_variables_initializer())
  sess.run(model["input"].assign(input_image))
  
  for i in range(num_iterations):
    sess.run(train_step)
    generated_image = sess.run(model["input"])
    
    if i%20 == 0:
      Jt, Jc, Js = sess.run([J, J_content, J_style])
      print("Iteration " + str(i) + ":")
      print("total cost = " + str(Jt))
      print("content cost = " + str(Jc))
      print("style cost = " + str(Js))
      
      save_image("output/" + str(i) + ".png", generated_image)
      
  save_image("output/generated_image.jpg", generated_image)
  
  return generated_image

model_nn(sess, generated_image)
