import pandas as pd 
import numpy as np 
import tensorflow as tf 
import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
import csv
import cv2
import glob

###### STEP 0: Load the Data ##############################################
# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = "../data/train.p"
validation_file = "../data/valid.p"
testing_file = "../data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# online_images = glob.glob('sign*.jpg')
X_test2 = []
# for idx, fname in enumerate(online_images):
#     image = cv2.imread(fname)
#     img = cv2.resize(image, (32,32))
#     X_test2.append(img)

image = cv2.imread('sign6.JPG')
img = cv2.resize(image, (32,32))
X_test2.append(img)
image = cv2.imread('sign7.JPG')
img = cv2.resize(image, (32,32))
X_test2.append(img)
image = cv2.imread('sign8.JPG')
img = cv2.resize(image, (32,32))
X_test2.append(img)
image = cv2.imread('sign9.JPG')
img = cv2.resize(image, (32,32))
X_test2.append(img)
image = cv2.imread('sign10.JPG')
img = cv2.resize(image, (32,32))
X_test2.append(img)
y_test2 = [12,33,5,40,24]


###### STEP 1: Dataset Summary & Exploration ##############################
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)
n_test2 = len(X_test2)

for image in X_train:
  image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

for image in X_valid:
  image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

for image in X_test:
  image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
f = open('signnames.csv')
lines = f.readlines()

classes = []
for line in lines:
  num_strs = line.split(',')
  try:
    label = float(num_strs[0])
  except ValueError:
    pass
  else:
    classes.append(label)

n_classes = len(classes)

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

index = random.randint(0, n_test2)
image = X_test2[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
# plt.savefig('../visual.jpg')
print(y_test2[index])

######## STEP 2: Design and Test a Model Architecture ####################
X_train, y_train = shuffle(X_train, y_train)

EPOCHS = 20
BATCH_SIZE = 128

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.05
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 10), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(10))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 10, 25), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(25))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(625, 425), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(425))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(425, 250), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(250))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(250, 120), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(120))
    fc3    = tf.matmul(fc2, fc3_W) + fc3_b
    
    # SOLUTION: Activation.
    fc3    = tf.nn.relu(fc3)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc4_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc4_b  = tf.Variable(tf.zeros(84))
    fc4    = tf.matmul(fc3, fc4_W) + fc4_b
    
    # SOLUTION: Activation.
    fc4    = tf.nn.relu(fc4)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc5_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    fc5_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc4, fc5_W) + fc5_b
    
    return logits


x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

# Training pipeline
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
#############################################################################################

# Model Evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
softmax_operation = tf.nn.softmax(logits)
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        softmax = sess.run(softmax_operation, feed_dict={x: batch_x, y: batch_y})
        # result = sess.run(tf.nn.top_k(tf.constant(cross_entropy), k=5))
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def evaluate_online_images(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        softmax = sess.run(softmax_operation, feed_dict={x: batch_x, y: batch_y})
        top_k = sess.run(tf.nn.top_k(tf.constant(softmax), k=5))
        print(top_k)
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
###############################################################################################

# Train the Model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate_online_images(X_test2, y_test2)
    print("Test Accuracy = {:.3f}".format(test_accuracy))



        