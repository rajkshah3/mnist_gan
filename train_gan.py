#gan training script
from models import ResNet, Disciminator, Classifier
import keras
from keras.datasets import mnist
import numpy as np 
import tensorflow as tf
import os

def train(tpu=False):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.expand_dims(x_train,-1)
    x_test = np.expand_dims(x_test,-1)

    backbone = ResNet()
    discriminator = Disciminator(backbone)
    classifier = Classifier(backbone,10)
    # import pdb; pdb.set_trace()  # breakpoint 396fe169 //
    classifier.predict(x_test)
    classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
    classifier.summary()

    if(tpu):
        classifier = convert_model_for_tpu(classifier)

    classifier.fit(x=x_train,y=y_train,batch_size=1000,epochs=1, validation_data=(x_test,y_test))

    classifier.save('classifier.h5')
    print('done')

def convert_model_for_tpu(model):
    strategy = tf.contrib.tpu.TPUDistributionStrategy(
        tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu='grpc://'+os.environ['COLAB_TPU_ADDR']))
    return tf.contrib.tpu.keras_to_tpu_model(model,strategy=strategy)

if __name__ == '__main__':
    train()