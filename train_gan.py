#gan training script
from models import ResNet, Disciminator, Classifier
import keras
from keras.datasets import mnist
import numpy as np 
import tensorflow as tf
import os

def calc_averages(x):
    averages = np.average(x,axis=0)
    return averages

def calc_stds(x):
    stds = np.std(x,axis=0)
    return stds

def normalise_inputs(x,averages,stds):
    x = [np.divide(a - averages,stds) for a in x]
    return np.array(x)

def train(tpu=False):
    # Detect hardware
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
    except ValueError: # If TPU not found
        tpu = None

    if tpu:
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu, steps_per_run=128)
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])  
    else:
        strategy = tf.distribute.get_strategy() # Default strategy that works on CPU and single GPU
        print('Running on CPU instead')

    scope = strategy.scope()

    print("Number of accelerators: ", strategy.num_replicas_in_sync)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    averages = calc_averages(x_train)
    stds = calc_stds(x_train)

    x_train = normalise_inputs(x_train,averages,stds)
    x_test = normalise_inputs(x_test,averages,stds)

    x_train = np.expand_dims(x_train,-1)
    x_test = np.expand_dims(x_test,-1)

    x_vali  = x_train[40000:50000]
    y_vali  = y_train[40000:50000]

    x_disc  = x_train[50000:]
    y_disc  = y_train[50000:]

    x_train = x_train[:40000]
    y_train = y_train[:40000]

    backbone = ResNet()
    discriminator = Disciminator(backbone)
    classifier = Classifier(backbone,10)
    preds = classifier.predict(x_test)
    classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    classifier.summary()

    if(tpu):
        classifier = convert_model_for_tpu(classifier)
    checkpoint = keras.callbacks.ModelCheckpoint('./checkpoints/classifier/classifier_{epoch:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True)
    classifier.fit(x=x_train,y=y_train,batch_size=6000,epochs=1, validation_data=(x_vali,y_vali),callbacks=[checkpoint])
    # classifier.fit(x=x_train,y=y_train,batch_size=6000,epochs=30, validation_data=(x_vali,y_vali),callbacks=[checkpoint])
    # import pdb; pdb.set_trace()  # breakpoint 396fe169 //
    print('done')
    return (classifier, x_test)

def convert_classifier_to_discriminator(model):
    backbone = model.get_backbone()
    return Disciminator(backbone)

def convert_model_for_tpu(model):
    strategy = tf.contrib.tpu.TPUDistributionStrategy(
        tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu='grpc://'+os.environ['COLAB_TPU_ADDR']))
    return tf.contrib.tpu.keras_to_tpu_model(model,strategy=strategy)

if __name__ == '__main__':
    train()