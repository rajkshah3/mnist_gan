#gan training script
from models import ResNet, Discriminator, Classifier, ResGen, Unet, mnist_data, GAN
import tensorflow.keras as keras
import numpy as np 
import tensorflow as tf
import os

#todo
#Check if backbone is copied by value or reference
#How to freeze layers
#Environment for transfer learning and swapping between models easily


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


def test_resgen():
    data = mnist_data()

    backbone = ResNet()
    preds = backbone(data.get_test()[0])
    gen = ResGen(backbone)
    input_shape = gen.get_input_shape()
    print(gen.get_output_shape())
    rand_data_shape = ((50,) + input_shape[1:] + (1,))
    random_noise_data = np.random.normal(size=rand_data_shape)
    # import pdb; pdb.set_trace()  # breakpoint 7e7a66fc //
    preds = gen.predict(random_noise_data)

    return True

def load_generator(backbone_data=None,generator_weights='generator_weights.h5',backbone_weights='backbone_posttrained_weights.h5',clear_session=True):
    if(clear_session):
        keras.backend.clear_session()

    backbone = ResNet()
    backbone(backbone_data.get_test()[0])
    # import pdb; pdb.set_trace()  # breakpoint 9e595caa //
    generator = ResGen(backbone)
    if(generator_weights):
        generator.load_weights(generator_weights)
    if(backbone_weights):
        backbone.load_weights(backbone_weights)

    return generator

def load_gan(backbone_data=None,gan_weights='gan_weights.h5',
    backbone_weights='backbone_posttrained_weights.h5',
    generator_weights=None,
    discriminator_weights='None',
    clear_session=True):

    if(clear_session):
        keras.backend.clear_session()
    
    backbone = ResNet()
    backbone(backbone_data.get_test()[0])
    # generator = ResGen(backbone)

    discriminator = load_discriminator(data=backbone_data,clear_session=False,backbone_weights=backbone_weights,discriminator_weights=discriminator_weights)
    generator     = load_generator(backbone_data=backbone_data,clear_session=False,backbone_weights=backbone_weights,generator_weights=generator_weights)

    # x = discriminator(inp)
    # x = generator(x)
    # return  keras.model(inputs=[inp],outputs=[x])

    input_shape   = generator.get_input_shape()

    rand_data_shape = ((50,) + input_shape[1:] + (1,))
    random_noise_data = np.random.normal(size=rand_data_shape)

    gan = GAN(generator=generator,discriminator=discriminator)

    if(gan_weights):
        gan.load_weights(gan_weights)

    return gan

def load_classifier(data=None,classes=10,classifier_weights='classifier_weights.h5',backbone_weights='backbone_posttrained_weights.h5',clear_session=True):
    if(clear_session):
        keras.backend.clear_session()

    backbone = ResNet()
    backbone(data.get_test()[0])
    classifier = Classifier(backbone,classes)
    if(classifier_weights):
        classifier.load_weights(classifier_weights)
    if(backbone_weights):
        backbone.load_weights(backbone_weights)

    return classifier

def load_discriminator(data=None,discriminator_weights='discriminator_weights.h5',backbone_weights='backbone_posttrained_weights.h5',clear_session=True):
    if(clear_session):
        keras.backend.clear_session()
    backbone = ResNet()
    backbone(data.get_test()[0])
    discriminator = Discriminator(backbone)
    if(discriminator_weights):
        discriminator.load_weights(discriminator_weights)
    if(backbone_weights):
        backbone.load_weights(backbone_weights)

    discriminator(data.get_test()[0])

    return discriminator

def test_coupled_weights_of_backbone():
    """
    This function will fail because there are multiple models defined 
    in the keras/tensorflow graph which are not used during training. 

    Returns:
        bool -- [description]
    """
    data = mnist_data()
    backbone = ResNet()

    preds = backbone(data.get_test()[0])
    gen = ResGen(backbone)
    input_shape = gen.get_input_shape()
    rand_data_shape = ((50,) + input_shape[1:] + (1,))
    random_noise_data = np.random.normal(size=rand_data_shape)

    discriminator = Discriminator(backbone)
    classifier = Classifier(backbone,10)

    discriminator_predicitons_1 = discriminator(data.get_test()[0])    
    classifier_predicitons_1 = classifier.predict(data.get_test()[0])
    generator_predictions_1 = gen.predict(random_noise_data)[0]

    classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    classifier.summary()
    # classifier.fit(x=x_train,y=y_train,batch_size=6000,epochs=1, validation_data=(x_vali,y_vali),callbacks=[checkpoint])
    classifier.fit(x=data.get_train()[0],y=data.get_train()[1],batch_size=6000,epochs=1, validation_data=data.get_vali())


    discriminator_predicitons_2 = discriminator(data.get_test()[0])    
    classifier_predicitons_2 = classifier.predict(data.get_test()[0])
    generator_predictions_2 = gen.predict(random_noise_data)[0]

    discriminator_diff = discriminator_predicitons_1 - discriminator_predicitons_2
    classifier_diff = classifier_predicitons_1 - classifier_predicitons_2
    generator_diff = generator_predicitons_1 - generator_predicitons_2

    return True

def test_unet():
    data = mnist_data()

    backbone = ResNet()
    # preds = backbone(data.get_test()[0])
    gen = Unet()
    # input_shape = gen.get_input_shape()
    # print(gen.get_output_shape())
    rand_data_shape = (50,28,28,1)
    random_noise_data = np.random.normal(size=rand_data_shape)
    # import pdb; pdb.set_trace()  # breakpoint 7e7a66fc //
    preds = gen.predict(random_noise_data)
    return True

def test_gan():
    data = mnist_data()

    gan = load_gan(backbone_data=data,gan_weights=None,backbone_weights='backbone_trained_weights.npy',generator_weights=None,discriminator_weights= None,clear_session=True)

    generator = gan.get_generator()
    input_shape   = generator.get_input_shape()

    # discriminator = gan.get_discriminator()

    images = data.get_n_samples(100)[0]
    actuals = np.random.randint(0, 1,size=images.shape[0])

    rand_data_shape = ((images.shape[0],) + input_shape[1:] + (1,))
    random_noise_data = np.random.normal(size=rand_data_shape)
    
    outputs = gan.predict([random_noise_data,images,actuals],batch_size=12)
    # gan.train()
    # outputs = gan.predict(random_noise_data,batch_size=32)
    gan.set_mode_to_discriminate()
    gan.fit(x=[random_noise_data[:100],images[:100],actuals[:100]],y=actuals[:100],batch_size=6000,epochs=1, validation_data=data.get_vali(),callbacks=[checkpoint])

    gan.save_weights('gan_weights.h5')

    return True

def train_classifier_depricated(tpu=False):

    scope = strategy.scope()

    print("Number of accelerators: ", strategy.num_replicas_in_sync)

    data = mnist_data()

    backbone = ResNet()
    discriminator = Discriminator(backbone)
    classifier = Classifier(backbone,10)
    preds = classifier.predict(data.get_test()[0])

    classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    classifier.summary()

    if(tpu):
       classifier = convert_model_for_tpu(classifier)

    checkpoint = keras.callbacks.ModelCheckpoint('./checkpoints/classifier/classifier_{epoch:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True)
    # classifier.fit(x=x_train,y=y_train,batch_size=6000,epochs=1, validation_data=(x_vali,y_vali),callbacks=[checkpoint])
    classifier.fit(x=data.get_n_samples(100)[0],y=data.get_n_samples(100)[1],batch_size=6000,epochs=1, validation_data=data.get_vali(),callbacks=[checkpoint])
    # import pdb; pdb.set_trace()  # breakpoint 396fe169 //
    backbone =  classifier.get_backbone()
    backbone.save_weights('backbone_weights.h5')
    return (classifier, x_test)

def convert_classifier_to_discriminator(model):
    backbone = model.get_backbone()
    return Discriminator(backbone)

def convert_model_for_tpu(model):
    strategy = tf.contrib.tpu.TPUDistributionStrategy(
        tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu='grpc://'+os.environ['COLAB_TPU_ADDR']))
    return tf.contrib.tpu.keras_to_tpu_model(model,strategy=strategy)

def train_classifier():
    data = mnist_data()

    classifier = load_classifier(data=data,classes=10,classifier_weights=None,backbone_weights=None)
    
    classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    classifier.predict(data.get_test_data()[0])
    classifier.summary()
    classifier.fit(x=data.get_n_samples(100)[0],y=data.get_n_samples(100)[1],batch_size=6000,epochs=1, validation_data=data.get_vali())


    backbone = classifier.get_backbone()
    backbone.save_weights('backbone_trained_weights.npy')

if __name__ == '__main__':
    tf.random.set_seed(
    24
    )
    np.random.seed(24)
    # train()
    # test_resgen()
    # test_unet()
    train_classifier()
    test_gan()
