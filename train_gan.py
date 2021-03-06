#gan training script
from models import ResNet, Discriminator, Classifier, ResGen, Unet, mnist_data, GAN, Generator
import tensorflow.keras as keras
from tensorflow.keras import optimizers
import numpy as np 
import tensorflow as tf
import os
#todo
#Check if backbone is copied by value or reference
#How to freeze layers
#Environment for transfer learning and swapping between models easily
bce = tf.keras.losses.BinaryCrossentropy()


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

def get_gan_data(images,input_shape,seed=None):

    actuals = np.random.randint(0, 2,size=images.shape[0])
    random_noise_data = get_n_random_inputs_for_gan(images.shape[0],input_shape,seed=seed)
    
    return [random_noise_data,images,actuals], actuals

def get_n_random_inputs_for_gan(n,input_shape,seed=None):
    if seed:
        np.random.seed(seed)
    rand_data_shape = ((n,) + input_shape[1:3] + (1,))
    random_noise_data = np.random.normal(loc=0,scale=100,size=rand_data_shape)
    return random_noise_data

def generator_loss(y_true, y_pred, sample_weight=None):
    # y_true are the actual images 
    y_true = tf.ones_like(y_true) - y_true
    return bce(y_true,y_pred,sample_weight=y_true)

def discriminator_loss(y_true, y_pred, sample_weight=None):
    return bce(y_true,y_pred,sample_weight)

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
    generator = ResGen(backbone,'MnistGenerator')
    if(generator_weights):
        generator.load_weights(generator_weights)
    if(backbone_weights):
        backbone.load_weights(backbone_weights)

    return generator

def load_generator_gen(generator_weights='generator_weights.h5',clear_session=True):
    if(clear_session):
        keras.backend.clear_session()

    inputs = np.ones((100,7,7,1))
    # import pdb; pdb.set_trace()  # breakpoint 9e595caa //
    generator = Generator('MnistGenerator')
    generator(inputs)

    if(generator_weights):
        generator.load_weights(generator_weights)

    return generator

def load_gan(backbone_data=None,gan_weights='gan_weights.h5',
    backbone_weights='backbone_posttrained_weights.h5',
    generator_weights=None,
    discriminator_weights=None,
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

    gan = GAN(generator=generator,discriminator=discriminator)

    if(gan_weights):
        input_shape   = generator.get_input_shape()
        input_x, _ = get_gan_data(backbone_data.get_n_samples(10)[0],input_shape)
        gan.predict(input_x)
        gan.load_weights(gan_weights)

    return gan

def load_gan2(backbone_data=None,gan_weights='gan_weights.h5',
    backbone_weights='backbone_posttrained_weights.h5',
    generator_weights=None,
    discriminator_weights=None,
    clear_session=True):

    if(clear_session):
        keras.backend.clear_session()
    
    backbone = ResNet()
    backbone(backbone_data.get_test()[0])
    # generator = ResGen(backbone)

    discriminator = load_discriminator(data=backbone_data,clear_session=False,backbone_weights=backbone_weights,discriminator_weights=discriminator_weights)
    generator     = load_generator_gen(clear_session=False,generator_weights=generator_weights)

    # x = discriminator(inp)
    # x = generator(x)
    # return  keras.model(inputs=[inp],outputs=[x])

    gan = GAN(generator=generator,discriminator=discriminator)

    input_shape   = (-1,7,7,1)
    input_x, _ = get_gan_data(backbone_data.get_n_samples(10)[0],input_shape)
    gan.predict(input_x)

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

def load_discriminator(data=None,discriminator_weights=None,backbone_weights='backbone_posttrained_weights.h5',clear_session=True):
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
    classifier.fit(x=data.get_n_samples(35)[0],y=data.get_n_samples(35)[1],batch_size=6000,epochs=1, validation_data=data.get_vali())


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

sgd1 = optimizers.SGD(lr=0.01)
sgd2 = optimizers.SGD(lr=0.001)

def test_gan(generate=False,gan_weights=None,epochs=1,training_steps=100,gen_batch_size=300,dis_batch_size=1000,train_images=3000):
    data = mnist_data()

    gan = load_gan2(backbone_data=data,gan_weights=gan_weights,backbone_weights=None,
        generator_weights=None,discriminator_weights= None,clear_session=True)

    generator = gan.get_generator()
    input_shape   = generator.get_input_shape()

    # discriminator = gan.get_discriminator()
    images = data.get_n_samples(35)[0]

    train_data_x, train_data_y = get_gan_data(images,input_shape)
    # validation_data_x, validation_data_y = get_gan_data(data.get_n_samples(200)[0],input_shape)
    validation_data_x, validation_data_y = get_gan_data(data.get_vali()[0],input_shape,seed=128)


    outputs = gan.predict(train_data_x,batch_size=12)
    # gan.train()
    # outputs = gan.predict(random_noise_data,batch_size=32)

    # All parameter gradients will be clipped to
    # a maximum norm of 1.


    for i in range(training_steps):
        images = data.get_randn_samples(train_images)[0]
        train_data_x, train_data_y = get_gan_data(images,input_shape)
        stop_loop =  False
        if(generate):
            print('training Generator')
            gan.set_mode_to_generate()
            gan.compile(optimizer=sgd1,loss=generator_loss,metrics=['accuracy',generator_loss,discriminator_loss])
            generate = False
            batch_size = gen_batch_size
        else:
            print('training Discriminator')
            gan.set_mode_to_discriminate()
            gan.compile(optimizer=sgd2,loss=discriminator_loss,metrics=['accuracy',generator_loss,discriminator_loss])
            generate = True
            batch_size = dis_batch_size

        epoch = 0
        
        while epoch < epochs:
            output = gan.fit(x=train_data_x,y=train_data_y,batch_size=batch_size,epochs=1, validation_data=(validation_data_x, validation_data_y),callbacks=[])
            hist = output.history

            if(generate):
                if(hist['val_accuracy'][-1]>0.9 or hist['accuracy'][-1]>0.95):
                    epoch = epochs
            else:
                if(hist['val_accuracy'][-1]<0.6 or hist['accuracy'][-1]<0.6):
                    epoch = epochs 

            epoch = epoch + 1    
        gan.save_weights('gan_weights.h5')


    rand_test_data = get_n_random_inputs_for_gan(10,input_shape)
    test_images  = generator(rand_test_data)
    train_images = generator(train_data_x[0][:10])


    return gan, validation_data_x

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
    classifier.fit(x=data.get_n_samples(35)[0],y=data.get_n_samples(35)[1],batch_size=6000,epochs=20, validation_data=data.get_vali(),callbacks=[checkpoint])
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

def train_classifier(epochs=2):
    data = mnist_data()

    classifier = load_classifier(data=data,classes=10,classifier_weights=None,backbone_weights=None)
    
    classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    classifier.predict(data.get_test()[0])
    classifier.summary()
    classifier.fit(x=data.get_train()[0],y=data.get_train()[1],batch_size=1000,epochs=epochs, validation_data=data.get_vali())


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
    # train_classifier()
    # test_gan()
    test_gan(generate=True,gan_weights=None,training_steps=1,epochs=3,train_images=100)
    # test_gan(generate=True,gan_weights='gan_weights.h5')