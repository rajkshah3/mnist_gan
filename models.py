#Change keras models to layers!

import tensorflow.keras as keras
from tensorflow.keras.regularizers import l2,l1
import os
import numpy as np 
from tensorflow.keras.datasets import mnist
import tensorflow as tf

regularizer = None #l2(0.001)

def calc_averages(x):
    """Calcualte averages
    
    
    Arguments:
        x {[iterable]} -- [input data]
    
    Returns:
        [number] -- Average of numbers passed in
    """
    averages = np.average(x,axis=0)
    return averages

def calc_stds(x):
    """Calcualte standard deviations
    
    
    Arguments:
        x {[iterable]} -- [input data]
    
    Returns:
        [number] -- Standard deviations of numbers passed in
    """
    stds = np.std(x)
    return stds

def fix_weights(self,model):
    """Fix weights
    
    Function that fixes all weights in a model
    
    Arguments:
        model {[keras.model]} -- Output model with fixed weights
    """
    for layer in model.layers:
        layer.trainable = False

def unfix_weights(self,model):
    """UnFix weights
    
    Function that releases all weights in a model
    
    Arguments:
        model {[keras.model]} -- Output model with trainable weights
    """
    for layer in model.layers:
        layer.trainable = True

class ModelFromLayer(keras.Model):
    """Model from Layer
    
    Subclass of keras.model which produces a keras model 
    from a single keras layer.
    
    Extends:
        keras.Model
    """
    def __init__(self,layer):
        super(ModelFromLayer, self).__init__(name='ModelFromLayer')
        self.layer = layer

    def call(self,input):
        return self.layer(input)

class LayerABC(keras.layers.Layer):
    """Base class of a layer

    
    Extends:
        keras.layers.Layer
    """
    def load_weights(self,weights_file):
        weights = np.load(weights_file,allow_pickle=True)
        self.set_weights(weights)

    def save_weights(self,weights_file):
        weights = self.get_weights()
        np.save(weights_file,weights,allow_pickle=True)

class mnist_data():
    """Mnist data class
    
    """
    def __init__(self):
        (x_train, self.y_train), (x_test, self.y_test) = mnist.load_data()

        self.averages = calc_averages(x_train)
        self.stds = calc_stds(x_train)

        self.x_train = self.normalise_inputs(x_train)
        self.x_test = self.normalise_inputs(x_test)

        self.x_train = np.expand_dims(self.x_train,-1)
        self.x_test = np.expand_dims(self.x_test,-1)

        self.x_vali  = self.x_train[40000:50000]
        self.y_vali  = self.y_train[40000:50000]

        self.x_disc  = self.x_train[50000:]
        self.y_disc  = self.y_train[50000:]

        self.x_train = self.x_train[:40000]
        self.y_train = self.y_train[:40000]

    def get_vali(self):
        return self.x_vali, self.y_vali

    def get_train(self):
        return self.x_train, self.y_train

    def get_debug(self):
        return self.x_train[:10], self.y_train[:10]

    def get_n_samples(self,n):
        return self.x_train[:n], self.y_train[:n]

    def get_randn_samples(self,n):
        randints = np.random.randint(0,len(self.x_train),n)
        x = np.array([self.x_train[i] for i in randints])
        y = np.array([self.y_train[i] for i in randints])
        return x, y

    def get_test(self):
        return self.x_test, self.y_test

    def normalise_inputs(self,x):
        x = [np.divide(a - self.averages,self.stds) for a in x]
        return np.array(x)

    def reverse_normalisation(self,x):
        x = np.squeeze(x)
        x = [np.multiply(a,self.stds) + self.averages for a in x]
        return np.array(x,dtype=int)

class GAN(keras.Model):
    """GAN Class
    
    
    Extends:
        keras.Model
    """
    def __init__(self,generator,discriminator):
        super(GAN, self).__init__(name='GAN')
        self.generator = generator
        self.discriminator = discriminator
        # self.generator = ModelFromLayer(self.generator_layer)
        # self.discriminator = ModelFromLayer(self.discriminator_layer)
        self.set_mode_to_discriminate()
        self.me_loss_scale = 0.01

    def set_mode_to_generate(self):
        self.generator.trainable = True
        self.discriminator.trainable = False
        self.generate_mode = True

    def set_mode_to_discriminate(self):
        self.generator.trainable = False
        self.discriminator.trainable = True
        self.discriminator.freeze_batchnorms()
        self.generate_mode = False

    def get_generator(self):
        return self.generator

    def get_discriminator(self):
        return self.discriminator

    def calc_me_loss(self,candidates,examples):
        candidate_predictions = tf.math.reduce_sum(self.discriminator.backbone(candidates),0)
        example_predictions = tf.math.reduce_sum(self.discriminator.backbone(examples),0)
        diff = tf.math.abs(candidate_predictions - example_predictions)
        return self.me_loss_scale * tf.reduce_sum(diff)

    def save_weights(self,name):
        swapped = False
        if(self.generate_mode):
            self.set_mode_to_discriminate()
            swapped = True

        super(GAN, self).save_weights(name)
        
        if(swapped):
            self.set_mode_to_generate()

    def load_weights(self,name):
        swapped = False
        if(self.generate_mode):
            self.set_mode_to_discriminate()
            swapped = True

        super(GAN, self).load_weights(name)
        
        if(swapped):
            self.set_mode_to_generate()

    def compute_output_shape(self,input_shape):
        return self.discriminator.compute_output_shape(input_shape)

    def select_candidates_from_examples(self,candidates,examples,real_or_generated):
        real_or_generated = tf.cast(real_or_generated, tf.float32)

        # real_or_generated = keras.backend.expand_dims(real_or_generated,1)
        # cands = keras.backend.reshape(candidates,(-1,*examples.shape[1:]))
        # real_or_generated = tf.broadcast_to(real_or_generated,(-1,*examples.shape[1:]))  
        # exgs = examples * real_or_generated
        one_minus_real_or_generated = tf.ones_like(real_or_generated) - real_or_generated
        # if(self.generate_mode is True):
        #     saved = one_minus_real_or_generated
        #     one_minus_real_or_generated = real_or_generated

        #Manually Broadcast and multiplication implementation (probably faster but inelegant)
        #Outputs of code tested against einsum calculations

        # real_or_generated_broadcasted = real_or_generated
        # while len(real_or_generated_broadcasted.shape) < len(examples.shape):
        #     real_or_generated_broadcasted = tf.expand_dims(real_or_generated_broadcasted,-1)

        # for ni, i in enumerate(examples.shape[1:]):
        #     real_or_generated_broadcasted = tf.keras.backend.repeat_elements(real_or_generated_broadcasted,i,axis=ni+1)

        


        #Broadcast with tf tile instead
        # real_or_generated_broadcasted = tf.reshape(tf.tile(real_or_generated,[1,np.product(examples.shape[1:])]),(-1,*examples.shape[1:]))
        # one_minus_real_or_generated_broadcasted = tf.ones_like(real_or_generated_broadcasted) - real_or_generated_broadcasted

        # #Broadcast calculation
        # exgs = examples * real_or_generated_broadcasted
        # cands = candidates * one_minus_real_or_generated_broadcasted

        #Einsum implementation
        exgs = tf.einsum('abcd,ad->abcd',examples,real_or_generated)
        cands = tf.einsum('abcd,ad->abcd',candidates,one_minus_real_or_generated)
        return exgs + cands

    def call(self,inputs):
        noise = inputs[0]
        examples = inputs[1]
        real_or_generated =  inputs[2]
        candidates = self.generator(noise)
        if(self.generate_mode):
            self.add_loss(self.calc_me_loss(candidates,examples))
        inputs_for_discriminator = self.select_candidates_from_examples(candidates,examples,real_or_generated)
        outputs = self.discriminator(inputs_for_discriminator)
        return outputs


class Generator(LayerABC):
    """Generator Layer
    
    
    Extends:
        LayerABC
    """
    def __init__(self,name='UnNamed'):
        super(Generator, self).__init__(name='Generator_{}'.format(name))
        #28,28
        self.upsample   = keras.layers.UpSampling2D(size=(2, 2),interpolation='nearest')
        self.c1     = keras.layers.Conv2D(256,padding='same',kernel_size=(4,4))
        self.activation = keras.layers.LeakyReLU()
        self.comb_conv1 = keras.layers.Conv2D(256,padding='same',kernel_size=(4,4))

        self.comb_conv2 = keras.layers.Conv2D(256,padding='same',kernel_size=(4,4))

        self.conv_out = keras.layers.Conv2D(1,padding='same',kernel_size=(1,1))
        self.scale = 5
        self.final_activation = keras.layers.Activation(tf.nn.tanh)

        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.bn3 = keras.layers.BatchNormalization()

    def compute_output_shape(self,input_shape):
        return self.get_output_shape()

    def get_output_shape(self):
        return self.conv_out.output_shape

    def get_input_shape(self):
        return self.iinput_shape

    def call(self,inputs):
        self.iinput_shape = inputs.shape
        x = self.c1(inputs)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.upsample(x)
        x = self.comb_conv1(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.upsample(x)
        x = self.comb_conv2(x)
        x = self.bn3(x)
        x = self.conv_out(x)
        x = tf.multiply(self.final_activation(x),self.scale)
        return x


class ResGen(LayerABC):
    """ ResGenerator Class
    
    Generator class capable of using RESNET trained weights
    to initialise.
    
    Extends:
        LayerABC
    """
    def __init__(self,backbone,name='UnNamed'):
        super(ResGen, self).__init__(name='ResGen_{}'.format(name))
        #28,28
        self.backbone = backbone
        self.upsample   = keras.layers.UpSampling2D(size=(2, 2),interpolation='nearest')
        resnet_input_shape = self.backbone.block9.get_input_shape()
        self.c1     = keras.layers.Conv2D(resnet_input_shape[-1],padding='same',
            kernel_size=(2,2))
        self.block9 = self.backbone.block9
        self.block8 = self.backbone.block8

        block6_input = self.backbone.block6.get_input_shape()
        self.comb_conv1 = keras.layers.Conv2D(block6_input[-1],padding='same',kernel_size=(2,2))

        self.block6 = self.backbone.block6
        self.block5 = self.backbone.block5

        block3_input = self.backbone.block3.get_input_shape()
        self.comb_conv2 = keras.layers.Conv2D(block3_input[-1],padding='same',kernel_size=(2,2))

        self.block3 = self.backbone.block3
        self.block2 = self.backbone.block2
        self.conv_out = keras.layers.Conv2D(1,padding='same',
            kernel_size=(1,1))
        self.model_input_shape = resnet_input_shape[:-1]
        # self.model_output_shape = self.block2.output_shape[:-1]

    def get_input_shape(self):
        return self.model_input_shape

    def compute_output_shape(self,input_shape):
        return self.get_output_shape()

    def set_layers_from_backbone(self):
        self.block9 = self.backbone.block9
        self.block8 = self.backbone.block8

        self.block6 = self.backbone.block6
        self.block5 = self.backbone.block5

        self.block3 = self.backbone.block3
        self.block2 = self.backbone.block2

    def fix_backbone_weights(self):
        self.backbone.trainable = False

    def unfix_backbone_weights(self):
        self.backbone.trainable = True

    def set_backbone(self,backbone):
        self.backbone = backbone
        self.set_layers_from_backbone()

    def get_output_shape(self):
        return self.conv_out.output_shape

    def call(self,inputs):
        x = self.c1(inputs)
        x = self.block9(x)
        x = self.block8(x)
        x = self.comb_conv1(x)
        x = self.upsample(x)
        x = self.block6(x)
        x = self.block5(x)
        x = self.comb_conv2(x)
        x = self.upsample(x)
        x = self.block3(x)
        x = self.block2(x)
        x = self.conv_out(x)
        return x

class Unet(LayerABC):
    """ Unet layer
    
    For GAN generator layer
    
    Extends:
        LayerABC
    """
    def __init__(self,backbone=None):
        super(Unet, self).__init__(name='Unet')
        #28,28
        self.backbone = backbone
        self.block1 = ResBlock(16,16,32)
        #14,14
        self.block2 = ResBlock(32,32,64,first_stride=2)     # Use bigger_stride, or downsample
        #7,7
        self.block3 = ResBlock(64,64,128,first_stride=2)    # Use bigger_stride, or downsample

        self.block_up_1 = ResBlock(16,16,128)
        self.upsample   = keras.layers.UpSampling2D(size=(2, 2),interpolation='nearest')
        self.block_up_2 = ResBlock(32,32,192) # 128 + 64 final filters required for output
        self.block_up_3 = ResBlock(64,64,224) # 192 + 32
        self.concat     = keras.layers.Concatenate()

    def get_input_shape(self):
        return self.block1.get_input_shape
    
    def fix_backbone_weights(self):
        self.backbone.trainable = False

    def unfix_backbone_weights(self):
        self.backbone.trainable = True

    def get_output_shape(self):
        return self.block_up_3.get_output_shape

    def get_backbone(self):
        return self.backbone

    def set_layers_from_backbone(self):
        pass

    def set_backbone(self,backbone):
        self.backbone = backbone
        self.set_layers_from_backbone()

    def call(self,inputs):
        x_out1 = self.block1(inputs)
        x_out2 = self.block2(x_out1)
        x_out3 = self.block3(x_out2)

        up1  = self.block_up_1(x_out3)

        up2_input  = self.upsample(up1)
        up2_input  = self.concat([x_out2,up2_input])
        # import pdb; pdb.set_trace()  # breakpoint 4e2c710f //

        up2_output = self.block_up_2(up2_input)

        up3_input  = self.upsample(up2_output)
        up3_input  = self.concat([x_out1,up3_input])
        

        up3_output = self.block_up_3(up3_input)
        
        return up3_output


class Classifier(keras.Model):
    """Classifier 
    
    Capable of taking a resnet (or other class) as a backbone
    
    Extends:
        keras.Model
    """
    def __init__(self,backbone,num_classes):
        super(Classifier, self).__init__(name='Classifier')
        #28,28
        self.backbone = backbone
        self.num_classes = num_classes
        self.bn = keras.layers.BatchNormalization()
        self.flatten = keras.layers.Flatten()
        self.init = keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
        self.classifier_layer = keras.layers.Dense(self.num_classes,activation='softmax',kernel_initializer=self.init)
    
    def get_backbone(self):
        return self.backbone

    def set_backbone(self,backbone):
        self.backbone = backbone

    def call(self,inputs):
        x = self.backbone(inputs)
        x = self.flatten(x)
        x = self.bn(x)
        return self.classifier_layer(x)

    def compute_output_shape(self,input_shape):
        return self.num_classes

    def fix_backbone_weights(self):
        self.backbone.trainable = False

    def unfix_backbone_weights(self):
        self.backbone.trainable = True

class Discriminator(LayerABC):
    """Discriminator 
    
    Capable of taking a resnet (or other class) as a backbone
    
    Extends:
        LayerABC
    """
    def __init__(self,backbone):
        super(Discriminator, self).__init__(name='Discriminator')
        #28,28
        self.backbone = backbone
        self.backbone.freeze_batchnorms()
        self.frozen_batchnorms = True

    def build(self,input_shape):
        self.flatten = keras.layers.Flatten()
        self.init = keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
        self.cl = keras.layers.Dense(256,activation='relu',kernel_initializer=self.init)
        self.classifier_layer = keras.layers.Dense(1,activation='sigmoid',kernel_initializer=self.init)
        super(Discriminator, self).build(input_shape)

    def get_backbone(self):
        return self.backbone

    def freeze_batchnorms(self):
        self.backbone.freeze_batchnorms()
        self.frozen_batchnorms = True

    def set_backbone(self,backbone):
        self.backbone = backbone

    def call(self,inputs):
        x = self.backbone(inputs)
        x = self.flatten(x)
        x = self.cl(x)
        return self.classifier_layer(x)

    def compute_output_shape(self,input_shape):
        return (1)

    def fix_backbone_weights(self):
        self.backbone.trainable = False

    def unfix_backbone_weights(self):
        self.backbone.trainable = True
        if self.frozen_batchnorms:
            self.freeze_batchnorms()

class ResNet(LayerABC):
    """RESNET Layer
    
    
    Extends:
        LayerABC
    """
    def __init__(self,):
        super(ResNet, self).__init__(name='ResNet')
        #28,28

    def build(self,input_shape):
        self.block1 = ResBlock(16,16,32,name='block1')
        self.block2 = ResBlock(16,16,32,name='block2')
        self.block3 = ResBlock(16,16,32,name='block3')
        #14,14
        self.block4 = ResBlock(32,32,64,first_stride=2,name='block4')
        self.block5 = ResBlock(32,32,64,name='block5')
        self.block6 = ResBlock(32,32,64,name='block6')
        #7,7
        self.block7 = ResBlock(64,64,128,first_stride=2,name='block7')
        self.block8 = ResBlock(64,64,128,name='block8')
        self.block9 = ResBlock(64,64,128,name='block9')
        self.blocks = [
                self.block1,
                self.block2,
                self.block3,
                #14,14
                self.block4,
                self.block5,
                self.block6,
                #7,7
                self.block7,
                self.block8,
                self.block9,
        ] 
        super(ResNet, self).build(input_shape)

    def compute_output_shape(self,input_shape):
        return self.block9.output_shape

    def freeze_batchnorms(self):
        [block.freeze_batchnorms() for block in self.blocks]

    def call(self, inputs):
        # import pdb; pdb.set_trace()  # breakpoint c5203db7 //
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        return x

class ResBlock(LayerABC):
    """RESBlock Layer
    
    Building blocks of RESNET architecture
    
    Extends:
        LayerABC
    """
    def __init__(self, l1=64,l2=64,l3=256,first_stride=1,name=None):
        super(ResBlock, self).__init__(name='Resblock_{}_{}_{}_{}'.format(l1,l2,l3,name))
        # self.name = 'Resblock_{}_{}_{}_{}'.format(l1,l2,l3,name)
        self.first_stride = first_stride
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.iinput_shape = None
    def build(self, input_shape):
        # self.input_shape = input_shape
        self.conv1 = keras.layers.Conv2D(filters=self.l1,kernel_size=(1,1),strides=(self.first_stride,self.first_stride),
            padding='same',input_shape=input_shape,activity_regularizer=regularizer)
        self.bn1   = keras.layers.BatchNormalization(axis=-1)
        self.conv2 = keras.layers.Conv2D(filters=self.l2,kernel_size=(3,3),strides=(1,1),padding='same')
        self.bn2   = keras.layers.BatchNormalization(axis=-1)
        self.conv3 = keras.layers.Conv2D(filters=self.l3,kernel_size=(1,1),strides=(1,1),padding='same')
        self.pool  = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
        super(ResBlock, self).build(input_shape)
        self.iinput_shape = input_shape
    def get_input_shape(self,):
        try:
            return self.input_shape
        except:
            if(self.iinput_shape):
                return self.iinput_shape
            else:
                print(self.name+' has no defined input shape!')

    def compute_output_shape(self,input_shape):
        return self.conv3.output_shape

    def freeze_batchnorms(self):
        self.bn1.trainable = False
        self.bn2.trainable = False

    def call(self, inputs):
        # import pdb; pdb.set_trace()  # breakpoint fba01bec //
        x = keras.activations.relu(self.bn1(self.conv1(inputs)))
        x = keras.activations.relu(self.bn2(self.conv2(x)))
        if(self.first_stride == 1):
            x = keras.activations.relu(self.conv3(x)+inputs)
        else:
            #@Todo implement resnet skip connection for blocks which change layer. 
            # A dimensions fudge is currently implemented
            # print('Adding pooling')
            x = keras.activations.relu(self.conv3(x)+keras.layers.Concatenate()([self.pool(inputs),self.pool(inputs)]))
        return x


