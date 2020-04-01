import keras
from keras.regularizers import l2

regularizer = l2(0.0001)

class Classifier(keras.Model):
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


class Disciminator(keras.Model):
    def __init__(self,backbone):
        super(Disciminator, self).__init__(name='Disciminator')
        #28,28
        self.backbone = backbone
        self.flatten = keras.layers.Flatten()
        self.init = keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
        self.classifier_layer = keras.layers.Dense(1,activation='sigmoid',kernel_initializer=self.init)
    
    def get_backbone(self):
        return self.backbone

    def set_backbone(self,backbone):
        self.backbone = backbone

    def call(self,inputs):
        x = self.backbone(inputs)
        x = self.flatten(x)
        return self.classifier_layer(x)

    def compute_output_shape(self,input_shape):
        return (1)


class ResNet(keras.Model):
    def __init__(self,):
        super(ResNet, self).__init__(name='ResNet')
        #28,28
        self.block1 = ResBlock(12,12,24)
        self.block2 = ResBlock(12,12,24)
        self.block3 = ResBlock(12,12,24)
        #14,14
        self.block4 = ResBlock(24,24,48,first_stride=2)
        self.block5 = ResBlock(24,24,48)
        self.block6 = ResBlock(24,24,48)
        #7,7
        self.block7 = ResBlock(48,48,96,first_stride=2)
        self.block8 = ResBlock(48,48,96)
        self.block9 = ResBlock(48,48,96)

        # self.layers = [
        #                 self.block1,
        #                 self.block2,
        #                 self.block3,
        #                 self.block4,
        #                 self.block5,
        #                 self.block6,
        #                 self.block7,
        #                 self.block8,
        #                 self.block9,
        #             ]

    def compute_output_shape(self,input_shape):
        return self.block9.output_shape

    def call(self, inputs):
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

class ResBlock(keras.Model):
    def __init__(self, l1=64,l2=64,l3=256,first_stride=1):
        super(ResBlock, self).__init__(name='ResBlock')
        self.first_stride = first_stride
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

    def build(self, input_shape):
        self.conv1 = keras.layers.Conv2D(filters=self.l1,kernel_size=(1,1),strides=(self.first_stride,self.first_stride),
            padding='same',input_shape=input_shape,activity_regularizer=regularizer)
        self.bn1   = keras.layers.BatchNormalization(axis=-1)
        self.conv2 = keras.layers.Conv2D(filters=self.l2,kernel_size=(3,3),strides=(1,1),padding='same',activity_regularizer=regularizer)
        self.bn2   = keras.layers.BatchNormalization(axis=-1)
        self.conv3 = keras.layers.Conv2D(filters=self.l3,kernel_size=(1,1),strides=(1,1),padding='same',activity_regularizer=regularizer)
        self.pool  = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)

    def compute_output_shape(self,input_shape):
        return self.conv3.output_shape

    def call(self, inputs):
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


