import tensorflow as tf

from src.config import *
from src.utils import *


#Encoder/Decoder built for 64x64 image sizes
def decoder(embedding, scope_name, reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        
        h0 = dense_layer(input_layer=embedding, units=8 * 8 * num_filters, scope='dec_h0')
        h0 = tf.reshape(h0, [-1, 8, 8, num_filters])

        conv1 = conv_layer(input_layer=h0, layer_depth=num_filters, scope='dec1')
        conv1 = tf.nn.elu(conv1)
        conv2 = conv_layer(input_layer=conv1, layer_depth=num_filters, scope='dec2')
        conv2 = tf.nn.elu(conv2)
        assert conv2.shape == (batch_size, 8, 8, num_filters)

        upsample1 = upsample(conv=conv2, size=[16,16])
        conv3 = conv_layer(input_layer=upsample1, layer_depth=num_filters, scope='dec3')
        conv3 = tf.nn.elu(conv3)
        conv4 = conv_layer(input_layer=conv3, layer_depth=num_filters, scope='dec4')
        conv4 = tf.nn.elu(conv4)
        assert conv4.shape == (batch_size, 16, 16, num_filters)

        upsample2 = upsample(conv=conv4, size=[32,32])
        conv5 = conv_layer(input_layer=upsample2, layer_depth=num_filters, scope='dec5')
        conv5 = tf.nn.elu(conv5)
        conv6 = conv_layer(input_layer=conv5, layer_depth=num_filters, scope='dec6')
        conv6 = tf.nn.elu(conv6)
        assert conv6.shape == (batch_size, 32, 32, num_filters)


        upsample3 = upsample(conv=conv6, size=[64,64])
        conv7 = conv_layer(input_layer=upsample3, layer_depth=num_filters, scope='dec7')
        conv7 = tf.nn.elu(conv7)
        conv8 = conv_layer(input_layer=conv7, layer_depth=num_filters, scope='dec8')
        conv8 = tf.nn.elu(conv8)
        assert conv8.shape == (batch_size, 64, 64, num_filters)

        decoder_output = conv_layer(input_layer=conv8, layer_depth=3, scope='decoder_image')
        return decoder_output

def encoder(images, scope_name, reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        organized_images = tf.reshape(images, [-1, image_size, image_size, 3])
        
        conv0 = conv_layer(input_layer=organized_images, layer_depth=num_filters, scope='enc0')
        conv0 = tf.nn.elu(conv0)
        conv1 = conv_layer(input_layer=conv0, layer_depth=num_filters, scope='enc1')
        conv1 = tf.nn.elu(conv1)
        conv2 = conv_layer(input_layer=conv1, layer_depth=num_filters, scope='enc2')
        conv2 = tf.nn.elu(conv2)
        assert conv2.shape == (batch_size, 64, 64, num_filters)

        sub1 = subsample(conv=conv2, num_filters=num_filters*2, scope='enc_sub1')
        conv3 = conv_layer(input_layer=sub1, layer_depth=num_filters*2, scope='enc3')
        conv3 = tf.nn.relu(conv3)
        conv4 = conv_layer(input_layer=conv3, layer_depth=num_filters*2, scope='enc4')
        conv4 = tf.nn.elu(conv4)
        assert conv4.shape == (batch_size, 32, 32, num_filters*2)

        sub2 = subsample(conv=conv4, num_filters=num_filters*3, scope='enc_sub2')
        conv5 = conv_layer(input_layer=sub2, layer_depth=num_filters*3, scope='enc5')
        tf.nn.elu(conv5)
        conv6 = conv_layer(input_layer=conv5, layer_depth=num_filters*3, scope='enc6')
        tf.nn.elu(conv6)
        assert conv6.shape == (batch_size, 16, 16, num_filters * 3)

        sub3 = subsample(conv=conv6, num_filters=num_filters*4, scope='enc_sub3')
        conv7 = conv_layer(input_layer=sub3, layer_depth=num_filters*4, scope='enc7')
        tf.nn.elu(conv6)
        conv8 = conv_layer(input_layer=conv7, layer_depth=num_filters*4, scope='enc8')
        tf.nn.elu(conv8)
        assert conv8.shape == (batch_size, 8, 8, num_filters * 4)

        encoder_output = dense_layer(input_layer=conv8, units=hidden_size, scope='encoder_output')
        return encoder_output

def forward_pass_generator(embedding, scope_name='generator', reuse=False):
    print("\nRunning forward pass through generator...\n")
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        print("\nGeneration successful\n")
        return decoder(embedding, scope_name, reuse)

def forward_pass_discriminator(images, scope_name='discriminator', reuse=False):
    print("\nRunning forward pass through discriminator...\n")
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        enc = encoder(images, scope_name, reuse)
        print("\nDiscrimination successful\n")
        return decoder(enc, scope_name, reuse)
