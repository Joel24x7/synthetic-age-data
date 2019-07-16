import tensorflow as tf
from wrapped_utils import *
from hyperparameters import *

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

        upsample1 = upsample(conv=conv2, size=[16,16])
        conv3 = conv_layer(input_layer=upsample1, layer_depth=num_filters, scope='dec3')
        conv3 = tf.nn.elu(conv3)
        conv4 = conv_layer(input_layer=conv3, layer_depth=num_filters, scope='dec4')
        conv4 = tf.nn.elu(conv4)

        upsample2 = upsample(conv=conv4, size=[32,32])
        conv5 = conv_layer(input_layer=upsample2, layer_depth=num_filters, scope='dec5')
        conv5 = tf.nn.elu(conv5)
        conv6 = conv_layer(input_layer=conv5, layer_depth=num_filters, scope='dec6')
        conv6 = tf.nn.elu(conv6)

        # 64x64
        upsample3 = upsample(conv=conv6, size=[64,64])
        conv7 = conv_layer(input_layer=upsample3, layer_depth=num_filters, scope='dec7')
        conv7 = tf.nn.elu(conv7)
        conv8 = conv_layer(input_layer=conv7, layer_depth=num_filters, scope='dec8')
        conv8 = tf.nn.elu(conv8)

        #128x128
        # upsample4 = upsample(conv=conv8, size=[128,128])
        # conv9 = conv_layer(input_layer=upsample4, layer_depth=num_filters, scope='dec9')
        # conv9 = tf.nn.elu(conv9)
        # conv10 = conv_layer(input_layer=conv9, layer_depth=num_filters, scope='dec10')
        # conv10 = tf.nn.elu(conv10)

        decoder_output = conv_layer(input_layer=conv8, layer_depth=3, scope='decoder_image')
        #decoder_output = tf.sigmoid(decoder_output)
        #decoder_output = tf.reshape(decoder_output, [-1, image_size * image_size * 3])
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

        sub1 = subsample(conv=conv2, num_filters=num_filters*2, scope='enc_sub1')
        conv3 = conv_layer(input_layer=sub1, layer_depth=num_filters*2, scope='enc3')
        conv3 = tf.nn.relu(conv3)
        conv4 = conv_layer(input_layer=conv3, layer_depth=num_filters*2, scope='enc4')
        conv4 = tf.nn.elu(conv4)

        sub2 = subsample(conv=conv4, num_filters=num_filters*3, scope='enc_sub2')
        conv5 = conv_layer(input_layer=sub2, layer_depth=num_filters*3, scope='enc5')
        tf.nn.elu(conv5)
        conv6 = conv_layer(input_layer=conv5, layer_depth=num_filters*3, scope='enc6')
        tf.nn.elu(conv6)

        sub3 = subsample(conv=conv6, num_filters=num_filters*4, scope='enc_sub3')
        conv7 = conv_layer(input_layer=sub3, layer_depth=num_filters*4, scope='enc7')
        tf.nn.elu(conv6)
        conv8 = conv_layer(input_layer=conv7, layer_depth=num_filters*4, scope='enc8')
        tf.nn.elu(conv8)

        # sub4 = subsample(conv=conv8, num_filters=num_filters*5, scope='enc_sub4')
        # conv9 = conv_layer(input_layer=sub4, layer_depth=num_filters*5, scope='9')
        # tf.nn.elu(conv9)
        # conv10 = conv_layer(input_layer=conv9, layer_depth=num_filters*5, scope='enc10')
        # tf.nn.elu(conv10)

        encoder_output = dense_layer(input_layer=conv8, units=hidden_size, scope='encoder_output')
        return encoder_output

def forward_pass_generator(embedding, scope_name='generator', reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        return decoder(embedding, scope_name, reuse)

def forward_pass_discriminator(images, scope_name='discriminator', reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        enc = encoder(images, scope_name, reuse)
        return decoder(enc, scope_name, reuse)