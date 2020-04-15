
def mbconv(inputs, expansion, stride, filters, alpha=1, block_id=0):

    in_nfilters = backend.int_shape(inputs)[-1]
    out_nfilters = int(alpha*filters)

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    # Expand
    x = layers.Conv2D(int(in_nfilters*expansion),
                     kernel_size=(1, 1),
                     padding='same',
                     use_bias=False,
                     activation=None,
                     name='{}_expand'.format(block_id))(inputs)
    # x = layers.BatchNormalization(axis=channel_axis,
    #                              name='{}_expand_bn'.format(block_id))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Activation('relu',
                         name='{}_expand_act'.format(block_id))(x)

    # Depthwise
    if stride > 1:
        depth_padding = 'valid'
        x = layers.ZeroPadding2D(padding=correct_pad(x, 3))(x)
    else:
        depth_padding = 'same'
    
    x = layers.DepthwiseConv2D((3, 3),
                              strides=stride,
                              padding=depth_padding,
                              activation=None,
                              use_bias=False,
                              name='{}_depth'.format(block_id))(x)
    # x = layers.BatchNormalization(axis=channel_axis,
    #                              name='{}_depth_bn'.format(block_id))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Activation('relu',
                         name='{}_depth_act'.format(block_id))(x)

    # Project
    x = layers.Conv2D(out_nfilters,
                     kernel_size=(1, 1),
                     padding='same',
                     use_bias=False,
                     activation=None,
                    name='{}_project'.format(block_id))(x)
    # This layers does not have an activation function, it is linear
    # x = layers.BatchNormalization(axis=channel_axis,
    #                              name='{}_project_bn'.format(block_id))(x)
    x = layers.Dropout(0.25)(x)
    if in_nfilters == out_nfilters and stride == 1:
        # Sum the output layers with the input layer
        return layers.Add(name='{}_add'.format(block_id))([inputs, x])
    return x

