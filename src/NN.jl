export Generator, sampleUQ

# function lrelu(x, th=0.2)
#   return tf.maximum(th * x, x)
# end

# function Generator(x, isTrain=true; vmin=-1, vmax=1)
#   local o
#   variable_scope("generator") do

#       # 1st hidden layer
#       # conv1 = tf.layers.conv2d_transpose(x, 128, [4, 4], strides=(1, 1), padding="valid")
#       lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)
#       # lrelu1 = lrelu(conv1, 0.2)


#       # 2nd hidden layer
#       conv2 = tf.layers.conv2d_transpose(lrelu1, 128, [8, 8], strides=(2, 2), padding="same")
#       lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
#       # lrelu2 = lrelu(conv2, 0.2)

#       # 3rd hidden layer
#       conv3 = tf.layers.conv2d_transpose(lrelu2, 128, [8, 8], strides=(2, 2), padding="same")
#       lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)
#       # lrelu3 = lrelu(conv3, 0.2)

#       # 4th hidden layer
#       conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [8, 8], strides=(2, 2), padding="same")
#       lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)
#       # lrelu4 = lrelu(conv4, 0.2)

#       # output layer
#       conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [8, 8], strides=(2, 2), padding="same")
#       # o = (vmax - vmin)/2. * (tf.nn.tanh(conv5) - (-1)) + vmin
#       o = conv5
#   end
#   return o
# end

function Generator(z, isTrain=true; num_layer=5, h0=4, w0=8, vmin=nothing, vmax=nothing)
  local o
  # activation = tf.keras.activations.relu
  activation = tf.keras.activations.tanh
  # activation = tf.keras.layers.LeakyReLU(alpha=0.1)
  variable_scope("generator") do
    x = tf.keras.layers.Dense(units = h0 * w0 * 8, use_bias=false)(z)
    x = tf.reshape(x, shape=[-1, w0, h0, 8])
    x = activation(x)

    # x = tf.keras.layers.Conv2DTranspose(64, (5,5), strides=(2, 2), padding="same", use_bias=false)(x)
    # x = tf.pad(x, ((0,0), (1,2), (1,2), (0,0)), "REFLECT")
    # x = tf.keras.layers.Conv2D(16, [4, 4], strides=(1, 1), padding="valid")(x)

    # x = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(x)
    # x = tf.keras.layers.Conv2D(16, [4, 4], strides=(1, 1), padding="same", use_bias=false)(x)
    # x = activation(x)

    # x = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(x)
    # x = tf.keras.layers.Conv2D(32, [4, 4], strides=(1, 1), padding="same", use_bias=false)(x)
    # x = activation(x)

    # x = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(x)
    # x = tf.keras.layers.Conv2D(64, [4, 4], strides=(1, 1), padding="same", use_bias=false)(x)
    # x = activation(x)

    # x = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(x)
    # x = tf.keras.layers.Conv2D(32, [4, 4], strides=(1, 1), padding="same", use_bias=false)(x)
    # x = activation(x)

    # x = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(x)
    # x = tf.keras.layers.Conv2D(16, [4, 4], strides=(1, 1), padding="same", use_bias=false)(x)
    # x = activation(x)

    for i = 1:num_layer-1
      x = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(x)
      x = tf.keras.layers.Conv2D(32, [4, 4], strides=(1, 1), padding="same", use_bias=false)(x)
      x = activation(x)
    end

    x = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(x)
    x = tf.keras.layers.Conv2D(1, [4, 4], strides=(1, 1), padding="same", use_bias=false)(x)

    # x = tf.keras.layers.Conv2D(1, [1, 1], strides=(1, 1), padding="same")(x)

    if !isnothing(vmax) && !isnothing(vmin)
        # o = (vmax - vmin) * tf.keras.activations.sigmoid(x) + vmin
        o = ((vmax - vmin) * tf.keras.activations.tanh(x) + (vmax + vmin))/2.0
        @info "vmin=$vmin, vmax=$vmax"
        # error()
    else
        # o = activation(x)
        o = x
        # error()
    end
    # error()
    o = tf.squeeze(o)
    # o = cast(o, Float64)
  end
  return o
end

function Generator(z, isTrain, dropout_rate; num_layer=5, h0=4, w0=8, vmin=nothing, vmax=nothing)
  local o
  # activation = tf.keras.activations.relu
  activation = tf.keras.activations.tanh
  # activation = tf.keras.layers.LeakyReLU(alpha=0.1)
  variable_scope("generator") do
    x = tf.keras.layers.Dense(units = h0 * w0 * 8, use_bias=false)(z)
    x = tf.reshape(x, shape=[-1, w0, h0, 8])
    x = activation(x)
    # x = tf.keras.layers.Dropout(dropout_rate)(x, isTrain)
    # x = tf.keras.layers.SpatialDropout2D(dropout_rate)(x, isTrain)

    # x = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(x)
    # x = tf.keras.layers.Conv2D(32, [4, 4], strides=(1, 1), padding="same")(x)
    # x = activation(x)
    # x = tf.keras.layers.Dropout(dropout_rate)(x, isTrain)
    # x = tf.keras.layers.SpatialDropout2D(dropout_rate)(x, isTrain)

    # x = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(x)
    # x = tf.keras.layers.Conv2D(64, [4, 4], strides=(1, 1), padding="same")(x)
    # x = activation(x)
    # x = tf.keras.layers.Dropout(dropout_rate)(x, isTrain)
    # x = tf.keras.layers.SpatialDropout2D(dropout_rate)(x, isTrain)
# 
    # x = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(x)
    # x = tf.keras.layers.Conv2D(32, [4, 4], strides=(1, 1), padding="same")(x)
    # x = activation(x)
    # x = tf.keras.layers.Dropout(dropout_rate)(x, isTrain)
    # x = tf.keras.layers.SpatialDropout2D(dropout_rate)(x, isTrain)
  
    # x = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(x)
    # x = tf.keras.layers.Conv2D(16, [4, 4], strides=(1, 1), padding="same", use_bias=false)(x)
    # x = tf.keras.layers.Dropout(dropout_rate)(x, isTrain)
    # x = activation(x)

    for i = 1:num_layer-1
      x = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(x)
      x = tf.keras.layers.Conv2D(32, [4, 4], strides=(1, 1), padding="same", use_bias=false)(x)
      x = tf.keras.layers.Dropout(dropout_rate)(x, isTrain)
      x = activation(x)
    end

    x = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(x)
    x = tf.keras.layers.Conv2D(1, [4, 4], strides=(1, 1), padding="same", use_bias=false)(x)

    # x = tf.keras.layers.Conv2D(1, [1, 1], strides=(1, 1), padding="same")(x)

    if !isnothing(vmax) && !isnothing(vmin)
        # o = (vmax - vmin) * (tf.keras.activations.tanh(x) - (-1))/2 + vmin
        o = ((vmax - vmin) * tf.keras.activations.tanh(x) + (vmax + vmin))/2.0
    else
        # o = activation(x)
        o = x
    end
    o = tf.squeeze(o)
    # o = cast(o, Float64)
  end
  return o
end

function sampleUQ(batch_size, sample_size, rcv_size; z_size=8, base=4, ratio=1, vmin=nothing, vmax=nothing)
  isTrain = placeholder(Bool)
  y = placeholder(Float64, shape=(sample_size, rcv_size...))
  z = placeholder(Float32, shape=(batch_size, z_size))
  G_z = cast(Float64, Generator(z, isTrain, base=base, ratio=ratio, vmin=vmin, vmax=vmax))
  return G_z, isTrain, y, z
end
