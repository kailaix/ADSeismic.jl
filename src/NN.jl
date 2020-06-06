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

function Generator(z, isTrain=true; base=4, ratio=1, vmin=nothing, vmax=nothing)
  local o
  variable_scope("generator") do
    x = tf.keras.layers.Dense(units = Int(round(base * ratio)) * base * 16)(z)
    x = tf.reshape(x, shape=[-1, Int(round(base * ratio)), base, 16])
    x = tf.keras.activations.tanh(x)

    x = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(x)
    x = tf.keras.layers.Conv2D(32, [4, 4], strides=(1, 1), padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(x)
    x = tf.keras.layers.Conv2D(32, [4, 4], strides=(1, 1), padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(x)
    x = tf.keras.layers.Conv2D(32, [4, 4], strides=(1, 1), padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(x)
    x = tf.keras.layers.Conv2D(1, [4, 4], strides=(1, 1), padding="same")(x)
    o = x

    if !isnothing(vmax) && !isnothing(vmin)
        o = (vmax - vmin) * (tf.keras.activations.tanh(x) - (-1))/2 + vmin
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
