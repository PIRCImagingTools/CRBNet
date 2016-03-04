if __name__ == '__main__':

    import network as nn
    import theano.tensor as T
    import time

 #   def ReLU(z): return T.maximum(0.0, z)

    mini_bach_size = 10

    print "started at: " + time.asctime()
    t1_asc = time.asctime()
    t1 = time.time()

## Layer syntax:
## image_shape = (mini batch size,
##                number of input dimensions/feature maps,
##               Image height, Image width)

## Filter_shape= (number of filters i.e. how many new feature maps,
##                input feature maps: number of input dimensions/feature maps,
##                filter H,W: h,w of local receptive field)

    training_data, validation_data, test_data = nn.load_data_shared()
    net = nn.Network([
                nn.ConvPoolLayer(image_shape=(mini_bach_size, 1, 28, 28),
                                 filter_shape=(20, 1, 5, 5),
                                 poolsize=(2,2),
                                 activation_fn="ReLU"),
                nn.ConvPoolLayer(image_shape=(mini_bach_size, 20, 12, 12),
                                 filter_shape=(40, 20, 5, 5),
                                 poolsize=(2,2),
                                 activation_fn="ReLU"),
                nn.FullyConnectedLayer(n_in=40*4*4, n_out=1000,
                                       activation_fn="ReLU", p_dropout=0.5),
                nn.FullyConnectedLayer(n_in=1000, n_out=1000,
                                       activation_fn="ReLU", p_dropout=0.5),
                nn.SoftmaxLayer(n_in=1000, n_out=10,p_dropout=0.5)], mini_bach_size)
    net.SGD(training_data, 40, mini_bach_size, 0.03,
            validation_data, test_data)


    print "started at: " + t1_asc
    print "finished at: " + time.asctime()

    TIME=time.time() - t1
    HOURS = TIME/3600
    MINUTES = (TIME%3600)/60
    SECONDS = (TIME%3600)%60

    print "duration:{0:3.0f} hours, {1:3.0f} minutes, {2:3.0f} seconds".format(
                                                          HOURS,MINUTES,SECONDS)


