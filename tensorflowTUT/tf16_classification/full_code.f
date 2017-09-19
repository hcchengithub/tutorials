
    --- marker ---

    <py>
    # View more python learning tutorial on my Youtube and Youku channel!!!

    # Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
    # Youku video tutorial: http://i.youku.com/pythontutorial

    """
    Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
    """
    # from __future__ import print_function
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # https://stackoverflow.com/questions/43134753/tensorflow-wasnt-compiled-to-use-sse-etc-instructions-but-these-are-availab
    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data
    # number 1 to 10 data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    def add_layer(inputs, in_size, out_size, activation_function=None,):
        # add one more layer and return the output of this layer
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b,)
        return outputs

    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
    ys = tf.placeholder(tf.float32, [None, 10])

    # add output layer
    prediction = add_layer(xs, 784, 10,  activation_function=tf.nn.softmax)
    
    def compute_accuracy(v_xs, v_ys):
        # global prediction
        y_pre = sess.run(prediction, feed_dict={xs: v_xs})
        correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
        return result

    # the error between prediction and real data
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))       # loss
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.Session()
    # important step
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
        if i % 50 == 0:
            print(compute_accuracy(
                mnist.test.images, mnist.test.labels))

    </py>
    
    
    stop 

    [ ] 因為會抓大量資料 working directory 用 downloads
    
    [ ] 執行結果如下，有問題！效果陡降，卡死。
    
        OK include c:\Users\hcche\Documents\GitHub\Morvan\tutorials\tensorflowTUT\tf16_classification\full_code.f
        Extracting MNIST_data\train-images-idx3-ubyte.gz
        Extracting MNIST_data\train-labels-idx1-ubyte.gz
        Extracting MNIST_data\t10k-images-idx3-ubyte.gz
        Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
        0.1309
        0.6571
        0.7463
        0.7831
        0.8036
        0.8233
        0.8327
        0.8343
        0.8427
        0.8479
        0.8542
        0.8561
        0.098
        0.098
        0.098
        0.098
        0.098
        0.098
        0.098
        0.098
        OK    