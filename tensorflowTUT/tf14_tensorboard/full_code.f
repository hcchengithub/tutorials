
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

    def add_layer(inputs, in_size, out_size, activation_function=None):
        # add one more layer and return the output of this layer
        with tf.name_scope('layer'):
            with tf.name_scope('weights'):
                Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            with tf.name_scope('biases'):
                biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            with tf.name_scope('Wx_plus_b'):
                Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
            if activation_function is None:
                outputs = Wx_plus_b
            else:
                outputs = activation_function(Wx_plus_b, )
            return outputs


    # define placeholder for inputs to network
    with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
        ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

    # add hidden layer
    l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
    # add output layer
    prediction = add_layer(l1, 10, 1, activation_function=None)

    # the error between prediciton and real data
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                            reduction_indices=[1]))

    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    sess = tf.Session()

    # tf.train.SummaryWriter soon be deprecated, use following
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
        writer = tf.train.SummaryWriter('logs/', sess.graph)
    else: # tensorflow version >= 0.12
        writer = tf.summary.FileWriter("logs/", sess.graph)

    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)

    # direct to the local dir and run this in terminal:
    # $ tensorboard --logdir=logs
    </py>
    
    stop 

See also Ynote: "Morvan Python TensorFlow Graph Visualization"
    
執行後，會在工作 directory 下產生 'logs' folder. 可以把它整個搬到你喜歡的地方去使用。
例如：c:\Users\hcche\Documents\GitHub\Morvan\tutorials\tensorflowTUT\tf14_tensorboard\logs 
照本例，在 c:\Users\hcche\Documents\GitHub\Morvan\tutorials\tensorflowTUT\tf14_tensorboard\
打開 DOS Box 執行 tensorboard --logdir=logs 真的跑得起來(要等一會兒)，如下
    
    Microsoft Windows [Version 10.0.15063]
    (c) 2017 Microsoft Corporation. All rights reserved.

    C:\Users\hcche\Documents\GitHub\Morvan\tutorials\tensorflowTUT\tf14_tensorboard>tensorboard --logdir=logs
    Starting TensorBoard b'47' at http://0.0.0.0:6006  <---- http://localhost:6006 for Windows 10
    (Press CTRL+C to quit)
    WARNING:tensorflow:path ../external\data/plugin/text/runs not found, sending 404
    WARNING:tensorflow:path ../external\data/plugin/text/runs not found, sending 404
    WARNING:tensorflow:path ../external\data/plugin/text/runs not found, sending 404
    WARNING:tensorflow:path ../external\data/plugin/text/runs not found, sending 404

Windows 10 的 localhost 不是 0.0.0.0 而是 127.0.0.1 
    
    

