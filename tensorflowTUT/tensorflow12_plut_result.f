
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
    import numpy as np
    import matplotlib.pyplot as plt

    def add_layer(inputs, in_size, out_size, activation_function=None):
        # add one more layer and return the output of this layer
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

    # Make up some real data
    x_data = np.linspace(-1,1,300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise

    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])
    # add hidden layer
    l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
    # add output layer
    prediction = add_layer(l1, 10, 1, activation_function=None)

    # the error between prediciton and real data
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                         reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # important step
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # plot the real data
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x_data, y_data)
    plt.ion()
    plt.show()

    for i in range(1000):
        # training
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            # to visualize the result and improvement
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            # plot the prediction
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            outport(locals()); ok('22> ');
            plt.pause(0.1)
    </py>

    stop _stop_

玩 mathplotlib，發現 pyplot.show() 會佔住整個 python process
See 帮酷编程问答 https://ask.helplib.com/431148 
討論 matplotlib 的 blocking/non-blocking 堵塞/非堵塞 阻塞/非阻塞 用法。

    <py>
    import matplotlib.pyplot as pyplot
    outport(locals())
    </py>
    pyplot :> plot :: ([1,2,3])  \ 到這裡看不出什麼
    pyplot :: show()             \ 到這裡才顯示出圖形，但 peforth 那邊也停住了。
    
雖然以下這樣就變成 non-blocking 的了，

    pyplot :: show(block=False)  

但是也變成怪怪的，Graphic window 變成 "not responding" 而且停掉它連帶整個 python 
都被停掉。以下可能是最好的辦法：

    <py>
    import matplotlib.pyplot as pyplot
    outport(locals())
    </py>
    pyplot :> plot :: ([1,2,3])  \ 到這裡看不出什麼
    pyplot :: draw() 
    version \ <------------- show() 之前仍在執行
    pyplot :: show() \ <---- show() 之後就 blocked 

    

