
    \ include c:\Users\hcche\Documents\GitHub\Morvan\tutorials\tensorflowTUT\tensorflow11_build_network.f 

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
    x_data = np.linspace(-1,1,300)[:, np.newaxis]   # ^22  不是 random 是均勻標上刻度
    noise = np.random.normal(0, 0.05, x_data.shape) # ^33 
    y_data = np.square(x_data) - 0.5 + noise        # ^44

    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])
    # add hidden layer
       # add_layer(inputs, in_size, out_size, activation_function=None)
    l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)   # ^55
    # add output layer
    prediction = add_layer(l1, 10, 1, activation_function=None) # ^66

    # the error between prediction and real data                ^77
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                         reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)   # ^88

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    outport(locals());ok('^99> ')

    for i in range(1000):
        # training
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            # to see the step improvement
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
    </py>
    stop _stop_

^22
    查 np.linspace 的說明：
    
        np :> linspace py: help(pop())
        Help on function linspace in module numpy.core.function_base:
        linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
            Return evenly spaced numbers over a specified interval.
            Returns `num` evenly spaced samples, calculated over the
            interval [`start`, `stop`].
            The endpoint of the interval can optionally be excluded.
        
    不懂為何 None 就 None 還要起個名字叫 np.newaxis? 確實就是 None.
    
        np :> newaxis tib. \ ==> None (<class 'NoneType'>)

    numpy.ndarray 有神奇的操作方式，普通 list 沒有：
        
        py> [11,22,33][:,None] .
        Failed in </py> (compiling=False): list indices must be integers or slices, not tuple

    看起來它的意思是要把一個 list 放進一個 list 裡面去。numpy.ndarray 真的可以：
    
        np :> array([11,22,33])[:,None] .
        [[11]
         [22]
         [33]]

    如果少了那個 ,None 或單 [:] 就是取出本來的 array 而已。這好像是 R 的語法。
    加了 None 充分利用它來表達這個意思：「在橫向多放一個維度進去」。
    
        np :> array([11,22,33])[:,] . cr
        [11 22 33]
        
        np :> array([11,22,33])[:] . cr
        [11 22 33]
        
    
    我學 R 時摸索過一陣子，筆記 ___________.
    嘗試不放 ,None 改放 1 結果是個錯誤：
    
        np :> array([11,22,33])[:,1] . cr
        Failed in </py> (compiling=False): too many indices for array
        
    當然 np.array() 也可以用來 convert tuple 
        
        np :> array((11,22,33))[:,None] . cr
        [[11]
         [22]
         [33]]
         
    但是 set 不行：
    
        np :> array({11,22,33})[:,None] . cr
        Failed in </py> (compiling=False): too many indices for array

    我大概知道為何 x_data 要多放進一個維度，可能是為了要 access by reference
    instead of access by value. By value 可能既浪費空間又不適用。昨天寫 selftest
    為了取得 screenbuffer 就是得定義成
        py> [""] value screen-buffer // ( -- ['string'] ) Selftest screen buffer
    而非
        py> "" value screen-buffer // ( -- 'string' ) Selftest screen buffer
    否則操作中會 access 不到這個特定的 string. 
    
^33

    ^99> noise :> [:10] . cr
    [[-0.01054578]
     [-0.01861341]   # random numbers
     ...snip...
     [ 0.00235038]]
    
    ^99> x_data :> shape . cr
    (300, 1)

    ^99> np :> random.normal(0,0.05,(10,2)) . cr
    [[ 0.02850939 -0.00595862]
     [-0.06682586 -0.03733732]
     [-0.0630333  -0.03053933]
     [ 0.02102613 -0.02396285]
     [ 0.0291464  -0.08322714]
     [-0.04734182  0.00385998]
     [ 0.01377979  0.06226352]
     [ 0.04962091 -0.04245547]
     [-0.05492529  0.05496064]
     [ 0.04798303 -0.11472538]]
    ^99>

^44

    np :> square((2,4,6)) tib. \ ==> [ 4 16 36] (<class 'numpy.ndarray'>)
    x_data :> shape tib. \ ==> (300, 1) (<class 'tuple'>)
    y_data :> shape tib. \ ==> (300, 1) (<class 'tuple'>)

^55
    ^99> words
    sess init train_step loss prediction l1 
    ys xs y_data noise x_data add_layer np tf

    l1 tib. \ ==> Tensor("Relu:0", shape=(?, 10), dtype=float32) 
                    (<class 'tensorflow.python.framework.ops.Tensor'>)
    
^66
    ^99> words
    sess init train_step loss prediction l1 
    ys xs y_data noise x_data add_layer np tf
^77
    ^99> words
    sess init train_step loss prediction l1 
    ys xs y_data noise x_data add_layer np tf
    
^88
    ^99> words
    sess init train_step loss prediction l1 
    ys xs y_data noise x_data add_layer np tf

    --
    OK tf :> reduce_mean py: help(pop())
    Help on function reduce_mean in module tensorflow.python.ops.math_ops:

    reduce_mean(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)
        Computes the mean of elements across dimensions of a tensor.

        Reduces `input_tensor` along the dimensions given in `axis`.
        Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
        entry in `axis`. If `keep_dims` is true, the reduced dimensions
        are retained with length 1.

        If `axis` has no entries, all dimensions are reduced, and a
        tensor with a single element is returned.

        For example:

        ```python
        # 'x' is [[1., 1.]
        #         [2., 2.]]
        tf.reduce_mean(x) ==> 1.5
        tf.reduce_mean(x, 0) ==> [1.5, 1.5]
        tf.reduce_mean(x, 1) ==> [1.,  2.]    
    --
    
^99
    ^99> words
    sess init train_step loss prediction l1 
    ys xs y_data noise x_data add_layer np tf


