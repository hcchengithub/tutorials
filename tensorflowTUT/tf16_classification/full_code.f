
    \ include c:\Users\hcche\Documents\GitHub\Morvan\tutorials\tensorflowTUT\tf16_classification\full_code.f 

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
    ys = tf.placeholder(tf.float32, [None, 10])  # 0~9 digits 

    # add output layer
    prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)
    
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

    # outport(locals()); ok('11> '); # Breakpoint 
    
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
        outport(locals()); ok('22> '); # Breakpoint 
        if i % 50 == 0:
            print(compute_accuracy(
                mnist.test.images, mnist.test.labels))

    </py>
    
    
    stop 

    [x] 因為會抓大量資料 working directory 用 downloads
    
    [ ] 執行結果如下，有問題！效果陡降，卡死。 <-- 時好時壞，怎麼回事？
    
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

    [x] 程式沒改，隔天再跑，又好了！
        ... snip ...
        0.868
        0.8703
        0.876
        0.8763
        0.8768
        OK

        ... snip ...
        0.872
        0.8695
        0.8742
        0.8743
        0.8768
        0.8833
        OK        
        
    11> words ==>
        --- init train_step cross_entropy compute_accuracy 
        add_layer mnist input_data ys xs tf sess prediction

    11> mnist . cr
    Datasets(
        train =      <tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x0000023DC01DCC88>, 
        validation = <tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x0000023DB980C198>, 
        test =       <tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x0000023DB980C390>
    )

    [x] mnist py: help(pop()) 果然有 help。除了 data 它還有個 next_match() method.
    
        Help on DataSet in module tensorflow.contrib.learn.python.learn.datasets.mnist 
        object:

        class DataSet(builtins.object)
         |  Methods defined here:
         |  next_batch(self, batch_size, fake_data=False, shuffle=True)
         |      Return the next `batch_size` examples from this data set.
         |
         ... snip ...

         
        mnist :> train type tib. \ ==> <class 'tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet'> (<class 'type'>)
        mnist :> train dir tib. \ ==> [..., 'images', 'labels', 'next_batch', 'num_examples'] (<class 'list'>)
        mnist :> train.images type tib. \ ==> <class 'numpy.ndarray'> (<class 'type'>)
        mnist :>      train.images.shape tib. \ ==> (55000, 784) (<class 'tuple'>)
        mnist :>       test.images.shape tib. \ ==> (10000, 784) (<class 'tuple'>)
        mnist :> validation.images.shape tib. \ ==> (5000, 784) (<class 'tuple'>)
        mnist :>      train.labels.shape tib. \ ==> (55000, 10) (<class 'tuple'>)
        mnist :>       test.labels.shape tib. \ ==> (10000, 10) (<class 'tuple'>)
        mnist :> validation.labels.shape tib. \ ==> (5000, 10) (<class 'tuple'>)

    [x] 想看看 mnist 三個 DataSet 裡面的圖片，回頭查出是用 PIL module 處理圖像
        YNote: "用來把 bitmap array 存成 jpg 圖畫檔的 python function"
        <accept>
        <py>
            import PIL.Image as image
            row, col = 300, 200
            pic_new = image.new("RGB", (row, col))  # "L" ; "RGB" m\
            for i in range(row):
                for j in range(col):
                    pic_new.putpixel((i,j),(223,250,223))
            pic_new.save("test.jpg", "JPEG")
            outport(locals())
        </py> </accept>
        11> tib.insert
        11> words ==> j i pic_new col row image
        row tib. \ ==> 300 (<class 'int'>)
        col tib. \ ==> 200 (<class 'int'>)
        pic_new type tib. \ ==> <class 'PIL.Image.Image'> (<class 'type'>)
        11> pic_new py: help(pop())  真好！ 以下我擷取了部分有用的 methods 
        
        Help on Image in module PIL.Image object:
        class Image(builtins.object)
         |  close(self)
         |      Closes the file pointer, if possible.
         |
         |      This operation will destroy the image core and release its memory.
         |      The image data will be unusable afterward.
         |
         |  putdata(self, data, scale=1.0, offset=0.0)
         |      Copies pixel data to this image.  This method copies data from a
         |      sequence object into the image, starting at the upper left
         |      corner (0, 0), and continuing until either the image or the
         |      sequence ends.  The scale and offset values are used to adjust
         |      the sequence values: **pixel = value*scale + offset**.
         |
         |  show(self, title=None, command=None)
         |      Displays this image. This method is mainly intended for
         |      debugging purposes.
         |
         | save(self, fp, format=None, **params)
         |      Saves this image under the given filename.  If no format is
         |      specified, the format to use is determined from the filename
         |      extension, if possible.
         |
         |  thumbnail(self, size, resample=3)
         |      Make this image into a thumbnail.  This method modifies the
         |      image to contain a thumbnail version of itself, no larger than
         |      the given size.  This method calculates an appropriate thumbnail
         |      size to preserve the aspect of the image, calls the
         |      :py:meth:`~PIL.Image.Image.draft` method to configure the file reader
         |      (where applicable), and finally resizes the image.

        11> words ==> j i pic_new col row image
        11> pic_new :: show()
        11> pic_new :: show(title="hehehe")
        11> pic_new :: show()
        11> pic_new :: thumbnail()

        Failed in </py> (compiling=False): thumbnail() missing 1 required positional argument: 'size'
        Body:
        pop().thumbnail()
        11> pic_new :: thumbnail((50,50))
        11> pic_new :: show()
        11>    
    [x] 弄一張圖出來 mnist :> train.images[0]
            mnist :> train.images[0] py> tuple(pop()) 
            pic :: putdata(pop())
        結果不行 : new style getargs format but argument is not a tuple    
        我猜是 長─寬 不對，手寫圖片是 28x28 所以要重新取得 pic object 
            py:~ import PIL.Image as image; push(image)
            value image 
            image :> new("L",(28,28)) value pic // ( -- PIL.image ) object
        --> 重來。。成功了！
        --> pic :: show() 很黑
        --> mnist :> train.images[0] py> tuple(pop()*256) \ 調整 scale 
            pic :: putdata(pop()) \ 真的成功了！！！
        --> mnist :> train.images[0] py> tuple(pop()*256) pic :: putdata(pop()) pic :: show()
            mnist :> train.labels[0] tib.
            mnist :> train.images[1] py> tuple(pop()*256) pic :: putdata(pop()) pic :: show()
            mnist :> train.labels[1] tib.
            mnist :> train.images[2] py> tuple(pop()*256) pic :: putdata(pop()) pic :: show()
            mnist :> train.labels[2] tib.
            mnist :> train.images[3] py> tuple(pop()*256) pic :: putdata(pop()) pic :: show()
            mnist :> train.labels[3] tib.
            mnist :> test.images[3] py> tuple(pop()*256) pic :: putdata(pop()) pic :: show()
            mnist :> test.labels[3] tib.
            mnist :> validation.images[3] py> tuple(pop()*256) pic :: putdata(pop()) pic :: show()
            mnist :> validation.labels[3] tib.
            
    [ ] 徹底看懂了資料，回頭 trace     
        batch_xs type tib. \ ==> <class 'numpy.ndarray'> (<class 'type'>)
        batch_xs :> shape tib. \ ==> (100, 784) (<class 'tuple'>)
        batch_ys :> shape tib. \ ==> (100, 10) (<class 'tuple'>)
    
        得知這行就是取樣而已
        batch_xs, batch_ys = mnist.train.next_batch(100)