
    \ T550
    \ include c:\Users\hcche\Documents\GitHub\Morvan\tutorials\tensorflowTUT\tf16_classification\full_code.f 
    \ X1 Yoga, Lrv2 
    \ include c:\Users\hcche\Documents\GitHub\ML\tutorials\tensorflowTUT\tf16_classification\full_code.f 
    
    \ Check working directory. It must be 'downloads' to avoid download datasets
    \ to an unexpected location or a duplicated download
    py> os.getcwd() :> find("ownloads")==-1 
    [if] cr ." Working directory must be ~/downloads" cr abort [then]
    
    --- marker ---

    <py>

    # English: https://www.youtube.com/watch?v=AhC6r4cwtq0&index=16&list=PLXO45tsB95cJHXaDKpbwr5fC_CCYylw1f

    # View more python learning tutorial on my Youtube and Youku channel!!!
    # Youtube video channel: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
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
        if debug==11: ok('11> ', loc=locals(), cmd="--- marker --- :> [0] inport") # Breakpoint 
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
        if debug==33: ok('33> ', loc=locals(), cmd="--- marker --- :> [0] inport") # Breakpoint 
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
        if debug==22: ok('22> ', loc=locals(), cmd="--- marker --- :> [0] inport") # Breakpoint 
        if i % 50 == 0:
            print(compute_accuracy(
                mnist.test.images, mnist.test.labels))
    </py>
    
    
    stop 

    [x] 因為會抓大量資料 working directory 用 downloads
    
    [/] 執行結果如下，有問題！效果陡降，卡死。 <-- 時好時壞，怎麼回事？
    
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
        --- batch_ys batch_xs i init train_step cross_entropy 
        compute_accuracy add_layer mnist input_data ys xs tf sess prediction

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
            
    [x] 徹底看懂了資料，回頭 trace     
        batch_xs type tib. \ ==> <class 'numpy.ndarray'> (<class 'type'>)
        batch_xs :> shape tib. \ ==> (100, 784) (<class 'tuple'>)
        batch_ys :> shape tib. \ ==> (100, 10) (<class 'tuple'>)
    
        得知這行就是取樣而已
        batch_xs, batch_ys = mnist.train.next_batch(100)
        
    [x] 看他的 loss (cross_entropy) 怎麼定義的
        reduce_mean 就是取 array 的平均值。 Reduce 是指工作在 matrix 的某維度上。
            tf :> reduce_mean py: help(pop()) \ <== 看範例，有很簡單理解。
            用 peforth 可以直接試！
            sess :> run(v('tf').reduce_mean([1.,2.])) tib. \ ==> 1.5 (<class 'numpy.float32'>)
            sess :> run(v('tf').reduce_mean([1.,2.,3.])) tib. \ ==> 2.0 (<class 'numpy.float32'>)
            py> 6/4 ==> 1.5
        tf :> reduce_sum py: help(pop()) \ <== 看範例，有範例很容易理解。 Reduce 是指工作在 matrix 的某維度上。
        [ ] reduce_sum 為什麼用負的？
            
        reduction_indices 是舊 argument 即如今的 axis 
            沒有指定 axis 就是全部平均
            sess :> run(v('tf').reduce_mean([1.,1.],[2.,2.])) tib. \ ==> 1.5 (<class 'numpy.float32'>)
            [0] 對 columns 一豎一豎取平均。即 axis=0
            sess :> run(v('tf').reduce_mean([[1.,1.],[2.,2.]],reduction_indices=[0])) tib. \ ==> [ 1.5  1.5] (<class 'numpy.ndarray'>)
            [1] 對 rows 一橫一橫取平均。即 axis=1
            sess :> run(v('tf').reduce_mean([[1.,1.],[2.,2.]],reduction_indices=[1])) tib. \ ==> [ 1.  2.] (<class 'numpy.ndarray'>)
        prediction 聽老師的講，猜是 10 個 digit 的機率，看看。。。

            
    [x] 看他是怎麼 evaluate 的

    [x] 查看 Neuro Network 的形狀
        [x] 22> Wx_plus_b :> shape . cr ==>     (?, 10)
            22> biases :> shape . cr ==>    (1, 10)
            22> Weights :> shape . cr ==>  (784, 10)
            22> sess :> run(v('Weights')) . cr
                [[-1.18724    -0.27526152 -0.98017192 ...,  0.63922352 -0.84101647
                   0.10125981]
                 [ 0.18985745  1.30138469 -0.08802816 ...,  0.95824087 -1.27545691
                  -0.00200335]
                 [ 0.87153876  0.19587995  1.14244318 ...,  0.74499661 -0.72789997
                   1.05316412]
                 ...,
                 [-0.60263687  0.38503972 -0.56423479 ..., -0.00635111 -0.27689263
                   0.49015754]
                 [ 0.53595567  1.51750755 -0.59952027 ..., -0.44937038  1.53320849
                  -0.40922189]
                 [-0.74838656 -0.01508466 -0.15110037 ...,  0.66303551  0.41519049
                  -0.58337498]]
            22> sess :> run(v('biases')) . cr
                [[ 0.16309452 -0.10721014  0.14430207  0.14978598  0.14420623  0.1439701
                   0.03571264  0.15155171  0.14499798  0.02958891]]
        [x] 從上面看起來，整個網路只有 10 顆 Neuron 
            因為 bias 只有 10 個。
            
            sess :> run(v('tf').matmul(v('batch_xs'),v('Weights'))).shape . cr ==> (100, 10)
            sess :> run(v('tf').matmul(v('batch_xs[0]'),v('Weights'))).shape . cr ==> (100, 10)

            周莫烦 https://m.youtube.com/watch?list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&v=aNjdw9w_Qyc
            你说得对,这里的结构只是 input+output, 没有中间的 hidden layer. 
            因为这个只是一个最简单的结构, 如果你想在之中再加上一个或多个 
            hidden layer 都是可以的.只需要使用 add layer 的功能﻿

[x] 察看 compute_accuracy() 內部，看他是怎麼評分的
        33> result type . cr
        <class 'numpy.float32'>
        33> result tib.
        result tib. \ ==> 0.10840000212192535 (<class 'numpy.float32'>)
        越接近 1.00 越好。
        
        33> v_xs . cr    這是 test dataset 共有 10000 張圖片跟相對的 labels 
        [[ 0.  0.  0. ...,  0.  0.  0.]
         [ 0.  0.  0. ...,  0.  0.  0.]
         [ 0.  0.  0. ...,  0.  0.  0.]
         ...,
         [ 0.  0.  0. ...,  0.  0.  0.]
         [ 0.  0.  0. ...,  0.  0.  0.]
         [ 0.  0.  0. ...,  0.  0.  0.]]
        33> v_xs :> shape tib.
        v_xs :> shape tib. \ ==> (10000, 784) (<class 'tuple'>)
        33> v_ys :> shape tib.
        v_ys :> shape tib. \ ==> (10000, 10) (<class 'tuple'>)

        compute_accuracy() 應該就是算出自己的 labels （即 y_pre）來跟 v_ys 比較。

        y_pre :> shape tib. \ ==> (10000, 10) (<class 'tuple'>)
        所以驗證的時候是用 test dataset 這一萬張整批去驗。
        一把就拿到 prediction() 打出來的一萬組 labels. 這一萬組 labels 對不對呢？
        要打分數來看看。

        這幾個都是 Tensor 想看要用 sess.run 過
        prediction type tib. \ ==> <class 'tensorflow.python.framework.ops.Tensor'> (<class 'type'>)
        correct_prediction tib. \ ==> Tensor("Equal:0", shape=(10000,), dtype=bool) (<class 'tensorflow.python.framework.ops.Tensor'>)
        accuracy type tib. \ ==> <class 'tensorflow.python.framework.ops.Tensor'> (<class 'type'>)

        result 是一萬組 label 的正確率，
            result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
        所以想必 accuracy 就是 分子/分母，分母是 10000，分子是正確 labels 的個數。
        或者 mean([0,0,0,1,0,0,0,1,1,1,...]) 標好對錯的每一組 predicted labels. <------ 就是 correct_prediction
           
        sess :> run(v('correct_prediction')) tib. \ ==> [False False False ...,  True False False] (<class 'numpy.ndarray'>)
        sess :> run(v('correct_prediction')).shape tib. \ ==> (10000,) (<class 'tuple'>)

        驗算一下這組 correct_prediction 的平均值
        sess :> run(v('correct_prediction')) constant c-p // ( -- array ) correct_prediction array

        py>~ [(i and 1 or 0) for i in v('c-p')]
        constant CP // ( array ) cooked correct_prediction array
        CP count tib. \ ==> 10000 (<class 'int'>)
        py> sum(v('CP')) tib. \ ==> 1084 (<class 'int'>)
        對了嗎？ 對了！！！
        
        所以確定 correct_prediction 就是上面猜測的東西了，標注如上。
        
        查 tf.argmax 啥東西。。。不用查，看下面跑出來的結果就懂了。
        sess :> run(v('tf').argmax(v('y_pre'),1)) tib. \ ==> [9 6 0 ..., 4 6 4] (<class 'numpy.ndarray'>)
        y_pre :> [0] tib. \ ==> [  1.48504273e-06   5.44097857e-04   5.17828780e-10   1.10162538e-03
           2.71292418e-01   2.83548925e-05   3.12378006e-07   1.44847242e-12
           1.23969920e-01   6.03061795e-01] (<class 'numpy.ndarray'>)
        y_pre :> [1] tib. \ ==> [  2.92419777e-09   1.19141513e-03   1.52244556e-05   1.04044523e-10
           1.69087325e-05   2.00210451e-07   9.98776257e-01   1.76555463e-14
           9.76547334e-14   1.08488241e-09] (<class 'numpy.ndarray'>)
        y_pre :> [2] tib. \ ==> [  3.69310051e-01   2.60156579e-03   4.57426831e-02   3.63132894e-01
           9.81909011e-07   1.96541101e-01   5.33089601e-03   1.19020171e-09
           1.64527833e-07   1.73395965e-02] (<class 'numpy.ndarray'>)
        
        那這樣我猜得到 tf.cast() 的用意了，跟我上面一樣，把 
        correct_prediction cast 成 tf.float32
        
        tf.cast(correct_prediction, tf.float32) 改寫成 peforth 的語法如下，得證:
        sess :> run(v('tf').cast(v('correct_prediction'),v('tf').float32)) tib. \ ==> [ 0.  0.  0. ...,  1.  0.  0.] (<class 'numpy.ndarray'>)
