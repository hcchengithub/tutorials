    \ include c:\Users\hcche\Documents\GitHub\Morvan\tutorials\tensorflowTUT\tf17_dropout\full_code.f
    
    --- \ remove older garbage
    
    : ce ( -- value ) // Get value of cross_entropy
        <text>
        locals().update(harry_port())
        rslt = sess.run(cross_entropy,feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
        push(rslt)
        </text> \ 不能用 <py>..</py> 否則 sess 會被 compile 而那時未定義
        -indent py: exec(pop()) ;

    py:~ import PIL.Image as image; push(image)
    value image
    image :> new("L",(8,8)) value pic // ( -- PIL.image ) object
      
    \ Macro 形式用在 [for]...[next] 會有問題 [ ] 待解
    \ : digit ( index -- ) // View the handwritten digit image pointed by the index
    \     <text>
    \     digits :> data[pop()] py> tuple(pop()*16) pic :: putdata(pop()) pic :: show() 
    \     </text> dictate ;

    marker --- \ set a fense
    
    <py>  #11
    # View more python learning tutorial on my Youtube and Youku channel!!!

    # Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
    # Youku video tutorial: http://i.youku.com/pythontutorial

    """
    Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
    """
    # from __future__ import print_function   #22
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #33 https://stackoverflow.com/questions/43134753/tensorflow-wasnt-compiled-to-use-sse-etc-instructions-but-these-are-availab
    import tensorflow as tf
    from sklearn.datasets import load_digits
    from sklearn.cross_validation import train_test_split
    from sklearn.preprocessing import LabelBinarizer

    # load data
    digits = load_digits()
    X = digits.data
    y0 = digits.target  #44
    y = LabelBinarizer().fit_transform(y0)    #55
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)


    def add_layer(inputs, in_size, out_size, layer_name, activation_function=None, ):
        # add one more layer and return the output of this layer
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        # here to dropout
        Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.summary.histogram(layer_name + '/outputs', outputs)
        if vm.debug==66:ok('66> ',loc=locals(),cmd=":> [0] inport") #66
        return outputs


    # define placeholder for inputs to network
    keep_prob = tf.placeholder(tf.float32)
    xs = tf.placeholder(tf.float32, [None, 64])  # 8x8
    ys = tf.placeholder(tf.float32, [None, 10])

    # add output layer
    l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
    prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)

    # the loss between prediction and real data
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))  # loss
    tf.summary.scalar('loss', cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.Session()
    merged = tf.summary.merge_all()
    # summary writer goes in here
    train_writer = tf.summary.FileWriter("logs/train", sess.graph)
    test_writer = tf.summary.FileWriter("logs/test", sess.graph)

    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(500):
        # here to determine the keeping probability
        sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})
        if i % 50 == 0:
            # record loss
            train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
            test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
            train_writer.add_summary(train_result, i)
            test_writer.add_summary(test_result, i)
            if vm.debug==77:ok('77> ',loc=locals(),cmd="marker --- :> [0] inport") #77
            print(sess.run(cross_entropy,feed_dict={xs: X_train, ys: y_train, keep_prob: 1}), end="\t") #88
            print(sess.run(cross_entropy,feed_dict={xs: X_test, ys: y_test, keep_prob: 1})) #99
    outport(locals()) #99
    </py>



    stop

    \ 先成功執行 tutorial

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

    18:35 2017-10-12
    tensorboard 畫出來了 See Ynote: "Use peforth to study Morvan's tensorflow tutorial #17"
    No breakpoint so nothing we can see. Now set breakpoint at

    我在 ~/downloads/logs 裡面堆了執行多次產生的 log 檔，如下 （所以 Morvan 執行新的要
    把舊的刪除）。Tensorboard 會畫出全部

        c:\Users\hcche\Downloads\logs>dir /s /w
         Volume in drive C is Windows
         Volume Serial Number is ACE4-3690

         Directory of c:\Users\hcche\Downloads\logs

        [.]          [..]         readme.txt   [test]       [train]
                       1 File(s)            136 bytes

         Directory of c:\Users\hcche\Downloads\logs\test

        [.]                                           [..]
        events.out.tfevents.1507804099.WKS-4AEN0404   events.out.tfevents.1507876129.WKS-4AEN0404
        events.out.tfevents.1507891514.WKS-4AEN0404   events.out.tfevents.1507891756.WKS-4AEN0404
        events.out.tfevents.1507943444.WKS-4AEN0404
                       5 File(s)        747,848 bytes

         Directory of c:\Users\hcche\Downloads\logs\train

        [.]                                           [..]
        events.out.tfevents.1507804099.WKS-4AEN0404   events.out.tfevents.1507876127.WKS-4AEN0404
        events.out.tfevents.1507891514.WKS-4AEN0404   events.out.tfevents.1507891756.WKS-4AEN0404
        events.out.tfevents.1507943444.WKS-4AEN0404
                       5 File(s)        752,568 bytes

             Total Files Listed:
                      11 File(s)      1,500,552 bytes
                       8 Dir(s)  237,247,062,016 bytes free

        c:\Users\hcche\Downloads\logs>

    \ 先查看資料，這 Tutorial 又是個手寫辨識，但資料用 sklearn 的

    77> digits dir . cr
    ['DESCR', 'data', 'images', 'target', 'target_names']

    77> digits :> DESCR . cr
    Optical Recognition of Handwritten Digits Data Set
    ===================================================

    Notes
    -----
    Data Set Characteristics:
        :Number of Instances: 5620  <======= 錯！只有 1797 張圖
        :Number of Attributes: 64
        :Attribute Information: 8x8 image of integer pixels in the range 0..16.
        :Missing Attribute Values: None
        :Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)
        :Date: July; 1998

    This is a copy of the test set of the UCI ML hand-written digits datasets
    http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

    The data set contains images of hand-written digits: 10 classes where
    each class refers to a digit.

    Preprocessing programs made available by NIST were used to extract
    normalized bitmaps of handwritten digits from a preprinted form. From a
    total of 43 people, 30 contributed to the training set and different 13
    to the test set. 32x32 bitmaps are divided into nonoverlapping blocks of
    4x4 and the number of on pixels are counted in each block. This generates
    an input matrix of 8x8 where each element is an integer in the range
    0..16. This reduces dimensionality and gives invariance to small
    distortions.

    For info on NIST preprocessing routines, see M. D. Garris, J. L. Blue, G.
    T. Candela, D. L. Dimmick, J. Geist, P. J. Grother, S. A. Janet, and C.
    L. Wilson, NIST Form-Based Handprint Recognition System, NISTIR 5469,
    1994.

    References
    ----------
      - C. Kaynak (1995) Methods of Combining Multiple Classifiers and Their
        Applications to Handwritten Digit Recognition, MSc Thesis, Institute of
        Graduate Studies in Science and Engineering, Bogazici University.
      - E. Alpaydin, C. Kaynak (1998) Cascading Classifiers, Kybernetika.
      - Ken Tang and Ponnuthurai N. Suganthan and Xi Yao and A. Kai Qin.
        Linear dimensionalityreduction using relevance weighted LDA. School of
        Electrical and Electronic Engineering Nanyang Technological University.
        2005.
      - Claudio Gentile. A New Approximate Maximal Margin Classification
        Algorithm. NIPS. 2000.

    77> digits :> data . cr
    [[  0.   0.   5. ...,   0.   0.   0.]
     [  0.   0.   0. ...,  10.   0.   0.]
     [  0.   0.   0. ...,  16.   9.   0.]
     ...,
     [  0.   0.   1. ...,   6.   0.   0.]
     [  0.   0.   2. ...,  12.   0.   0.]
     [  0.   0.  10. ...,  12.   1.   0.]]
    77> digits :> data type . cr
    <class 'numpy.ndarray'>
    77> digits :> data.shape . cr
    (1797, 64)
    77> digits :> data[0] . cr
    [  0.   0.   5.  13.   9.   1.   0.   0.   0.   0.  13.  15.  10.  15.   5.
       0.   0.   3.  15.   2.   0.  11.   8.   0.   0.   4.  12.   0.   0.   8.
       8.   0.   0.   5.   8.   0.   0.   9.   8.   0.   0.   4.  11.   0.   1.
      12.   7.   0.   0.   2.  14.   5.  10.  12.   0.   0.   0.   0.   6.  13.
      10.   0.   0.   0.]
    77> digits :> data[0].shape . cr
    (64,)
    77>

    \ digits.images 只是 digits.data 的不同 form 而已，內容一樣。
    
    77> digits :> images.shape . cr
    (1797, 8, 8)
    77> digits :> images[0] .  cr
    [[  0.   0.   5.  13.   9.   1.   0.   0.]
     [  0.   0.  13.  15.  10.  15.   5.   0.]
     [  0.   3.  15.   2.   0.  11.   8.   0.]
     [  0.   4.  12.   0.   0.   8.   8.   0.]
     [  0.   5.   8.   0.   0.   9.   8.   0.]
     [  0.   4.  11.   0.   1.  12.   7.   0.]
     [  0.   2.  14.   5.  10.  12.   0.   0.]
     [  0.   0.   6.  13.  10.   0.   0.   0.]]
    77>

    \ digits.target 就是 label （target 即 label）
    
    77> digits :> target . cr
    [0 1 2 ..., 8 9 8]
    77> digits :> target type . cr
    <class 'numpy.ndarray'>
    77> digits :> target.shape . cr
    (1797,)
    77>

    \ digits.target_names 沒什麼，就標示 label 而已
    77> digits :> target_names . cr
    [0 1 2 3 4 5 6 7 8 9]
    77>

    \ 查看圖片
    
    77> py:~ import PIL.Image as image; push(image)
    77> value image
    77> image :> new("L",(8,8)) value pic // ( -- PIL.image ) object
    77> digits :> data[0] py> tuple(pop()*16) pic :: putdata(pop()) pic :: show()
    77> digits :> data[1] py> tuple(pop()*16) pic :: putdata(pop()) pic :: show()
    
    : digit ( index -- ) // View the handwritten digit image pointed by the index
        digits :> data[pop()] py> tuple(pop()*16) pic :: putdata(pop()) pic :: show() ;

    \ 一口氣看 20 張圖片
    77> 20 [for] t@ digit [next]

    \ Trace y0 and y 只是把 y0 改成 binarized
    77> y0 :> [1000:1020] . cr
    [1 4 0 5 3 6 9 6 1 7 5 4 4 7 2 8 2 2 5 7]
    77> y0 :> shape . cr
    (1797,)
    
    77> y type . cr
    <class 'numpy.ndarray'>
    77> y :> shape . cr
    (1797, 10)  <----------------- binarized
    77> y :> [1000:1020] . cr
    [[0 1 0 0 0 0 0 0 0 0]
     [0 0 0 0 1 0 0 0 0 0]
     [1 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 1 0 0 0 0]
     [0 0 0 1 0 0 0 0 0 0]
     [0 0 0 0 0 0 1 0 0 0]
     [0 0 0 0 0 0 0 0 0 1]
     [0 0 0 0 0 0 1 0 0 0]
     [0 1 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 1 0 0]
     [0 0 0 0 0 1 0 0 0 0]
     [0 0 0 0 1 0 0 0 0 0]
     [0 0 0 0 1 0 0 0 0 0]
     [0 0 0 0 0 0 0 1 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 1 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 0 0 0 1 0 0 0 0]
     [0 0 0 0 0 0 0 1 0 0]]
    77>

    \ train_test_split(X, y, test_size=.3) is easily understandable
    
    77> X_train :> shape . cr
    (1257, 64)
    77> y_train :> shape . cr
    (1257, 10)
    77> y_test :> shape . cr
    (540, 10)
    77> 1257 540 over over + .s
          0:           0           0h (<class 'int'>)
          1:       1,257         4E9h (<class 'int'>)
          2:         540         21Ch (<class 'int'>)
          3:       1,797         705h (<class 'int'>)
    77> / . cr
    0.3005008347245409  <--------- test_size=.3

    77> keep_prob . cr  <---- [ ] 老師說這啥？
    Tensor("Placeholder:0", dtype=float32)
    77> xs . cr
    Tensor("Placeholder_1:0", shape=(?, 64), dtype=float32)
    77> ys . cr
    Tensor("Placeholder_2:0", shape=(?, 10), dtype=float32)

    77> l1 . cr
    Tensor("Tanh:0", shape=(?, 50), dtype=float32)

    77> prediction . cr
    Tensor("Softmax:0", shape=(?, 10), dtype=float32)

    77> cross_entropy . cr
    Tensor("Mean:0", shape=(), dtype=float32)

    77> train_step . cr
    name: "GradientDescent"
    op: "NoOp"
    input: "^GradientDescent/update_Variable/ApplyGradientDescent"
    input: "^GradientDescent/update_Variable_1/ApplyGradientDescent"
    input: "^GradientDescent/update_Variable_2/ApplyGradientDescent"
    input: "^GradientDescent/update_Variable_3/ApplyGradientDescent"
    77> train_step type . cr
    <class 'tensorflow.python.framework.ops.Operation'>
    77>

    \ 看看，不太懂 summary ── 好像就是 tensorboard 的東西。
    \ 先把正常結果跑出來再說。。。

    \ cross_entropy 就是 loss (或 cost) 用 peforth 可以就可以直接查看
    77> sess :>~ run(v('cross_entropy'),feed_dict={v('xs'): v('X_train'), v('ys'): v('y_train'), v('keep_prob'): 1})
    77> . cr
    8.80481  <--------- 一開始的 loss 很大
    77>

    \ train_result

    OK words    從 outport(locals()) 取得的 values
    ...snip ....
    --- test_result train_result i init test_writer train_writer 
    merged sess train_step cross_entropy prediction l1 ys xs 
    add_layer y_test y_train X_test X_train y y0 X digits LabelBinarizer 
    train_test_split load_digits tf keep_prob digit
    
    OK i . cr
    499

    \ train_result 是 summary tensorboard 的東西，
    OK train_result . cr
    b'\n\xb2\x0b\n\nl1/outputs*\xa3\x0b\t\x00\x00\x00 ... snip ...
    \x00\x00\x00\x00\x00\x00\x00\n\x0b\n\x04loss\x15\x10p:>'
    
    OK train_result type . cr
    <class 'bytes'>
    
    \ train_result test_result are both generated from merged
    OK merged type . cr
    <class 'tensorflow.python.framework.ops.Tensor'>
    OK merged . cr
    Tensor("Merge/MergeSummary:0", shape=(), dtype=string)
    
    OK train_result :> [0] . cr
    10
    OK train_result :> [0] type . cr
    <class 'int'>
    OK test_result :> [0] type . cr
    <class 'int'>
    OK test_result :> [0] . cr
    10
    
    \ see their count 
    OK test_result count . cr
    5847
    OK train_result count . cr
    6151
    OK    
    
    \ 我看懂了
    See Ynote: "Use peforth to study Morvan's tensorflow tutorial #17"
    "那如果是 train , test 兩組作比較，這有何意義呢？從曲線看來，兩組都是 
    cross_entropy 的下降曲線，那等於是用了兩組資料做了兩次，看看兩次的 loss 
    曲線一不一致。我做 moocs 題目時，也用過這個方式評估各種不同 classifier 
    的效果，還是不知道那個好，但有三個就可以用投票的來比。如今是自己跟自己比，
    那就是比穩定性。這麼說，分組時 train_test_split(X, y, test_size=.3) 
    就應該用 0.5 才對，而非 0.3。改成 0.5 看看。。。。。"
    
    我自己取得 train test 兩組 cross_entropy 用 excel 畫成曲線來比較也可以，
    無需用 tensorboard 這東西，平添學習負擔。See the Ynote , excel really 
    works too.
    
    