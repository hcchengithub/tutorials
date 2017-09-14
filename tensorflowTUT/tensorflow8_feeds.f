
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

    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.multiply(input1, input2)

    with tf.Session() as sess:
        print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))
    
    outport(locals())
    </py>
    
    stop
  
    隨便讓 sess 去跑個東西，就可以看出它是否已經 closed :
    
        OK sess :> run(1)
        Failed in </py> (compiling=False): Attempted to use a closed Session.

    搞個新的 sess
    
        tf :> Session() to sess 
    
    試試看如何去 run 一個 placeholder，手動執行，成功了！
    
        sess :> run(v('input1'),feed_dict={v('input1'):[3]}) tib. 
        \ ==> [ 3.] (<class 'numpy.ndarray'>) 會自動把 3 convert 成 3. 
        sess :> run(v('output'),feed_dict={v('input1'):[7.],v('input2'):[7.]}) tib. 
        \ ==> [ 49.] (<class 'numpy.ndarray'>)
        
    feed_dict 裡面用 object 當 key 很新鮮，多試幾下
    
        output input1 input2 sess :> run(pop(2),feed_dict={pop(1):[7.],pop():[7.]}) tib. 
        \ ==> [ 49.] (<class 'numpy.ndarray'>)
        
        看 dict 能不能當 dict key?
            OK .s
                  0: {'a': 11, 'b': 22} (<class 'dict'>)
            OK constant ddd
            OK ddd . \ ==> {'a': 11, 'b': 22} ddd 是個 dictionary
            OK ddd py> {pop():222} 直接用 ddd 當 key ....
            Failed in </py> (compiling=False): unhashable type: 'dict'
            Body:
            push({pop():222})
            OK
        看 obj 能不能當 dict key? 可以，可以，可以！
            <py>
            def dummy():
                pass
            push(dummy)
            </py> constant ooo
            OK ooo py> {pop():222}  真的可以！
            OK .s
                  0: {<function compyle_anonymous.<locals>.dummy at 0x000001A2F4DC3620>: 222} (<class 'dict'>)
            OK constant hash
            OK ooo hash :> [pop()] . \ ==> 222 真的是這樣！
            
        看 obj 能不能當 attribute? 不行了，哈。Error message 如下，徹底拒絕。
        
            <py>
            def dummy2():
                pass
            push(dummy2)
            </py> constant ooo2
            OK ooo ooo2 py> setattr(pop(1),pop(),11111)  
            Failed in </py> (compiling=False): attribute name must be string, not 'function'

        
    TensorFlow 的「東西」都有很長的 help. input1 跟 output 都是一樣的 help 
    
        cls input1 py: help(pop())
        cls output py: help(pop())

    They are of the same type. They are all Tensors.
    
        input1 type tib. \ ==> <class 'tensorflow.python.framework.ops.Tensor'> (<class 'type'>)
        output type tib. \ ==> <class 'tensorflow.python.framework.ops.Tensor'> (<class 'type'>)
