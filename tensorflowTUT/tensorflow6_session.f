
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

    matrix1 = tf.constant([[3, 3]])
    matrix2 = tf.constant([[2],
                           [2]])
    product = tf.matmul(matrix1, matrix2)  # matrix multiply np.dot(m1, m2)

    # method 1
    sess = tf.Session()
    result = sess.run(product)
    print(result)
    # sess.close() 故意不 close, 傳回 locals() 玩玩看...

    # method 2
    with tf.Session() as sess:
        result2 = sess.run(product)
        print(result2)

    push(locals())
    </py>

    value l // ( -- locals ) the playground locals
    
    stop 

    OK l :> keys() .  # 注意, 不是 dir 看到的 attributes.
    dict_keys(['result2', 'result', 'sess', 'product', 'matrix2', 'matrix1', 'tf'])OK

    # 雖然沒有 sess.close(), 後來才發現 session 還是結束了。可能離開 <Py> section 時就已經
    # closed 了。
        OK l :> ['sess'] value sess
        OK sess .  ==> <tensorflow.python.client.session.Session object at 0x000001B7952CCFD0>

    # result 當然還在
        l :> ['result2'] tib. \ ==> [[12]] (<class 'numpy.ndarray'>)

    # 我想用 sess 去跑東西，跳過 product 不生成，看可不可以？
        OK r@ sess :> run(pop()) .s
        Failed in </py> (compiling=False): Attempted to use a closed Session.
        Body:
        push(pop().run(pop()))

    # 只好重新創建一個 session，在弄一個 constant 放在 data stack 裡
    # 果然可以 access tf.constant, 也能重新執行 product 
        OK tf :> Session() to sess
        OK tf :> constant(123)
        OK >r
        OK r@ .
        Tensor("Const_4:0", shape=(), dtype=int32)OK
        OK r@ sess :> run(pop()) .s
              0: 123 (<class 'numpy.int32'>)

        OK l :> ['product'] tib. \ ==>  Tensor("MatMul_1:0", shape=(1, 1), dtype=int32)
        OK l :> ['product'] sess :> run(pop()) tib.
        l :> ['product'] sess :> run(pop()) tib. \ ==> [[12]] (<class 'numpy.ndarray'>)
        OK