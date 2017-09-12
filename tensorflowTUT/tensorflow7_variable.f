
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

    state = tf.Variable(0, name='counter')
    #print(state.name)
    one = tf.constant(1)

    new_value = tf.add(state, one)
    update = tf.assign(state, new_value)

    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for _ in range(3):
            sess.run(update)
            print(sess.run(state))
    outport(locals())
    </py>
    
    stop 
    
    玩後心得
    o init 是 init variables, 可以隨時亂 init
    o 用 peforth 亂玩比用 python 直接跑玩啥也不知道好玩多了！
    o 用 init py: help(pop()) 查看 init 以及其他東西。
    
    
    
    
