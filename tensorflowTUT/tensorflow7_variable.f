
    \ include c:\Users\hcche\Documents\GitHub\ML\tutorials\tensorflowTUT\tensorflow7_variable.f 
    
    --- marker ---
    
    <py>
    # View more python learning tutorial on my Youtube and Youku channel!!!

    # Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
    # Youku video tutorial: http://i.youku.com/pythontutorial

    """
    Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
    """
    # from __future__ import print_function  #11
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #22 https://stackoverflow.com/questions/43134753/tensorflow-wasnt-compiled-to-use-sse-etc-instructions-but-these-are-availab
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
    outport(locals()) #33 
    </py>
    
    stop 
    
    \ 本課的重點是：
    
        TensorFlow 的布置就是在畫一個 Graph ！
        上面 init 之前的程式碼，就是在畫 Graph ！
        See Ynote: "Morvan's tensorflow tutorial #07 筆記"
    
    o init 是 init variables, 可以隨時亂 init
    o 用 peforth 亂玩比用 python 直接跑玩啥也不知道好玩多了！
    o 用 init py: help(pop()) 查看 init 以及其他東西。
    
    tf :> Session() to sess
    
    \ 執行完印出 1 2 3 如下
    OK include c:\Users\hcche\Documents\GitHub\ML\tutorials\tensorflowTUT\tensorflow7_variable.f
    1
    2
    3
    
    \ sess 已經被 close 掉了
    OK tf :> Session() to sess
    OK words
    ... snip ....
    --- _ sess init update new_value one state tf

    \ sess 被 close 掉，Variable 都飄走了，還得重新 init 
    OK sess :> run(v('state')) . cr
    Failed in </py> (compiling=False): Attempting to use uninitialized value counter

    OK state . cr  /* state 是個 tf.Variable */               #
    <tf.Variable 'counter:0' shape=() dtype=int32_ref>        #
    OK init sess :> run(pop()) . cr   /* init Variables */    # Variable 跟
    None                                                      # Tensor 這些東西要
    OK sess :> run(v('state')) . cr                           # 看都得用個 sesssion
    0                                                         # 來 run() 才行。
    OK                                                        #
                                                              #
    \ 每個東西都看一看，熟悉熟悉。。。                        #
                                                              #
    OK one . cr  /* a constant */                             #
    Tensor("Const:0", shape=(), dtype=int32)                  #
                                                              #
    OK new_value . cr   /* sum of tf.add() */                 #
    Tensor("Add:0", shape=(), dtype=int32)                    #
                                                              #
    OK update . cr   /* a variable of tf.assign()  */         #
    Tensor("Assign:0", shape=(), dtype=int32_ref)             #
    
    \ init is an operation
    OK init . cr
        name: "init"
        op: "NoOp"
        input: "^counter/Assign"
    OK init type . cr
        <class 'tensorflow.python.framework.ops.Operation'>
    OK

    \ Sess is an session object 
    OK sess . cr
    <tensorflow.python.client.session.Session object at 0x0000022E4EC4CD30>
    OK
    
    display-off
    sess :> run(v('state'))     tib.
    sess :> run(v('one'))       tib.
    sess :> run(v('new_value')) tib.
    sess :> run(v('update'))    tib.
    display-on
    screen-buffer :> [0] . cr
    OK sess :> run(v('state'))     tib. \ ==> 0 (<class 'numpy.int32'>)
    OK sess :> run(v('one'))       tib. \ ==> 1 (<class 'numpy.int32'>)
    OK sess :> run(v('new_value')) tib. \ ==> 1 (<class 'numpy.int32'>)
    OK sess :> run(v('update'))    tib. \ ==> 1 (<class 'numpy.int32'>)

    <text> locals().update(harry_port()); new_value = tf.add(state, one) </text>
    py: exec(pop())
    display-off
    sess :> run(v('state'))     tib.
    sess :> run(v('one'))       tib.
    sess :> run(v('new_value')) tib.
    sess :> run(v('update'))    tib.
    display-on
    screen-buffer :> [0] . cr
    
    OK     display-off
        sess :> run(v('state'))     tib.
        sess :> run(v('one'))       tib.
        sess :> run(v('new_value')) tib.
        sess :> run(v('update'))    tib.
        display-on
    OK     screen-buffer :> [0] . cr
    OK sess :> run(v('state'))     tib. \ ==> 1 (<class 'numpy.int32'>)
    OK sess :> run(v('one'))       tib. \ ==> 1 (<class 'numpy.int32'>)
    OK sess :> run(v('new_value')) tib. \ ==> 2 (<class 'numpy.int32'>)
    OK sess :> run(v('update'))    tib. \ ==> 2 (<class 'numpy.int32'>)
    OK
    OK     <text> locals().update(harry_port()); new_value = tf.add(state, one) </text>
    OK     py: exec(pop())
    OK     display-off
        sess :> run(v('state'))     tib.
        sess :> run(v('one'))       tib.
        sess :> run(v('new_value')) tib.
        sess :> run(v('update'))    tib.
        display-on
    OK     screen-buffer :> [0] . cr
    OK sess :> run(v('state'))     tib. \ ==> 2 (<class 'numpy.int32'>)
    OK sess :> run(v('one'))       tib. \ ==> 1 (<class 'numpy.int32'>)
    OK sess :> run(v('new_value')) tib. \ ==> 3 (<class 'numpy.int32'>)
    OK sess :> run(v('update'))    tib. \ ==> 3 (<class 'numpy.int32'>)
    OK
    OK     display-off
        sess :> run(v('state'))     tib.
        sess :> run(v('one'))       tib.
        sess :> run(v('new_value')) tib.
        sess :> run(v('update'))    tib.
        display-on
    OK     screen-buffer :> [0] . cr
    OK sess :> run(v('state'))     tib. \ ==> 3 (<class 'numpy.int32'>)
    OK sess :> run(v('one'))       tib. \ ==> 1 (<class 'numpy.int32'>)
    OK sess :> run(v('new_value')) tib. \ ==> 4 (<class 'numpy.int32'>)
    OK sess :> run(v('update'))    tib. \ ==> 4 (<class 'numpy.int32'>)
    OK
    OK     display-off
        sess :> run(v('state'))     tib.
        sess :> run(v('one'))       tib.
        sess :> run(v('new_value')) tib.
        sess :> run(v('update'))    tib.
        display-on
    OK     screen-buffer :> [0] . cr
    OK sess :> run(v('state'))     tib. \ ==> 4 (<class 'numpy.int32'>)
    OK sess :> run(v('one'))       tib. \ ==> 1 (<class 'numpy.int32'>)
    OK sess :> run(v('new_value')) tib. \ ==> 5 (<class 'numpy.int32'>)
    OK sess :> run(v('update'))    tib. \ ==> 5 (<class 'numpy.int32'>)
    OK
    OK     display-off
        sess :> run(v('state'))     tib.
        sess :> run(v('one'))       tib.
        sess :> run(v('new_value')) tib.
        sess :> run(v('update'))    tib.
        display-on
    OK     screen-buffer :> [0] . cr
    OK sess :> run(v('state'))     tib. \ ==> 5 (<class 'numpy.int32'>)
    OK sess :> run(v('one'))       tib. \ ==> 1 (<class 'numpy.int32'>)
    OK sess :> run(v('new_value')) tib. \ ==> 6 (<class 'numpy.int32'>)
    OK sess :> run(v('update'))    tib. \ ==> 6 (<class 'numpy.int32'>)
    OK
    OK     display-off
        sess :> run(v('state'))     tib.
        sess :> run(v('one'))       tib.
        sess :> run(v('new_value')) tib.
        sess :> run(v('update'))    tib.
        display-on
    OK     screen-buffer :> [0] . cr
    OK sess :> run(v('state'))     tib. \ ==> 6 (<class 'numpy.int32'>)
    OK sess :> run(v('one'))       tib. \ ==> 1 (<class 'numpy.int32'>)
    OK sess :> run(v('new_value')) tib. \ ==> 7 (<class 'numpy.int32'>)
    OK sess :> run(v('update'))    tib. \ ==> 7 (<class 'numpy.int32'>)
    OK
    OK     display-off
        sess :> run(v('state'))     tib.
        sess :> run(v('one'))       tib.
        sess :> run(v('new_value')) tib.
        sess :> run(v('update'))    tib.
        display-on
    OK     screen-buffer :> [0] . cr
    OK sess :> run(v('state'))     tib. \ ==> 7 (<class 'numpy.int32'>)
    OK sess :> run(v('one'))       tib. \ ==> 1 (<class 'numpy.int32'>)
    OK sess :> run(v('new_value')) tib. \ ==> 8 (<class 'numpy.int32'>)
    OK sess :> run(v('update'))    tib. \ ==> 8 (<class 'numpy.int32'>)
    OK
    OK     display-off
        sess :> run(v('state'))     tib.
        sess :> run(v('one'))       tib.
        sess :> run(v('new_value')) tib.
        sess :> run(v('update'))    tib.
        display-on
    OK     screen-buffer :> [0] . cr
    OK sess :> run(v('state'))     tib. \ ==> 8 (<class 'numpy.int32'>)
    OK sess :> run(v('one'))       tib. \ ==> 1 (<class 'numpy.int32'>)
    OK sess :> run(v('new_value')) tib. \ ==> 9 (<class 'numpy.int32'>)
    OK sess :> run(v('update'))    tib. \ ==> 9 (<class 'numpy.int32'>)
    OK
    OK     display-off
        sess :> run(v('state'))     tib.
        sess :> run(v('one'))       tib.
        sess :> run(v('new_value')) tib.
        sess :> run(v('update'))    tib.
        display-on
    OK     screen-buffer :> [0] . cr
    OK sess :> run(v('state'))     tib. \ ==> 9 (<class 'numpy.int32'>)
    OK sess :> run(v('one'))       tib. \ ==> 1 (<class 'numpy.int32'>)
    OK sess :> run(v('new_value')) tib. \ ==> 10 (<class 'numpy.int32'>)
    OK sess :> run(v('update'))    tib. \ ==> 10 (<class 'numpy.int32'>)
    OK
    OK     display-off
        sess :> run(v('state'))     tib.
        sess :> run(v('one'))       tib.
        sess :> run(v('new_value')) tib.
        sess :> run(v('update'))    tib.
        display-on
    OK     screen-buffer :> [0] . cr
    OK sess :> run(v('state'))     tib. \ ==> 10 (<class 'numpy.int32'>)
    OK sess :> run(v('one'))       tib. \ ==> 1 (<class 'numpy.int32'>)
    OK sess :> run(v('new_value')) tib. \ ==> 11 (<class 'numpy.int32'>)
    OK sess :> run(v('update'))    tib. \ ==> 11 (<class 'numpy.int32'>)
    OK
    OK     display-off
        sess :> run(v('state'))     tib.
        sess :> run(v('one'))       tib.
        sess :> run(v('new_value')) tib.
        sess :> run(v('update'))    tib.
        display-on
    OK     screen-buffer :> [0] . cr
    OK sess :> run(v('state'))     tib. \ ==> 11 (<class 'numpy.int32'>)
    OK sess :> run(v('one'))       tib. \ ==> 1 (<class 'numpy.int32'>)
    OK sess :> run(v('new_value')) tib. \ ==> 12 (<class 'numpy.int32'>)
    OK sess :> run(v('update'))    tib. \ ==> 12 (<class 'numpy.int32'>)
    OK
    OK     display-off
        sess :> run(v('state'))     tib.
        sess :> run(v('one'))       tib.
        sess :> run(v('new_value')) tib.
        sess :> run(v('update'))    tib.
        display-on
    OK     screen-buffer :> [0] . cr
    OK sess :> run(v('state'))     tib. \ ==> 12 (<class 'numpy.int32'>)
    OK sess :> run(v('one'))       tib. \ ==> 1 (<class 'numpy.int32'>)
    OK sess :> run(v('new_value')) tib. \ ==> 13 (<class 'numpy.int32'>)
    OK sess :> run(v('update'))    tib. \ ==> 13 (<class 'numpy.int32'>)
    OK
    OK     display-off
        sess :> run(v('state'))     tib.
        sess :> run(v('one'))       tib.
        sess :> run(v('new_value')) tib.
        sess :> run(v('update'))    tib.
        display-on
    OK     screen-buffer :> [0] . cr
    OK sess :> run(v('state'))     tib. \ ==> 13 (<class 'numpy.int32'>)
    OK sess :> run(v('one'))       tib. \ ==> 1 (<class 'numpy.int32'>)
    OK sess :> run(v('new_value')) tib. \ ==> 14 (<class 'numpy.int32'>)
    OK sess :> run(v('update'))    tib. \ ==> 14 (<class 'numpy.int32'>)
    OK
    OK     display-off
        sess :> run(v('state'))     tib.
        sess :> run(v('one'))       tib.
        sess :> run(v('new_value')) tib.
        sess :> run(v('update'))    tib.
        display-on
    OK     screen-buffer :> [0] . cr
    OK sess :> run(v('state'))     tib. \ ==> 14 (<class 'numpy.int32'>)
    OK sess :> run(v('one'))       tib. \ ==> 1 (<class 'numpy.int32'>)
    OK sess :> run(v('new_value')) tib. \ ==> 15 (<class 'numpy.int32'>)
    OK sess :> run(v('update'))    tib. \ ==> 15 (<class 'numpy.int32'>)
    OK
    OK     display-off
        sess :> run(v('state'))     tib.
        sess :> run(v('one'))       tib.
        sess :> run(v('new_value')) tib.
        sess :> run(v('update'))    tib.
        display-on
    OK     screen-buffer :> [0] . cr
    OK sess :> run(v('state'))     tib. \ ==> 15 (<class 'numpy.int32'>)
    OK sess :> run(v('one'))       tib. \ ==> 1 (<class 'numpy.int32'>)
    OK sess :> run(v('new_value')) tib. \ ==> 16 (<class 'numpy.int32'>)
    OK sess :> run(v('update'))    tib. \ ==> 16 (<class 'numpy.int32'>)
    OK
    OK     display-off
        sess :> run(v('state'))     tib.
        sess :> run(v('one'))       tib.
        sess :> run(v('new_value')) tib.
        sess :> run(v('update'))    tib.
        display-on
    OK     screen-buffer :> [0] . cr
    OK sess :> run(v('state'))     tib. \ ==> 16 (<class 'numpy.int32'>)
    OK sess :> run(v('one'))       tib. \ ==> 1 (<class 'numpy.int32'>)
    OK sess :> run(v('new_value')) tib. \ ==> 17 (<class 'numpy.int32'>)
    OK sess :> run(v('update'))    tib. \ ==> 17 (<class 'numpy.int32'>)
    OK
    OK     <text> locals().update(harry_port()); new_value = tf.add(state, one) </text>
    OK     py: exec(pop())
    OK     <text> locals().update(harry_port()); new_value = tf.add(state, one) </text>
    OK     py: exec(pop())
    OK     <text> locals().update(harry_port()); new_value = tf.add(state, one) </text>
    OK     py: exec(pop())
    OK     <text> locals().update(harry_port()); new_value = tf.add(state, one) </text>
    OK     py: exec(pop())
    OK     <text> locals().update(harry_port()); new_value = tf.add(state, one) </text>
    OK     py: exec(pop())
    OK     <text> locals().update(harry_port()); new_value = tf.add(state, one) </text>
    OK     py: exec(pop())
    OK     display-off
        sess :> run(v('state'))     tib.
        sess :> run(v('one'))       tib.
        sess :> run(v('new_value')) tib.
        sess :> run(v('update'))    tib.
        display-on
    OK     screen-buffer :> [0] . cr
    OK sess :> run(v('state'))     tib. \ ==> 17 (<class 'numpy.int32'>)
    OK sess :> run(v('one'))       tib. \ ==> 1 (<class 'numpy.int32'>)
    OK sess :> run(v('new_value')) tib. \ ==> 18 (<class 'numpy.int32'>)
    OK sess :> run(v('update'))    tib. \ ==> 18 (<class 'numpy.int32'>)
    OK
    OK
    
    \ 跑了半天，幸好有了悟到：
    