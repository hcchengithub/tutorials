
    \ include c:\Users\hcche\Documents\GitHub\Morvan\tutorials\tensorflowTUT\tensorflow10_def_add_layer.f 
    
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
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        outport(locals())
        return outputs
    outport(locals())
    </py>
    
    stop

    一開始只有外層的幾個東西
    
        add_layer np tf
        
    先搞懂 shape, 
    
        Weights 的 shape 
        py> [v('in_size'),v('out_size')] tib. \ ==> [3, 1] (<class 'list'>)
        
        Weights 需要 shape 為 (1,3) 的 inputs
        
        np :> array([[1],[2],[3]]).shape tib. \ ==> (3, 1) (<class 'tuple'>)
        np :> array([[1,2,3]]).shape tib. \ ==> (1, 3) (<class 'tuple'>)
        
    執行 add_layer 才會產生 function 裡的東西
    
        add_layer :: ([[1.,2.,3.]],3,1)

        words \ ==> add_layer np tf outputs Wx_plus_b 
            biases Weights activation_function out_size in_size inputs tf
                    
    想看看 TensorFlow 的東西都要用 sesssion.run() 去看，但是要記得先 initialize

        OK tf :> Session() value sess
        OK biases sess :> run(pop()) tib.
        Failed in </py> (compiling=False): Attempting to use uninitialized value Variable_1

    init 又有新舊兩種
    
        OK sess :> run(v('tf').initialize_all_variables()) tib.
        WARNING:tensorflow:From <string>:2: initialize_all_variables 
        (from tensorflow.python.ops.variables) is deprecated and will be 
        removed after 2017-03-02. Instructions for updating:
        Use `tf.global_variables_initializer` instead.
        
        sess :> run(v('tf').initialize_all_variables()) tib. \ ==> None (<class 'NoneType'>)
        sess :> run(v('tf').global_variables_initializer()) tib. \ ==> None (<class 'NoneType'>)

    可以開始把玩了。。。。
    
        biases 初值是 0 + 0.1 shape 是 [1, out_size], shape 是怎樣要想一想  
        biases tib. \ ==> <tf.Variable 'Variable_1:0' shape=(1, 1) dtype=float32_ref> (<class 'tensorflow.python.ops.variables.Variable'>)
        biases sess :> run(pop()) tib. \ ==> [[ 0.1]] (<class 'numpy.ndarray'>)

        Weights 本來就是亂數產生的
        Weights sess :> run(pop()) tib. \ ==> 
        [[-0.34333229]
         [-0.78374338]
         [-0.82603163]] (<class 'numpy.ndarray'>)

        Wx_plus_b sess :> run(pop()) tib. \ ==> [[-4.2889142]] (<class 'numpy.ndarray'>)
        outputs sess :> run(pop()) tib. \ ==> [[-4.2889142]] (<class 'numpy.ndarray'>)
    
    再玩一次，這回 inputs 還是 (1,3) 但 out_size 是 2 。。。
    
        add_layer :: ([[1.,2.,3.]],3,2)
        biases tib.
        biases sess :> run(pop()) tib.
        Weights sess :> run(pop()) tib.
        Wx_plus_b sess :> run(pop()) tib.
        outputs sess :> run(pop()) tib.
        
        可能是因為沒有經過 init 結果與上回都一樣。
        試試 init 一下，
        
        sess :> run(v('tf').global_variables_initializer()) tib.        
        biases sess :> run(pop()) tib. \ ==> [[ 0.1]] (<class 'numpy.ndarray'>)
        Weights sess :> run(pop()) tib. \ ==> 
        [[ 2.30763626]
         [-0.23060918]
         [-0.71045363]] (<class 'numpy.ndarray'>)
        Wx_plus_b sess :> run(pop()) tib. \ ==> [[-0.18494311]] (<class 'numpy.ndarray'>)
        outputs sess :> run(pop()) tib. \ ==> [[-0.18494311]] (<class 'numpy.ndarray'>)

        這不對了，我說 out_size 是 2 的呀！？ 一查，還是 1 
        
        out_size tib. \ ==> 1 (<class 'int'>)

        先 init 完再執行 add_layer() 也一樣。
        不管了，整個重來 ---> 對了！ 如下：
        
        include c:\Users\hcche\Documents\GitHub\Morvan\tutorials\tensorflowTUT\tensorflow10_def_add_layer.f 
        add_layer :: ([[1.,2.,3.]],3,2)  \ 執行完，只是在布置這個 graph 而已
        tf :> Session() value sess \ 變出一個 session 來觀察東西
        sess :> run(v('tf').global_variables_initializer()) tib. \ ==> None (<class 'NoneType'>)
        out_size tib. \ ==> 2 (<class 'int'>)
        biases tib. \ ==> <tf.Variable 'Variable_1:0' shape=(1, 2) dtype=float32_ref> (<class 'tensorflow.python.ops.variables.Variable'>)
        biases sess :> run(pop()) tib. \ ==> [[ 0.1  0.1]] (<class 'numpy.ndarray'>)
        Weights sess :> run(pop()) tib. \ ==> 
        [[ 0.66803414  0.62326759]
         [-0.21582599  0.76987833]
         [-1.09043634  0.86057591]] (<class 'numpy.ndarray'>)
        Wx_plus_b sess :> run(pop()) tib. \ ==> [[-2.93492675  4.84475183]] (<class 'numpy.ndarray'>)
        outputs sess :> run(pop()) tib. \ ==> [[-2.93492675  4.84475183]] (<class 'numpy.ndarray'>)

    所以 add_layer() 是個通用的 function 用來佈建一層 neural layer.
    就這麼簡單，變出了神經網路的一層！ 
    一層其實是一串 :
    
        input -> | neurons |
        input -> | neurons |
        input -> | neurons | -> output
        input -> | neurons | -> output
        input -> | neurons |
        input -> | neurons |
        
        
        
        