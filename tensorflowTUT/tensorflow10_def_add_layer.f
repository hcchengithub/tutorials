
    \ include c:\Users\hcche\Documents\GitHub\Morvan\tutorials\tensorflowTUT\tensorflow10_def_add_layer.f 
    
    --- marker ---

    <py>
    # View more python learning tutorial on my Youtube and Youku channel!!!

    # This class on Youtube https://www.youtube.com/watch?v=Vu_lIJ_Yexk
    # Source code on GitHub https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tensorflow10_def_add_layer.py
    
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
        
        
Let's see how peforth assists learning a TensorFlow neural network lesson on YouTube.
You don't need to watch it at the moment. I just let you know where is it.

Tensorflow 10 example3 def add_layer() function (neural network tutorials)
https://www.youtube.com/watch?v=Vu_lIJ_Yexk

And its source code on GitHub,
https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tensorflow10_def_add_layer.py
it's very short that we can show it all here:

    from __future__ import print_function
    import tensorflow as tf
    def add_layer(inputs, in_size, out_size, activation_function=None):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

It does almost nothing, only gets the tensorflow module and uses it to define
a function. However, looking at the code I 
can understand that Weights seems like a matrix initialized with random numbers. 
I want to see whether I am correct eagerly, but nothing I can do. 
        
Now with peforth, we can play a lot! Make a few modifications, like this:
    
    <py> #11
    # from __future__ import print_function   #22
    import tensorflow as tf
    def add_layer(inputs, in_size, out_size, activation_function=None):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        outport(locals()) #33
        return outputs
    outport(locals()) #44
    </py> \ #55 

Those #11, #22, .. #55 marks indicate what are modified.
If you have read previous peforth wiki pages then you have known \<py>..\</py> envelops
pythin code when in peforth FORTH environment. 
The only thing new is the function outport() which is called two times
above both use locals() as the input argument. What outport() does is to convert
the given python dict into FORTH variables. Lets play with it in prior further explanations. 

Install peforth,

    pip install peforth 

Run peforth
    
    python -m peforth
    
    p e f o r t h    v1.03
    source code http://github.com/hcchengithub/peforth
    Type 'peforth.ok()' enters forth interpreter, 'exit' to come back.

Peforth supports multiple line input. Ctrl-C copy the above code, press Ctrl-D
in peforth interpreter to open a multiple line input, press Ctrl-V to paste
the code, then press the ending Ctrl-D, as shown below: (if Ctrl-V does not
work then press Alt-Space > Edit > Paste or change your DOS Box settings)

    p e f o r t h    v1.03
    source code http://github.com/hcchengithub/peforth
    Type 'peforth.ok()' enters forth interpreter, 'exit' to come back.
    
    OK ^D
        <py> #11
        # from __future__ import print_function   #22
        import tensorflow as tf
        def add_layer(inputs, in_size, out_size, activation_function=None):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
            if activation_function is None:
                outputs = Wx_plus_b
            else:
                outputs = activation_function(Wx_plus_b)
            outport(locals()) #33
            return outputs
        outport(locals()) #44
        </py> \ #55
    ^D
    OK

Now, let's see what happened

    OK words
    0 code end-code \ // \<selftest> \</selftest> bye /// immediate stop 
    compyle trim indent -indent \<py> \</py> \</pyV> words . cr help 
    interpret-only compile-only literal reveal privacy (create) : ; ( 
    ... snip ...
    test-result [all-pass] *** OK dir keys --- add_layer tf
    OK

See the last two FORTH words "add_layer" and "tf"? They are FORTH "value"s 
created by the 
line: 

    outport(locals()) #44

At that time the tensorflow module "tf" and the function "add_layer" 
were only local variables existing in the \<py>..\</py> section. And we have bring them
out to FORTH for our examinations. Let see them in some different view angles:

Their default appearance, like normal python things, are to show what they are,
in fact their 'type':

    OK tf . cr
    <module 'tensorflow' from 'C:\\Users\\hcche\\AppData\\Local\\Programs\\Python\\Python36\\lib\\site-packages\\tensorflow\\__init__.py'>
    
    OK add_layer . cr
    <function compyle_anonymous.<locals>.add_layer at 0x00000208831EFBF8>
    
We can even see the 'help' of the function we defined! This is a python's 
intrinsic featrue, good to you python!

    OK add_layer py: help(pop())
    Help on function add_layer in module peforth.projectk:

    add_layer(inputs, in_size, out_size, activation_function=None)

While TensorFlow's help is a real epic!

    OK tf py: help(pop())
    Help on package tensorflow:

    NAME
        tensorflow

    DESCRIPTION
        # Copyright 2015 The TensorFlow Authors. All Rights Reserved.
        #
        # Licensed under the Apache License, Version 2.0 (the "License");
        # you may not use this file except in compliance with the License.
        # You may obtain a copy of the License at
        #
        #     http://www.apache.org/licenses/LICENSE-2.0
        #
        # Unless required by applicable law or agreed to in writing, software
        # distributed under the License is distributed on an "AS IS" BASIS,
        # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        # See the License for the specific language governing permissions and
        # limitations under the License.
        # ==============================================================================

    PACKAGE CONTENTS
        contrib (package)
        core (package)
        examples (package)
        
    ... snip ....
    
I always want to see 'type' and 'dir' of things in python to have
a clearer picture of them:
    
    OK tf type . cr
    <class 'module'>
    
    OK add_layer type . cr
    <class 'function'>
    
    OK tf dir . cr
    ['AggregationMethod', 'Assert', 'AttrValue', 'COMPILER_VERSION', 
    .... snip .... 
    'while_loop', 'write_file', 'zeros', 'zeros_initializer', 'zeros_like', 'zeta']
    
    OK add_layer dir . cr
    ['__annotations__', '__call__', '__class__', '__closure__', 
    ... snip ...
    '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__']

It's fun to go on playing with each of the above attributes.
But we want to look deeper into the add_layer() function. If you have read
[]() then this line is no problem to you:

    add_layer :: ([[1.,2.,3.]],3,2)  \ execute add_layer() 
        \ inputs is [[1.,2.,3.]] which is a list of shape (1,3)
        \ in_size is 3
        \ out_size is 2

Remember we added this line into the add_layer function?

    outport(locals()) #33

It makes all python local variables that can be seen at the moment 
become FORTH values. Let's see:

    OK words
    0 code end-code \ // \<selftest> \</selftest> bye /// immediate 
    stop compyle trim indent -indent <py> </py> </pyV> words . cr help 
    ... snip... dir keys --- add_layer tf outputs Wx_plus_b biases Weights 
    activation_function out_size in_size inputs tf
    OK
    
Note many new words appeared from "outputs" ... to "inputs tf". The last
"tf" has actually triggered a "reDef tf" warning to which you may have noticed. 
That indicates the TensorFlow module is seen in add_layer() function too. 
As the teacher, Morvan, has told us in his previous lessons, we need a TensorFlow 
Session to see a object. So do we create it like this:
    
    tf :> Session() value sess // ( -- obj ) A TensorFlow session object

And we need to initialize TensorFlow. I am new too so I don't know why 
either, just do what teacher said:
    
    sess :> run(v('tf').global_variables_initializer()) tib. \ ==> None (<class 'NoneType'>)

Here, ```tib.``` command is like ``` . cr ``` but it prints the entire 
command line and shows the type of the return value, good for studying.
``` v('tf') ``` is the way to access the FORTH value 'rf' within python
code. Another way to do the samething is like this:
    
    OK tf sess :> run(pop().global_variables_initializer()) tib.
    tf sess :> run(pop().global_variables_initializer()) tib. \ ==> None (<class 'NoneType'>)
    OK    

But we can't use 'tf' directly like this:
    
    OK sess :> run(tf.global_variables_initializer()) tib.

    Failed in </py> (compiling=False): name 'tf' is not defined
    Body:
    push(pop().run(tf.global_variables_initializer()))
    OK

Because we are no longer in add_layer() nor the outer \<py>...\</py> block. 
However, through our FORTH values we can examine all those once existed 
local variables in the add_layer() function:

    out_size tib. \ ==> 2 (<class 'int'>)
    
    biases tib. \ ==> \<tf.Variable 'Variable_1:0' shape=(1, 2) dtype=float32_ref> (<class 'tensorflow.python.ops.variables.Variable'>)
    
    biases sess :> run(pop()) tib. \ ==> [[ 0.1  0.1]] (<class 'numpy.ndarray'>)
    
    Weights sess :> run(pop()) tib. \ ==> 
    [[ 0.66803414  0.62326759]
     [-0.21582599  0.76987833]
     [-1.09043634  0.86057591]] (<class 'numpy.ndarray'>)
     
    Wx_plus_b sess :> run(pop()) tib. \ ==> [[-2.93492675  4.84475183]] (<class 'numpy.ndarray'>)
    
    outputs sess :> run(pop()) tib. \ ==> [[-2.93492675  4.84475183]] (<class 'numpy.ndarray'>)

Due to FORTH programming language's simplicity in syntacs and the freedom it brings,
I enjoy doing the exercises with peforth when studying TensorFlow or even python itself.

May the FORTH be with you, and 
Happy programming !






