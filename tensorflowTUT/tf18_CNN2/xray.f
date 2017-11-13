
    \ Change DOSBox title
    :> [0] constant parent // ( -- locals ) Caller's locals() dict
    s" dos title " __main__ :> __file__ + CRLF + dictate drop

    \ Imports 
    \ 把 mnist_data 放在了公共的地方，改由 peforth 來 import 避免重複浪費時間、空間。
    
    py> os.getcwd() constant working-directory // ( -- "path" ) Saved copy of tutorial home directory path
    
    \ my MNIST_data directory is there
    cd c:\Users\hcche\Downloads
    
    py:~ from tensorflow.examples.tutorials.mnist import input_data; push(input_data)
    constant mnist_module // ( -- mnist_module ) module

    \ # Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
    \ # mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
    mnist_module :>~ read_data_sets('MNIST_data', one_hot=True)
    constant mnist // ( -- datasets ) 3 datasets (train, validation, test)
    mnist parent :: ['mnist']=pop(1) \ feedback to the tutorial 
    mnist_module parent :: ['input_data']=pop(1) \ feedback to the tutorial 

    working-directory py: os.chdir(pop()) \ Go home
    
    exit break-include \ ------------------- Never Land -------------------------------------------
   
    ." Error!! You reached never land, what's the problem?" cr
    ." Error!! You reached never land, what's the problem?" cr
    ." Error!! You reached never land, what's the problem?" cr
    ." Error!! You reached never land, what's the problem?" cr
    ." Error!! You reached never land, what's the problem?" cr
    ." Press enter to continue but don't!" accept
    
    \ Common tools 

    \ To drop a breakpoint into python
    import peforth;peforth.ok('11> ',loc=locals(),cmd="parent inport")
   
    
    dos title Tensorflow MNIST tutorial playground

    1000 value pause // ( -- n ) Loop count to pause 
    : autoexec pause py> tos(1)%pop() not and if py: ok('loop-100>>',cmd="cr") then ;
    // ( i -- ) Auto-run at breakpoint 
    
    cr ."     Tensorflow version "  tf :> __version__ . cr
    
    <text>
    
    MNIST dataset imported.
    You can now make some setups, like py: vm.debug=22 or the likes.
    You can play around and 'exit' to continue the tutorial.

    </text> . cr
    
    marker ---xray---
    \ py: vm.debug=22

    \ Initial Check 
    dos title working play ground
    cr version drop ." Current directory is : " cd
    dos if exist MNIST_data exit 34157
    34157 = [if] 
        ." TenserFlow dataset ./MNIST_data is existing. Good! Good! Good! let's go on ...." cr 
        \ exit \ <---------- exit to tutorial
    [else]
        ." TenserFlow dataset at ./MNIST_data is expected but not found!" cr
        ." Move it over to here if you have it already." cr
        \ ." Type <Enter> to proceed downloading it or 'abort' to STOP "
        \ accept char abort = [if] ." Action aborted by user." cr bye \ terminate
        \ [else] exit *debug* 22 ( <---------------- exit to tutorial ) [then] 
    [then]
    ." Type <Enter> to proceed " accept drop 
    break-include
    

    \ 抽換 marker 界線，把 --- 改成 ---xray--- replace marker 
        <accept> <text> 
        locals().update(harry_port());  # bring in all FORTH value.outport
        dictate("### marker ---xray---"); outport(locals()) # bring out all locals()
        </text> -indent py: exec(pop())
        </accept> dictate 
    
    \ This snippet adds batch_X, batch_Y into value.outport for investigation
        <accept> <text> 
        locals().update(harry_port());  # bring in all things
        # ------------ get what we want --------------------------
        batch_X, batch_Y = mnist.train.next_batch(100);  
        # ------------ get what we want --------------------------
        dictate("---xray--- marker ---xray---"); outport(locals()) # bring out all things
        </text> -indent py: exec(pop())
        </accept> dictate 

    bp11> batch_X :> [0].shape . cr
    (28, 28, 1)
    bp11> batch_X :> shape . cr
    (100, 28, 28, 1)
    bp11>
    bp11> batch_Y :> shape . cr
    (100, 10)
    bp11> batch_Y :> [0] . cr
    [ 0.  0.  0.  0.  0.  0.  0.  0.  1.  0.]
    bp11>

    \ If we can see placeholders X and Y then we can see anything...
    <text>
    locals().update(harry_port());  # bring in all things
    myX = sess.run(X,feed_dict={X: batch_X, Y_: batch_Y})
    myY = sess.run(Y,feed_dict={X: batch_X, Y_: batch_Y})
    ok(cmd="---xray--- marker ---xray--- exit");  # clear things in forth
    outport(locals())
    </text> -indent py: exec(pop()) 

    \ it works!!! hahahaha
    bp11> words
    ... snip ....
    yclude) pyclude .members .source dos cd ### --- __name__ __doc__ __package__ 
    __loader__ __spec__ __annotations__ __builtins__ __file__ __cached__ peforth 
    tf tensorflowvisu mnist_data mnist X Y_ W b XX Y cross_entropy 
    correct_prediction accuracy train_step allweights allbiases I It datavis init 
    sess training_step batch_X batch_Y myX myY
    bp11>              ^^^^^^^^^^^^^^^^^^^^^^^^------Bingo!!                

    \ peforth can import modules with the ability of cd, dos,
    \ os.chdir() and os.getcwd() they can be at different paths
    
    py:~ import tensorflow as tf; push(tf)
    parent :: ['tf']=pop(1)
    py:~ import tensorflowvisu; push(tensorflowvisu)
    parent :: ['tensorflowvisu']=pop(1)

    
    \ include c:\Users\hcche\Documents\GitHub\ML\tutorials\tensorflowTUT\tf18_CNN2\full_code.f 
    
    --- \ remove older garbage
    
    : ce ( -- value ) // Get value of cross_entropy
        <text>
        locals().update(harry_port())
        rslt = sess.run(cross_entropy,feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
        push(rslt)
        </text> \ 不能用 <py>..</py> 否則 sess 會馬上被 compile 而那時 harry_port() 帶進的 variables 未定義
        -indent py: exec(pop()) ;

    py:~ import PIL.Image as image; push(image)
    value image
    image :> new("L",(8,8)) value pic // ( -- PIL.image ) object
      
    \ Macro 形式用在 [for]...[next] 會有問題 [ ] 待解 --> 用 exec(code) 看看。。。。
    \ : digit ( index -- ) // View the handwritten digit image pointed by the index
    \     <text>
    \     digits :> data[pop()] py> tuple(pop()*16) pic :: putdata(pop()) pic :: show() 
    \     </text> dictate ;

    marker --- \ set a fense
    