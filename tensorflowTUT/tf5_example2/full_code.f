
    --- marker ---
    
    <py>
    # View more python tutorial on my Youtube and Youku channel!!!

    # Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
    # Youku video tutorial: http://i.youku.com/pythontutorial

    """
    Please note, this code is only for python 3+. If you are using 
    python 2+, please modify the code accordingly.
    """
    # from __future__ import print_function
    import tensorflow as tf
    import numpy as np

    vm.debug=[] # '00-11-22-33-44-55-66-77-88-12-13-99-2222'
    
    # create data 
    # X - y 之間是簡單的線性關係。就是要看看 tf 能不能自己看出這個關係。
    x_data = np.random.rand(100).astype(np.float32) # 100 個
    y_data = x_data*0.1 + 0.3

    ### create tensorflow structure start ###
    # Weights = tf.Variable(tf.random_uniform([1])) # 不給範圍也可以，範圍給很寬只是需要更多迭代而已。
    Weights = tf.Variable(tf.random_uniform([1], -100.0, 108.0))
    biases = tf.Variable(tf.zeros([1]))

    y = Weights*x_data + biases

    loss = tf.reduce_mean(tf.square(y-y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5) # 取出 optimizer instance 
    train = optimizer.minimize(loss)

    # 重複了，這行應該刪除
    # init = tf.initialize_all_variables()
    # prompt=55;(str(prompt) in debug and [ok(str(prompt)+'> ',locals())] or [None])[0] # breakpoint 55>
    
    ### create tensorflow structure end ###

    sess = tf.Session()

    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
        prompt=77;(str(prompt) in debug and [ok(str(prompt)+'> ',locals())] or [None])[0] # breakpoint 77>
    else:
        init = tf.global_variables_initializer()
        prompt=88;(str(prompt) in debug and [ok(str(prompt)+'> ',locals())] or [None])[0] # breakpoint 88>

    # 取得 initializer 提供給 session，到這裡整個布置好了。 
    sess.run(init)
    prompt=99;(str(prompt) in debug and [ok(str(prompt)+'> ',locals())] or [None])[0] # breakpoint 99>

    for step in range(201):
        sess.run(train)
        print(step, sess.run(Weights), sess.run(biases))
        prompt=13;(str(prompt) in debug and [ok(str(prompt)+'> ',locals())] or [None])[0] # breakpoint 13>
            
    prompt=2222;(str(prompt) in debug and [ok(str(prompt)+'> ',locals())] or [None])[0] # breakpoint
    outport(locals())
    </py>
    

    stop 
    
    [x] 我從 outport 出來的 values 看出這個 sess 並沒有 closed 掉。所以可以
        繼續 train 出更接近的結果。問題是： 為何 tensorflow6_session.f 的 sess
        被 closed? ──> 因為它自己程式重複用了 sess 第二次關的！ ha ha haha
        [x] 證實 <== Yes 第二個改名叫 sess2 就好了。
    
    感覺，這裡只有一顆 neuron, 即 y = Weights*x_data + biases 者。
    所謂 tf.Variable() 與一般 programming language 的 variable 意思不一樣。
    tf.Variable() 產生的東西就是 tf 的 optimizer 要去調整，或要去鍛煉的東西。
    本例題中 variable 只有 Weights biases 這兩個 tf 是知道的。
    
    選定 optimizer 之後，就要告訴它目標是什麼。本題的目標是 loss 要小：
        train = optimizer.minimize(loss)
    
    有了 train 之後，每執行一次 train 就會調整所有的 variable (Weights biases)
    企圖使 loss 達到最小。
        
    11> 
    cls r@ :> ['x_data'] dup . cr :> shape tib. \ ==> (100,) (<class 'tuple'>)
    [ 0.69772786  0.06008626  0.66457355  0.27102727  0.17405169  0.70356649
      0.48676038  0.35575387  0.28897026  0.75155598  0.02669112  0.33670753
      0.7240954   0.51244891  0.66051161  0.06531314  0.94651216  0.74972343
      0.5902546   0.79755801  0.80254471  0.9081853   0.28317916  0.9042176
      0.32330152  0.3029542   0.24547592  0.67951858  0.71886832  0.25154191
      0.33687904  0.63434571  0.29191211  0.9198727   0.93803066  0.85245442
      0.85945159  0.9976213   0.92373735  0.90189511  0.71578312  0.68464524
      0.28605539  0.44959086  0.92583096  0.04860092  0.54973543  0.15894704
      0.5882988   0.3349202   0.46179298  0.163525    0.59511358  0.46161065
      0.64557439  0.80705357  0.20058864  0.15841039  0.60752767  0.5181908
      0.40163603  0.16276675  0.51862073  0.42048669  0.28413963  0.68119818
      0.10102321  0.39323381  0.34893927  0.15438248  0.76294035  0.21034595
      0.38149571  0.609025    0.45675954  0.69034302  0.34971884  0.03074021
      0.80500698  0.93165046  0.06419659  0.33019364  0.00535272  0.51986486
      0.62394559  0.47240952  0.70960855  0.968261    0.69947827  0.53928137
      0.4722791   0.4860293   0.99662513  0.80307764  0.74991983  0.26170972
      0.54336154  0.5194847   0.87675864  0.1055611 ]
    11> r@ :> ['y_data'] dup . cr :> shape tib.
    [ 0.36977279  0.30600864  0.36645737  0.32710275  0.31740519  0.37035668
      0.34867606  0.3355754   0.32889703  0.37515563  0.30266914  0.33367077
      0.37240955  0.3512449   0.36605117  0.30653134  0.39465123  0.37497234
      0.35902548  0.37975582  0.38025448  0.39081854  0.32831794  0.39042178
      0.33233017  0.33029544  0.32454759  0.36795187  0.37188685  0.32515422
      0.3336879   0.36343458  0.32919121  0.39198729  0.39380309  0.38524544
      0.38594517  0.39976215  0.39237374  0.39018953  0.37157834  0.36846453
      0.32860556  0.34495911  0.3925831   0.30486012  0.35497355  0.31589472
      0.35882989  0.33349204  0.34617931  0.31635252  0.35951138  0.34616107
      0.36455745  0.38070536  0.32005888  0.31584105  0.36075279  0.3518191
      0.34016362  0.3162767   0.35186207  0.34204867  0.32841396  0.36811984
      0.31010234  0.3393234   0.33489394  0.31543827  0.37629405  0.32103461
      0.33814958  0.36090252  0.34567598  0.36903432  0.3349719   0.30307403
      0.3805007   0.39316505  0.30641967  0.33301938  0.30053529  0.3519865
      0.36239457  0.34724095  0.37096086  0.39682612  0.36994785  0.35392815
      0.34722793  0.34860295  0.39966252  0.38030779  0.37499201  0.32617098
      0.35433617  0.35194847  0.38767588  0.31055611]
    r@ :> ['y_data'] dup . cr :> shape tib. \ ==> (100,) (<class 'tuple'>)
    11> r@ :> keys() .
    dict_keys(['np', 'tf', 'y_data', 'x_data'])11> r@ :> ['y_data'] type . cr
    <class 'numpy.ndarray'>
    11> r@ :> ['y_data'][2] type . cr
    <class 'numpy.float32'>
    11> r@ :> ['x_data'][2] type . cr
    <class 'numpy.float32'>
    11>    
    --
    
    22>
    'Weights' :  < tf.Variable 'Variable:0' shape = (1, )dtype = float32_ref >
    'biases' :  < tf.Variable 'Variable_1:0' shape = (1, )dtype = float32_ref > ,
    
    Weights, biases 這兩個 nuron cell 的 elements 在此應該是 class 
    真的跟什麼 instance 運算之後才會有實體
    
    dup :> [3] :> keys() tib. \ ==> dict_keys(['prompt', 'np', 'tf', 'y_data', 'x_data', 'biases', 'Weights']) (<class 'dict_keys'>)
    dup :> [3] :> ['biases'] type tib.   \ ==> <class 'tensorflow.python.ops.variables.Variable'> (<class 'type'>)
    dup :> [3] :> ['Weights'] type tib.  \ ==> <class 'tensorflow.python.ops.variables.Variable'> (<class 'type'>)
    dup :> [3] :> ['Weights'] dir tib.   \ ==> ['SaveSliceInfo', '_AsTensor', '_OverloadAllOperators', '_OverloadOperator', '_TensorConversionFunction', '__abs__', '__add__', '__and__', '__array_priority__', '__class__', '__delattr__', '__dict__', '__dir__', '__div__', '__doc__', '__eq__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__invert__', '__iter__', '__le__', '__lt__', '__matmul__', '__mod__', '__module__', '__mul__', '__ne__', '__neg__', '__new__', '__or__', '__pow__', '__radd__', '__rand__', '__rdiv__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rmatmul__', '__rmod__', '__rmul__', '__ror__', '__rpow__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__weakref__', '__xor__', '_as_graph_element', '_caching_device', '_get_save_slice_info', '_init_from_args', '_init_from_proto', '_initial_value', '_initializer_op', '_ref', '_save_slice_info', '_set_save_slice_info', '_snapshot', '_variable', 'assign', 'assign_add', 'assign_sub', 'count_up_to', 'device', 'dtype', 'eval', 'from_proto', 'get_shape', 'graph', 'initial_value', 'initialized_value', 'initializer', 'load', 'name', 'op', 'read_value', 'scatter_sub', 'set_shape', 'shape', 'to_proto', 'value'] (<class 'list'>)
    dup :> [3] :> ['Weights'].shape tib. \ ==> (1,) (<class 'tensorflow.python.framework.tensor_shape.TensorShape'>)
        
    --
    
    33>
    dup :> [3] :> keys() tib. \ ==> dict_keys(['prompt', 'np', 'tf', 'y_data', 'x_data', 'biases', 'Weights', 'y']) (<class 'dict_keys'>)
    dup :> [3] :> ['y'] tib. \ ==> Tensor("add:0", shape=(100,), dtype=float32) 
                                   (<class 'tensorflow.python.framework.ops.Tensor'>)
    dup :> [3] :> ['y'] :> name tib. \ ==> add:0 (<class 'str'>)
    dup :> [3] :> ['y'] :> shape tib. \ ==> (100,) (<class 'tensorflow.python.framework.tensor_shape.TensorShape'>)
    dup :> [3] :> ['y'] type tib. \ ==> <class 'tensorflow.python.framework.ops.Tensor'> (<class 'type'>)
    dup :> [3] :> ['y'] dir tib. \ ==> ['OVERLOADABLE_OPERATORS', '__abs__', '__add__', '__and__', '__array_priority__', '__bool__', '__class__', '__delattr__', '__dict__', '__dir__', '__div__', '__doc__', '__eq__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__invert__', '__iter__', '__le__', '__lt__', '__matmul__', '__mod__', '__module__', '__mul__', '__ne__', '__neg__', '__new__', '__nonzero__', '__or__', '__pow__', '__radd__', '__rand__', '__rdiv__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rmatmul__', '__rmod__', '__rmul__', '__ror__', '__rpow__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__weakref__', '__xor__', '_add_consumer', '_as_node_def_input', '_consumers', '_dtype', '_handle_dtype', '_handle_shape', '_op', '_override_operator', '_shape', '_shape_as_list', '_value_index', 'consumers', 'device', 'dtype', 'eval', 'get_shape', 'graph', 'name', 'op', 'set_shape', 'shape', 'value_index'] (<class 'list'>)
    
    dup :> [3] :> ['y'] py> str(pop())  tib. \ ==> Tensor("add:0", shape=(100,), dtype=float32) (<class 'str'>)
    dup :> [3] :> ['y'] py> repr(pop()) tib. \ ==> <tf.Tensor 'add:0' shape=(100,) dtype=float32> (<class 'str'>)

    ──
    44>
    dup :> [3] :> keys() tib. 
        \ ==> 
        dict_keys(['prompt', 'np', 'tf', 
        'y_data', 'x_data', 'biases', 'Weights', 'y', 
        'train', 'optimizer', 'loss']) (<class 'dict_keys'>)
    dup :> [3] :> ['loss'] tib. 
        \ ==> 
        Tensor("Mean:0", shape=(), dtype=float32) 
        (<class 'tensorflow.python.framework.ops.Tensor'>)
        # 這裡會用就好，別鑽牛角尖兒。loss 就是看 y 跟 y_data 的差距，裡面有相減，有 square
        # 還有 mean 這就明白了，將來也是照抄。loss 是一個 scalar。
    dup :> [3] :> ['optimizer'] tib. 
        \ ==> 
        <tensorflow.python.training.gradient_descent.GradientDescentOptimizer object at 0x000002107CE4CB38> 
        (<class 'tensorflow.python.training.gradient_descent.GradientDescentOptimizer'>)
        # optimizer 有很多種，祭出某一種。在此還是個 object 
    dup :> [3] :> ['train'] tib. 
        \ ==> 
        name: "GradientDescent"
        op: "NoOp"
        input: "^GradientDescent/update_Variable/ApplyGradientDescent"
        input: "^GradientDescent/update_Variable_1/ApplyGradientDescent"
        (<class 'tensorflow.python.framework.ops.Operation'>)
        # 告訴 optimizer 要怎麼做之後，就是 train 了。
        dup :> [3] :> ['train'] type tib. 看看 train 啥 type 
            \ ==> <class 'tensorflow.python.framework.ops.Operation'> (<class 'type'>)
        dup :> [3] :> ['train'] dir tib.  看看 train 有啥功能：
            \ ==> ['_InputList', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', 
            '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', 
            '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', 
            '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', 
            '__subclasshook__', '__weakref__', '_add_control_input', '_add_control_inputs', 
            '_add_input', '_control_flow_context', '_control_inputs', '_get_control_flow_context', 
            '_graph', '_id', '_id_value', '_input_dtypes', '_input_types', '_inputs', '_node_def', 
            '_op_def', '_original_op', '_output_types', '_outputs', '_recompute_node_def', 
            '_set_control_flow_context', '_set_device', '_traceback', '_update_input', 
            'colocation_groups', 'control_inputs', 'device', 'get_attr', 'graph', 'inputs', 'name', 
            'node_def', 'op_def', 'outputs', 'run', 'traceback', 'type', 'values'] (<class 'list'>)
    --
    66> 
    看看 session 是啥東西。
    locals :> [3] :> keys() tib. 
        \ ==> dict_keys(['prompt', 'np', 'tf', 'y_data', 'x_data', 'biases', 'Weights', 
        'y', 'train', 'optimizer', 'loss', 'init', 'sess']) (<class 'dict_keys'>)
    locals :> [3] :> ['sess'] tib. 
        \ ==> <tensorflow.python.client.session.Session object at 0x00000210013E7E48> 
        (<class 'tensorflow.python.client.session.Session'>)
        有 instance 地址的，以上僅 optimizer 也有。
    locals :> [3] :> ['sess'] type tib. 
        \ ==> 
         <class 'tensorflow.python.client.session.Session'> (<class 'type'>)
    locals :> [3] :> ['sess'] dir tib. 
        \ ==> ['_DEAD_HANDLES_THRESHOLD', '_NODEDEF_NAME_RE', '__class__', '__del__', 
        '__delattr__', '__dict__', '__dir__', '__doc__', '__enter__', '__eq__', '__exit__', 
        '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', 
        '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', 
        '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', 
        '__subclasshook__', '__weakref__', '_add_shapes', '_closed', '_config', '_current_version', 
        '_dead_handles', '_default_graph_context_manager', '_default_session_context_manager', 
        '_delete_lock', '_do_call', '_do_run', '_extend_graph', '_extend_lock', '_graph', 
        '_opened', '_register_dead_handle', '_run', '_session', '_target', '_update_with_movers', 
        'as_default', 'close', 'graph', 'graph_def', 'partial_run', 'partial_run_setup', 
        'reset', 'run', 'sess_str'] (<class 'list'>)
    --
    88> 
    locals :> [3] :> ['init'] tib. 
        \ ==> name: "init_1"
                op: "NoOp"
                input: "^Variable/Assign"
                input: "^Variable_1/Assign"
                 (<class 'tensorflow.python.framework.ops.Operation'>)
    --
    99>
    sess run 過 init 之後
    locals :> [3] :> keys() tib. 
        \ ==> dict_keys(['prompt', 'np', 'tf', 'y_data', 'x_data', 'biases', 'Weights', 
            'y', 'train', 'optimizer', 'loss', 'init', 'sess']) (<class 'dict_keys'>)
    --
    2222> 
    locals :> [3].keys() . cr
        dict_keys(['biases', 'Weights', 'y_data', 'x_data', 'prompt', 'np', 
        'tf', 'step', 'init', 'sess', 'train', 'optimizer', 'loss', 'y'])
    2222> locals :> [3]['y'] constant y
    2222> locals :> [3]['sess'] constant sess
    2222> py> v('sess').run(v('y')) .
    [ 0.33174449  0.30135444  0.37779647  0.33364162  0.3379555   0.31519002
      0.38864312  0.37099391  0.35468873  0.32427108  0.34224457  0.38378558
      ...snip... 0.39441976  0.39256316  0.30014914  0.33264452]2222>
      
    py> v('locals')[3]['loss'] tib. 
        \ ==> Tensor("Mean:0", shape=(), dtype=float32) (<class 'tensorflow.python.framework.ops.Tensor'>)
 
    py> v('locals')[3]['sess'].run(v('locals')[3]['loss']) tib. 
        \ ==> 8.215650382226158e-15 (<class 'numpy.float32'>)

    py> v('locals')[3]['sess'].run(v('locals')[3]['loss']) tib. 
        
        