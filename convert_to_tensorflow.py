from keras.models import Model
from keras.layers import *
import os
import tensorflow as tf

from keras.models import load_model

def keras_to_tensorflow(keras_model, output_dir, model_name,out_prefix="output_", log_tensorboard=True):

    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)

    out_nodes = []

    for i in range(len(keras_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(keras_model.output[i], out_prefix + str(i + 1))

    sess = K.get_session()

    from tensorflow.python.framework import graph_util, graph_io

    init_graph = sess.graph.as_graph_def()

    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)

    graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)

    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard

        import_pb_to_tensorboard.import_to_tensorboard(
            os.path.join(output_dir, model_name),
            output_dir)
            
            
# Load keras model
model = load_model("/home/ajay/PycharmProjects/image_classification/tool_analysing.model")
#convert keras model to tensorflow
keras_to_tensorflow(model,output_dir="/home/ajay/PycharmProjects/image_classification",model_name="tool_classification.pb")
print("MODEL SAVED")
