python -m tensorflow.python.tools.optimize_for_inference \
--input results/dnn-100-base/frozen_graph.pb \
--output results/dnn-100-base/graph_optimized.pb \
--input_names=x \
--output_names=sequential/dense_3/Softmax