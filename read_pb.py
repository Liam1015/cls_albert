# -*- coding:utf-8 -*-
import tensorflow as tf
import tokenization
import time
import numpy as np
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("vocab_file", "albert_tiny/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

pb_file_path = './model_pb/sentiment'



def process_input(inputs, max_seq_length):

    data = []
    if type(inputs) == str or type(inputs) == unicode:
        data.append(inputs)
    elif type(inputs) == list:
        data = data + inputs

    data_list = []
    for input in data:
        tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=True)
        tokens_text = tokenizer.tokenize(input)
        if len(tokens_text) > max_seq_length - 2:
            tokens_text = tokens_text[0:(max_seq_length - 2)]
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_text:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        label_ids = 0
        tuple_ele = ([input_ids], [input_mask], [segment_ids], [label_ids])
        data_list.append(tuple_ele)
    return data_list



with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ["serve"], pb_file_path)
    op_list = sess.graph.get_operations()
    # print(type(op_list))
    # for op in op_list:
    #     print(op.name)
    inputs = ['我草你妈', '我喜欢你', '相信你', '你他妈再说一句']
    # vocab_idx, idx_vocab = load_vocab(FLAGS.vocab_file)
    receive_lists = process_input(inputs, 128)

    input_ids = sess.graph.get_tensor_by_name('input_ids:0')
    input_mask = sess.graph.get_tensor_by_name('input_mask:0')
    segment_ids = sess.graph.get_tensor_by_name('segment_ids:0')
    label_ids = sess.graph.get_tensor_by_name('label_ids:0')
    print(input_ids)
    print(input_mask)
    print(segment_ids)
    print(label_ids)


    ENCODING_VECTOR_NODE = 'bert/pooler/dense/Tanh:0'
    op = sess.graph.get_tensor_by_name(ENCODING_VECTOR_NODE)
    print(op)
    print(op)
    results = []
    for input_ids_data, input_mask_data, segment_ids_data, label_ids_data in receive_lists:

        start = time.time()
        result = sess.run(op, feed_dict={input_ids: input_ids_data,
                                         input_mask: input_mask_data,
                                         segment_ids: segment_ids_data,
                                         label_ids: label_ids_data})


        print(type(result))
        print(result.size)
        print(result.shape)
        end = time.time()
        print("cost time is %f" % (end - start) + " seconds")
        results.append(result[0])
    print(len(results))
    results = np.array(results)
    print(results.shape)

