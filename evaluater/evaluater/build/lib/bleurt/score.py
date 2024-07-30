import itertools
from evaluater.bleurt import checkpoint as checkpoint_lib
from evaluater.bleurt import encoding
from evaluater.bleurt.lib import tokenization
import tensorflow.compat.v1 as tf

class BleurtScorer(object):
    def __init__(self, checkpoint=None, predict_fn=None):
        if not checkpoint:
            checkpoint = self._get_default_checkpoint()
        config = checkpoint_lib.read_bleurt_config(checkpoint)
        self.max_seq_length = config["max_seq_length"]
        vocab_file = config["vocab_file"]
        do_lower_case = config["do_lower_case"]
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
        if predict_fn:
            self.predict_fn = predict_fn
        else:
            self.chkpt_dir = checkpoint
            self.predict_fn = self._make_predict_fn_from_checkpoint(checkpoint)

    def _get_default_checkpoint(self):
        return "bleurt/test_checkpoint"

    def _make_eager_predict_fn_from_checkpoint(self, checkpoint):
        assert tf.executing_eagerly()
        imported = tf.saved_model.load_v2(checkpoint)
        bleurt_model_ops = imported.signatures["serving_default"]
        def _predict_fn(input_dict):
            input_ids=tf.constant(input_dict["input_ids"]);input_ids = tf.cast(input_ids, tf.int64)
            input_mask=tf.constant(input_dict["input_mask"]);input_mask = tf.cast(input_mask, tf.int64)
            segment_ids=tf.constant(input_dict["segment_ids"]);segment_ids = tf.cast(segment_ids, tf.int64)
            return bleurt_model_ops(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)["predictions"].numpy()
        return _predict_fn

    def _make_lazy_predict_fn_from_checkpoint(self, checkpoint):
        assert not tf.executing_eagerly()
        bleurt_graph = tf.Graph()
        with bleurt_graph.as_default():
            imported = tf.saved_model.load_v2(checkpoint)
            bleurt_model_ops = imported.signatures["serving_default"]
            init_op = tf.group(tf.global_variables_initializer(), tf.tables_initializer())
        def _predict_fn(input_dict):
            with tf.Session(graph=bleurt_graph) as session:
                session.run(init_op)
                bleurt_ops = bleurt_model_ops(input_ids=tf.constant(input_dict["input_ids"]),
                                              input_mask=tf.constant(input_dict["input_mask"]),
                                              segment_ids=tf.constant(input_dict["segment_ids"]))
                bleurt_out = session.run(bleurt_ops)
            return bleurt_out["predictions"]
        return _predict_fn

    def _make_predict_fn_from_checkpoint(self, checkpoint):
        if tf.executing_eagerly():
            return self._make_eager_predict_fn_from_checkpoint(checkpoint)
        else:
            return self._make_lazy_predict_fn_from_checkpoint(checkpoint)

    def score(self, references, candidates, batch_size=100):
        candidates, references = list(candidates), list(references)
        assert len(candidates) == len(references), "The number of candidate sentences must match the number of reference sentences."
        if not candidates:
            return []
        all_results = []
        for i in range(0, len(candidates), batch_size):
            batch_ref = references[i:i + batch_size]
            batch_cand = candidates[i:i + batch_size]
            input_ids, input_mask, segment_ids = encoding.encode_batch(batch_ref, batch_cand, self.tokenizer, self.max_seq_length)
            tf_input = {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids}
            predict_out = self.predict_fn(tf_input)
            batch_results = predict_out.tolist()
            all_results.extend(batch_results)
        assert len(all_results) == len(candidates), "Number of predictions does not match sentences: {} vs. {}".format(len(all_results), len(candidates))
        return all_results

def bleurt_eval(candidates, references):
    ref_buffer = []
    cand_buffer = []
    scores_buffer = []
    bleurt_checkpoint = "/data1/juseondo/bridge_inputs/evaluater/bleurt/test_checkpoint"
    scorer = BleurtScorer(bleurt_checkpoint)
    def _consume_buffer():
        scores = scorer.score(ref_buffer, cand_buffer, batch_size=100)
        del ref_buffer[:]
        del cand_buffer[:]
        scores_buffer.extend(scores)
    for ref_sentence, cand_sentence in zip(references, candidates):
        ref_buffer.append(ref_sentence.strip())
        cand_buffer.append(cand_sentence.strip())
        if len(ref_buffer) >= 100:
            _consume_buffer()
    if ref_buffer:
        _consume_buffer()
    return sum(scores_buffer)/len(scores_buffer)

# Example usage:
# candidates = ["candidate sentence 1", "candidate sentence 2", ...]
# references = ["reference sentence 1", "reference sentence 2", ...]
# scores = calculate_bleurt_scores(candidates, references)
# print(scores)
