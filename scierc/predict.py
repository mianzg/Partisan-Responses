#!/usr/bin/env python

import os
import re
import sys
sys.path.append(os.getcwd())
import time
import datetime
import random
import shutil
import json
import csv

import numpy as np
import tensorflow as tf
import argparse
from torch.utils.data import DataLoader, random_split, Dataset

from lsgn_data import LSGNData
from lsgn_evaluator import LSGNEvaluator
from srl_model import SRLModel
import util
import inference_utils
from input_utils import pad_batch_tensors
import operator
import srl_eval_utils

def copy_checkpoint(source, target):
  for ext in (".index", ".data-00000-of-00001"):
    shutil.copyfile(source + ext, target + ext)

class MySet(Dataset):
  def __init__(self, eval_examples, questions):
    self.inputs = eval_examples
    self.questions = questions 
  def __len__(self):
    return len(self.inputs)
  def __getitem__(self, idx):
    return {"inputs": self.inputs[idx], "question": self.questions[idx]}

def collate(b):
  inputs = []
  questions = []
  for i in b:
    inputs.append(i["inputs"])
    questions.append(i["question"])
  return inputs, questions

class Predictor(object):
    def __init__(self, config, ckpt, outfn):
        self.config = config
        self.eval_data = None
        self.ckpt_file = ckpt
        self.outfn = outfn
        if os.path.exists(ckpt):
            with open(ckpt, "r") as f:
                self.ckpt = int(f.readline())
        else:
            self.ckpt = 0
            with open(ckpt, "w") as f:
                f.write(str(self.ckpt))
    def _process_entity(self, sent_id, s, ents, ent_phrase, ent_order, ent_type):
        for e in ents:
            start, end = e[0], e[1]
            phrase = " ".join(s[start:end+1])
            if phrase in ent_phrase: # duplicate
                ent_order[(sent_id, start, end)] = ent_phrase.index(phrase)
            else:
                ent_phrase.append(phrase)
                ent_type.append("<{}>".format(e[2]))
                ent_order[(sent_id, start, end)] =  ent_phrase.index(phrase)
                assert ent_phrase.index(phrase) == len(ent_phrase) - 1
        return ent_phrase, ent_order, ent_type
    
    def _process_relation(self, sent_id, rels, ent_order, triples):
        num_rel = 0
        for rel in rels:
          try:
            head, tail = ent_order[(sent_id, rel[0], rel[1])], ent_order[(sent_id, rel[2], rel[3])]
            rel_id = self.config["relation_labels"].index(rel[4])
            triples.append('{} {} {}'.format(head, rel_id, tail)) #TODO
            num_rel += 1
          except KeyError:
            continue
        return triples, num_rel
    
    def _mask_sents(self, sentences, ent_phrase, decoded_predictions, num_rel_dict):
        flagged = set()
        parsed = []
        masked_text = ''
        
        for idx, s in enumerate(sentences):
            lastEnd = 0
            masked_s = ''
            ents = decoded_predictions["ner"][idx]
            ents = sorted(ents, key = lambda x: x[0])
            for e in ents:
                start, end, cat = e[0], e[1], e[2]
                masked_s += " ".join(s[lastEnd:start]) # everything up to this entity
                ent_id = ent_phrase.index(" ".join(s[start:end+1]))
                masked_s += " <{}_{}> ".format(cat, ent_id)
                lastEnd = end+1
            tmp = [x.split("_")[1][:-1] for x in masked_s.split() if x.startswith("<") and x.endswith(">")]
            for j in tmp:
                if j not in flagged:
                    parsed.append(j)
                    flagged.add(j)
            masked_text += masked_s
            num_rel = num_rel_dict[idx]
            parsed.append("len(triples) + len(ent_phrase) - " + str(num_rel)+ " if " + str(num_rel) + " < len(triples) else None")        
            parsed.append("len(ent_phrase)")
            parsed.append("-1")
        return masked_text, parsed

    # TODO: Split to multiple functions.
    def process_to_gw(self, sentences, decoded_predictions):
        # Init vars
        ent_phrase = []
        ent_type = []
        triples = []
        ent_order = {}
        num_rel_dict = {}
        for idx, s in enumerate(sentences):
            ents = decoded_predictions["ner"][idx]
            # Unique ents and their type
            ent_phrase, ent_order, ent_type = self._process_entity(idx, s, ents, ent_phrase, ent_order, ent_type)
            # All triples
            rels = decoded_predictions["rel"][idx]
            triples, num_rel = self._process_relation(idx, rels, ent_order, triples)
            num_rel_dict[idx] = num_rel 
        masked_text, parsed = self._mask_sents(sentences, ent_phrase, decoded_predictions, num_rel_dict)
        for j, x in enumerate(parsed):
            parsed[j] = eval(x)
        while None in parsed:
            parsed.remove(None)
        
        ent_phrase = " ; ".join(ent_phrase)
        ent_type = " ".join(ent_type)
        triples = " ; ".join(triples)
        parsed = " ".join([str(order) for order in parsed])
        return ent_phrase, ent_type, triples, masked_text, parsed

    def predict(self, session, data, predictions, loss, batch_size, 
                questions, official_stdout=False):
        
        # Define Dataset
        with open(self.config["eval_path"]) as f:
            examples = [json.loads(jsonline) for jsonline in f.readlines()][self.ckpt:]
        with open(questions, "r") as f:
            questions = f.readlines()
            num_tot = len(questions)
            questions = questions[self.ckpt:]
        print((len(questions), len(examples)))
        dataset = MySet(examples, questions)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=False, collate_fn=collate)
        predict_result = []
        count = 0
        for batch_data in loader:
            eval_examples, questions = batch_data
            count += 1
            if self.eval_data is None:
                self.eval_tensors, self.coref_eval_data = data.load_predict_data(eval_examples)
            for i, doc_tensors in enumerate(self.eval_tensors):
                feed_dict = dict(zip(
                    data.input_tensors,
                    [pad_batch_tensors(doc_tensors, tn) for tn in data.input_names + data.label_names]))
                predict_names = []
                for tn in data.predict_names:
                    if tn in predictions:
                        predict_names.append(tn)
                predict_tensors = [predictions[tn] for tn in predict_names] + [loss]
                predict_tensors = session.run(predict_tensors, feed_dict=feed_dict)
                predict_dict = dict(zip(predict_names + ["loss"], predict_tensors))

                doc_example = self.coref_eval_data[i]
                sentences = doc_example["sentences"]
                decoded_predictions = inference_utils.mtl_decode(
                    sentences, predict_dict, data.ner_labels_inv, data.rel_labels_inv,
                    self.config)
                ent_phrase, ent_type, triples, masked_text, parsed = self.process_to_gw(sentences, decoded_predictions)
                # WRITE TSV
                if triples != '' and ent_phrase != '':
                    row = [questions[i].replace("\n", "")] + [ent_phrase, ent_type, triples, masked_text, parsed]
                    if not os.path.exists(self.outfn):
                        with open(self.outfn, "w") as tsv:
                            tsvwriter = csv.writer(tsv, delimiter="\t")
                            tsvwriter.writerow(row)
                            self.ckpt += 1
                            with open(self.ckpt_file, "w") as f:
                                f.write(str(self.ckpt))
                    else:
                        with open(self.outfn, "a") as tsv:
                            tsvwriter = csv.writer(tsv, delimiter="\t")
                            tsvwriter.writerow(row)   
                            self.ckpt += 1 
                            with open(self.ckpt_file, "w") as f:
                                f.write(str(self.ckpt))
                else:
                    self.ckpt += 1
            print(f"{self.ckpt}/{num_tot} predicted", file=sys.stdout)
        return predict_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expt", default = 'partisan')
    parser.add_argument("--config", default = "./scierc/experiments.conf")
    parser.add_argument("--lm_path_dev", required=True, help="Path to elmo hdf5 file")
    parser.add_argument("--eval_path", required=True, help = "path to unprocessed json file")
    parser.add_argument("--questions", required=True, help = "path to corresponding questions file to json file")
    parser.add_argument("--outfn", required=True, help = 'path to output tsv file')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ckpt", default = "path to checkpoint file")

    args = parser.parse_args()
    direc, fn = os.path.split(args.outfn)
    if not os.path.exists(direc):
        os.makedirs(direc)
        print(f'Create direc: {direc}')

    util.set_gpus()
    name = args.expt
    print("Running experiment: {} (from command-line argument).".format(name))
    # Modify Config
    config = util.get_config(args.config)[name]
    config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))
    
    config["lm_path_dev"] = args.lm_path_dev
    config["eval_path"] = args.eval_path
    # Dynamic batch size.
    config["batch_size"] = -1
    config["max_tokens_per_batch"] = -1
  
    # Use dev lm, if provided.
    if config["lm_path"] and "lm_path_dev" in config and config["lm_path_dev"]:
        config["lm_path"] = config["lm_path_dev"]

    #util.print_config(config)
    data = LSGNData(config) #TODO: batch process?
    model = SRLModel(data, config)
    predictor = Predictor(config, args.ckpt, args.outfn)

    variables_to_restore = []
    for var in tf.global_variables():
        if "module/" not in var.name:
            variables_to_restore.append(var)
        else:
            print("Not restoring from checkpoint:", var.name)

    saver = tf.train.Saver(variables_to_restore)
    log_dir = config["log_dir"]
    assert not ("final" in name)  # Make sure we don't override a finalized checkpoint.

    checkpoint_pattern = re.compile(".*model.ckpt-([0-9]*)\Z")
    
    with tf.Session() as session:
        checkpoint_path = os.path.join(log_dir, "model.max.ckpt")
        print("Evaluating {}".format(checkpoint_path))
        tf.global_variables_initializer().run()
        saver.restore(session, checkpoint_path)
        print("Start predicting ...")
        predict_result = predictor.predict(session, data, model.predictions, 
                                           model.loss, args.batch_size, args.questions)
        
        




