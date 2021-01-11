import gpt_2_simple as gpt2
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf 
import os
import argparse


def write_gpt2_input(df, save_path):
    speeches = []
    for i in range(df.shape[0]):
      answer = df.answer.iloc[i].replace("\n", " ") +"\n"
      speeches.append(answer)
    if not os.path.exists(save_path):
      with open(save_path, "w") as f:
        f.writelines(speeches)

def run_gpt2(dataset, run_name):
    sess = gpt2.start_tf_sess()
    gpt2.finetune(sess,
              dataset=dataset,
              steps = 1000,
              restore_from="fresh",
              run_name=run_name,
              print_every=50,
              sample_every=200,
              save_every=250)
    return sess

def generate_text(run_name, questions, sess):
    if sess is None:
        sess = gpt2.start_tf_sess()
        gpt2.load_gpt2(sess, run_name=run_name)
    generated = []
    for i in range(len(questions)):
      text = gpt2.generate(sess,
                          run_name=run_name,
                          length=100,
                          temperature=0.7,
                          prefix=questions[i],
                          nsamples=1,
                          batch_size=1,
                          include_prefix=False,
                          return_as_list=True)[0]
      generated.append(text)
    return generated

def print_comparison(questions, ground_truth, generated):
  for i in range(len(questions)):
    print(questions[i])
    print("Original Answer:\n")
    print(ground_truth.iloc[i][:1000])
    print("Generated Answer:\n")
    print(generated[i].replace(questions[i], ""))
    print("="*20)

def get_run_name(party, with_q):
  if with_q:
    run_name = party.lower()+"_"+"with_questions"
  else:
    run_name = party.lower()+"_"+"wo_questions"
  return run_name

def finetune_gpt2(df, party, with_q):
    """
    party:str, Republican or Democrat
    """
    direc = "./data/presidency_project/newsconference"
    run_name = get_run_name(party, with_q)
    filepath = os.path.join(direc, "gpt2_"+run_name+".txt")
    write_gpt2_input(df, filepath)
    sess = run_gpt2(filepath, run_name)
    print("Please restart kernel to generate text")
    return sess 

def eval_gpt2(testset, party, with_q, is_read, sess):
    run_name = get_run_name(party, with_q)
    if hasattr(testset, 'question'):
        questions = testset.question 
    generated = generate_text(run_name, questions, sess)
    generated = [generated[i].replace(questions[i], "") for i in range(len(questions))]

    if is_read:
        print_comparison(questions, testset.answer, generated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default = "./data/presidency_project/newsconference/rep_train.csv")
    parser.add_argument("--test", default = "./data/presidency_project/newsconference/rep_test.csv")
    parser.add_argument("--party", default= "Republican")
    parser.add_argument("--withq", default=True, type=bool)
    #parser.add_argument("--only_ft", default = False, type=bool)
    #parser.add_argument("--only_gen", default = False)
    args = parser.parse_args()
    
    # Init Pre-Trained Model
    model_name="124M"
    if not os.path.isdir(os.path.join("models", model_name)):
        print(f"Downloading {model_name} model ...")
        gpt2.download_gpt2(model_name) 
    
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    party = args.party
    with_q = args.withq
    
    # if args.only_ft:
    #     finetune_gpt2(train_df, party, with_q)
    # if args.only_gen:
    #     eval_gpt2(train_df, party, with_q)
    sess = finetune_gpt2(train_df, party, with_q)
    eval_gpt(testset, party, with_q, is_read, sess)