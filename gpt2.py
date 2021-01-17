import gpt_2_simple as gpt2
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf 
import os
import argparse
import sys

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

def generate_text(run_name, questions, sess, outfn):
    if sess is None:
        sess = gpt2.start_tf_sess()
        gpt2.load_gpt2(sess, run_name=run_name)
    num_tot = len(questions)
    direc, fn = os.path.split(outfn)
    ckptpath = os.path.join(direc, "ckpt.txt")
    if os.path.exists(ckptpath):
        ckpt = int(open(ckptpath).read())
        questions = questions[ckpt:]
    else:
        ckpt = 0
    print("Start Generation ... ", file = sys.stdout)
    for i in range(len(questions)):
        text = gpt2.generate(sess,
                          run_name=run_name,
                          length=500,
                          truncate = "\n",
                          temperature=0.7,
                          prefix=questions[i],
                          nsamples=1,
                          batch_size=1,
                          include_prefix=False,
                          return_as_list=True)[0]
      
        text = text.replace(questions[i], "")
        if os.path.exists(outfn):
            with open(outfn, "a") as f:
                f.write(text+"\n")
        else:
            with open(outfn, "w") as f:
                f.write(text+ "\n")
        ckpt += 1
        
        with open(os.path.join(ckptpath), "w") as f:
            f.write(str(ckpt))
        if i == 0:
            print(text, file = sys.stdout) # used to debug early
        if i %50 ==0:
            print(f"{ckpt+i+1}/{num_tot} of answers have been generated!", file = sys.stdout)
            tf.reset_default_graph()
            sess.close()
            sess = gpt2.start_tf_sess()
            gpt2.load_gpt2(sess, run_name=run_name)
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

def eval_gpt2(testset, party, with_q, is_read, sess, outfn):
    run_name = get_run_name(party, with_q)
    if hasattr(testset, 'question'):
        questions = testset.question.tolist() 
    generated = generate_text(run_name, questions, sess, outfn)

    if is_read:
        print_comparison(questions, testset.answer, generated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default = "./data/presidency_project/newsconference/rep_train.csv")
    parser.add_argument("--test", default = "./data/presidency_project/newsconference/rep_test.csv")
    parser.add_argument("--party", default= "Republican")
    parser.add_argument("--withq", default=True, type=bool)
    parser.add_argument("--only_ft", default = False, type=bool)
    parser.add_argument("--only_gen", default = True, type=bool)
    parser.add_argument("--outfn", default = "./generation/gpt2/rep/rep_test_predict.txt")
    args = parser.parse_args()
    
    direc, fn = os.path.split(args.outfn)
    if not os.path.exists(direc):
        os.makedirs(direc)

    # Init Pre-Trained Model
    model_name="124M"
    if not os.path.isdir(os.path.join("models", model_name)):
        print(f"Downloading {model_name} model ...")
        gpt2.download_gpt2(model_name=model_name) 
    
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    party = args.party
    with_q = args.withq
    is_read =False    
    if args.only_ft:
        finetune_gpt2(train_df, party, with_q)
        sys.stdout.write("Finished Finetuning! Exit Program.")
        exit()
    if args.only_gen:
        sys.stdout.write("Only generation mode, starting to generate...")
        sess = None
        eval_gpt2(test_df, party, with_q, is_read, sess, args.outfn)
        sys.stdout.write("Finished Generatiion! Exit Program.")
    else:
        sess = finetune_gpt2(train_df, party, with_q)
        eval_gpt2(test_df, party, with_q, is_read, sess, args.outfn)
