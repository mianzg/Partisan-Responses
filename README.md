# Partisan-Responses
Project for the class "Introduction of Natural Language Processing (Fall 2020)"

## Dataset (in-progress)
- [Presidency Project](https://www.presidency.ucsb.edu/)
  - [News Conference](https://www.presidency.ucsb.edu/documents/app-categories/presidential/news-conferences)  
- [Gallup Public Opinion Questions categorized by topics](https://news.gallup.com/poll/trends.aspx#P)

## Usage
Clone the repository and Install Anaconda, then create a conda environment for this project and retrieve datasets.
```{bash}
conda create -n 
```
### Acquire Dataset
**RECOMMENDED**: You can directly download the data folder from [THIS LINK](), and put it under this repository. 

Otherwise, data come from various sources, and the following provides how I scrape and coarsely process them:
#### News Conference Data (Presidency Project)
It will either load or scrape news conference from the website, then split by democratic and replican. Under each party, the dataset is split into train, validation and test datasets. 
```
python dataset.py
```
Now under `data/presidency_project/newsconference/`, we should have `dem_train.csv`, `dem_val.csv`, `dem_test.csv`, `rep_train.csv`, `rep_val.csv` and `rep_test.csv`. These six files serve the start to further pre-process needed input files for the following models. The baseline model gpt-2 will directly use these files, and moreover, the above command also creates input files for vanilla GraphWriter under `data/presidency_project/newsconference/gwnaive/`
#### Gallup Questions
```
python Questions.py
```
So far, under the `data/` directory, we should have the following structure:
```
TODO
```

## Models
### Finetuned GPT-2 (Baseline)

### GraphWriter (naive)
```
# To Start
## Democratic
bsub -W 04:00 -N -R "rusage[mem=20480, ngpus_excl_p=2]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]" "python train.py -bsz 1 -t1size 1 -t2size 1 -t3size 1 -datadir ../data/gwnaive/Democratic/ -save ../output/gwnaive/Democratic/ -esz 256 -hsz 256"
## Republican
bsub -W 04:00 -N -R "rusage[mem=20480, ngpus_excl_p=2]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]" "python train.py -bsz 4 -t1size 4 -t2size 2 -t3size 1 -datadir ../data/gwnaive/Republican/ -save ../output/gwnaive/Republican/ -esz 256 -hsz 256"

# The training usually takes more than one job, then you need to follow up with last checkpoint you have, e.g, last checkpoint is 9th epoch (0-indexing)
bsub -W 04:00 -N -R "rusage[mem=20480, ngpus_excl_p=2]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]" "python train.py -bsz 1 -t1size 1 -t2size 1 -t3size 1 -datadir ../data/presidency_project/newsconference/gwnaive/Democratic/ -save ../output/gwnaive/Democratic/ -ckpt ../output/gwnaive/Democratic/8.vloss-3.715168.lr-0.1 -esz 256 -hsz 256"
```
### SCIERC
```
cd scierc
scripts/.sh
scripts/.sh
```
To train and validate on annotated data over Leonhard cluster
```
bsub -W 02:00 -N -R "rusage[mem=20480, ngpus_excl_p=4]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]" < run_scierc.sh 
```
Use the best trained NER-Relation model to automatically annotate over train, val and test to generate input data for GraphWriter
```
python scierc/generate_elmo.py -fn ./data/scierc_predict/dem/train.json -outfn ./data/scierc_predict/dem/train.hdf5
python scierc/generate_elmo.py -fn ./data/scierc_predict/dem/val.json -outfn ./data/scierc_predict/dem/dev.hdf5
python scierc/generate_elmo.py -fn ./data/scierc_predict/dem/test.json -outfn ./data/scierc_predict/dem/test.hdf5

python scierc/generate_elmo.py -fn ./data/scierc_predict/rep/train.json -outfn ./data/scierc_predict/rep/train.hdf5
python scierc/generate_elmo.py -fn ./data/scierc_predict/rep/val.json -outfn ./data/scierc_predict/rep/dev.hdf5
python scierc/generate_elmo.py -fn ./data/scierc_predict/rep/test.json -outfn ./data/scierc_predict/rep/test.hdf5

#republican train
bsub -n 8 -N -R "rusage[mem=2560]" 'python scierc/predict.py --expt partisan --lm_path_dev ./data/scierc_predict/rep/train.hdf5 --eval_path ./data/scierc_predict/rep/train.json --questions ./data/scierc_predict/rep/train_questions.txt --outfn ./data/gw_scierc/rep/preprocessed.train.tsv --batch_size 128 --ckpt ./data/gw_scierc/rep/train_ckpt.txt'

# republican validation
bsub -n 8 -N -R "rusage[mem=2560]" 'python scierc/predict.py --expt partisan --lm_path_dev ./data/scierc_predict/rep/dev.hdf5 --eval_path ./data/scierc_predict/rep/val.json --questions ./data/scierc_predict/rep/val_questions.txt --outfn ./data/gw_scierc/rep/preprocessed.val.tsv --batch_size 128 --ckpt ./data/gw_scierc/rep/val_ckpt.txt'

# republican test
bsub -n 8 -N -R "rusage[mem=2560]" 'python scierc/predict.py --expt partisan --lm_path_dev ./data/scierc_predict/rep/test.hdf5 --eval_path ./data/scierc_predict/rep/test.json --questions ./data/scierc_predict/rep/test_questions.txt --outfn ./data/gw_scierc/rep/preprocessed.test.tsv --batch_size 128 --ckpt ./data/gw_scierc/rep/test_ckpt.txt'

# democratic train
bsub -n 8 -N -R "rusage[mem=5000]" 'python scierc/predict.py --expt partisan --lm_path_dev ./data/scierc_predict/dem/train.hdf5 --eval_path ./data/scierc_predict/dem/train.json --questions ./data/scierc_predict/dem/train_questions.txt --outfn ./data/gw_scierc/dem/preprocessed.train.tsv --batch_size 128 --ckpt ./data/gw_scierc/dem/train_ckpt.txt'

# democratic validation
bsub -n 8 -N -R "rusage[mem=2560]" 'python scierc/predict.py --expt partisan --lm_path_dev ./data/scierc_predict/dem/dev.hdf5 --eval_path ./data/scierc_predict/dem/val.json --questions ./data/scierc_predict/dem/val_questions.txt --outfn ./data/gw_scierc/dem/preprocessed.val.tsv --batch_size 128 --ckpt ./data/gw_scierc/dem/val_ckpt.txt'

# democratic test
bsub -n 8 -N -R "rusage[mem=2560]" 'python scierc/predict.py --expt partisan --lm_path_dev ./data/scierc_predict/dem/test.hdf5 --eval_path ./data/scierc_predict/dem/test.json --questions ./data/scierc_predict/dem/test_questions.txt --outfn ./data/gw_scierc/dem/preprocessed.test.tsv --batch_size 128 --ckpt ./data/gw_scierc/dem/test_ckpt.txt'
```

### GraphWriter (using scierc)
```
###### LR 0.01 Embedding size 256 Hidden size 256 ######
# republican
bsub -N -R "rusage[mem=20480, ngpus_excl_p=2]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]" "python ./GraphWriter/train.py -bsz 4 -t1size 4 -t2size 2 -t3size 1 -datadir './data/gw_scierc/rep/' -save './gw_scierc_models/rep_lr0.01_256/' -ent_type 'Actor Implementation Institution Achievement OtherPolitical' -esz 256 -hsz 256 -lr 0.01 -ckpt './gw_scierc_models/rep_lr0.01_256/15.vloss-3.703604.lr-0.01'"


bsub -W 04:00 -N -R "rusage[mem=20480, ngpus_excl_p=2]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]" "python ./GraphWriter/train.py -bsz 4 -t1size 4 -t2size 2 -t3size 1 -datadir './data/gw_scierc/rep/' -save './gw_scierc_models/rep/' -ent_type 'Actor Implementation Institution Achievement OtherPolitical'"

python ./GraphWriter/train.py -bsz 1 -t1size 1 -t2size 1 -t3size 1 -datadir './data/gw_scierc/dem/' -save './gw_scierc_models/dem_lr0.01_256/' -ent_type 'Actor Implementation Institution Achievement OtherPolitical' -esz 256 -hsz 256 -lr 0.01

```

## Important papers
Building Knowledge Graph
- [Opinion-aware Knowledge Graph for Political Ideology Detection](https://www.ijcai.org/Proceedings/2017/0510.pdf)
  - only half applies, since we do not have a background graph like ConceptNet
- [An Automatic Knowledge Graph Creation Framework fromNatural Language Text](https://pdfs.semanticscholar.org/eb1d/438e7aca8600cfd87d7b0ecfaf36f36f5c37.pdf)
- [Knowledge Graph Construction](https://hal.archives-ouvertes.fr/hal-02277063/document)
  - [Coreference Resolution](https://github.com/huggingface/neuralcoref)
  - [Open Information Extraction](https://demo.allennlp.org/open-information-extraction)
  - "To merge nodes, the TF-IDF overlap of the new node’s name is calculated with the existing graph node names, and the new node is merged into an existing node if theTF-IDF  is  higher  than  some  threshold."
  - [Graph Engine](https://networkx.github.io/)
  
Graph to text 
- [Enhancing Topic-to-Essay Generation with External Commonsense
Knowledge](https://www.aclweb.org/anthology/P19-1193.pdf)
- [Graph Writer](https://arxiv.org/pdf/1904.02342.pdf)
- [Graph Attention Networks](https://github.com/PetarV-/GAT)
