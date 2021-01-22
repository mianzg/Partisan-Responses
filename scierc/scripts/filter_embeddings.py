#!/usr/bin/env python

import sys
import json


if __name__ == "__main__":
  if len(sys.argv) < 5:
    sys.exit("Usage: {} <embeddings> <filtered_embeddings> <json1> <json2> ...".format(sys.argv[0]))

  words_to_keep = set()
  for json_filename in sys.argv[4:]:
    with open(json_filename) as json_file:
      for line in json_file.readlines():
        for sentence in json.loads(line)["sentences"]:
          words_to_keep.update(sentence)

  print("Found {} words in {} dataset(s).".format(len(words_to_keep), len(sys.argv) - 4))
  total_lines = 0
  kept_lines = 0
  out_filename = sys.argv[2]
  with open(sys.argv[1]) as in_file:
    with open(out_filename, "w") as out_file:
      for line in in_file.readlines():
        total_lines += 1
        splits = line.split()
        word = " ".join(splits[:-300]).strip()#line.split()[0]
        if word in words_to_keep:
          kept_lines += 1
          new_line = "".join([i+"\t" for i in [word] + splits[-300:-1]] + splits[-1:])+"\n"
          out_file.write(new_line)
  with open(sys.argv[3]) as f:
    old = f.readlines()
    new=""
    for line in old:
      splits = line.split()
      word = " ".join(splits[:-300]).strip()#line.split()[0]
      new_line = "".join([i+"\t" for i in [word] + splits[-300:-1]] + splits[-1:])+"\n"
      new+=new_line
  with open(sys.argv[3], "w") as f:
    f.write(new)
    
    
  print("Kept {} out of {} lines.".format(kept_lines, total_lines))
  print("Wrote result to {}.".format(out_filename))
