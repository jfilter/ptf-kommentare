#!/bin/sh
set -e
set -x

out_folder=/mnt/data2/ptf/co_exp/lemma_nostop
# out_folder=/mnt/data2/ptf/co_exp/orig_nostop
# out_folder=/mnt/data2/ptf/co_exp/orig
# out_folder=/mnt/data2/ptf/co_exp/lemma

mkdir -p $out_folder

if ! [ $1 = "evaluate" ]; then

# Download and install word2vecf
if [ ! -f word2vecf ]; then
    scripts/install_word2vecf.sh
fi


CORPUS=lemma_nostop.txt
# CORPUS=orig_nostop.txt
# CORPUS=orig.txt
# CORPUS=lemma.txt
cp /mnt/data2/ptf/co_exp/$CORPUS $out_folder
#scripts/clean_corpus.sh $CORPUS > $CORPUS.clean
# mv $CORPUS $out_folder/$CORPUS

# Create two $out_folder collections of word-context pairs:

# A) Window size 2 with "clean" subsampling
mkdir -p $out_folder/w2_sub
python hyperwords/corpus2pairs.py --win 2 --sub 1e-5 $out_folder/${CORPUS} > $out_folder/w2_sub/pairs
scripts/pairs2counts.sh $out_folder/w2_sub/pairs > $out_folder/w2_sub/counts
python hyperwords/counts2vocab.py $out_folder/w2_sub/counts

# B) Window size 5 with dynamic contexts and "dirty" subsampling
mkdir -p $out_folder/w5_dyn_sub_del
python hyperwords/corpus2pairs.py --win 5 --dyn --sub 1e-5 --del $out_folder/${CORPUS} > $out_folder/w5_dyn_sub_del/pairs
scripts/pairs2counts.sh $out_folder/w5_dyn_sub_del/pairs > $out_folder/w5_dyn_sub_del/counts
python hyperwords/counts2vocab.py $out_folder/w5_dyn_sub_del/counts

# Calculate PMI matrices for each collection of pairs
python hyperwords/counts2pmi.py --cds 0.75 $out_folder/w2_sub/counts $out_folder/w2_sub/pmi
python hyperwords/counts2pmi.py --cds 0.75 $out_folder/w5_dyn_sub_del/counts $out_folder/w5_dyn_sub_del/pmi


# Create embeddings with SVD
python hyperwords/pmi2svd.py --dim 500 --neg 5 $out_folder/w2_sub/pmi $out_folder/w2_sub/svd
cp $out_folder/w2_sub/pmi.words.vocab $out_folder/w2_sub/svd.words.vocab
cp $out_folder/w2_sub/pmi.contexts.vocab $out_folder/w2_sub/svd.contexts.vocab
python hyperwords/pmi2svd.py --dim 500 --neg 5 $out_folder/w5_dyn_sub_del/pmi $out_folder/w5_dyn_sub_del/svd
cp $out_folder/w5_dyn_sub_del/pmi.words.vocab $out_folder/w5_dyn_sub_del/svd.words.vocab
cp $out_folder/w5_dyn_sub_del/pmi.contexts.vocab $out_folder/w5_dyn_sub_del/svd.contexts.vocab


# Create embeddings with SGNS (A). Commands 2-5 are necessary for loading the vectors with embeddings.py
word2vecf/word2vecf -train $out_folder/w2_sub/pairs -pow 0.75 -cvocab $out_folder/w2_sub/counts.contexts.vocab -wvocab $out_folder/w2_sub/counts.words.vocab -dumpcv $out_folder/w2_sub/sgns.contexts -output $out_folder/w2_sub/sgns.words -threads 4 -negative 15 -size 500;
python hyperwords/text2numpy.py $out_folder/w2_sub/sgns.words
rm $out_folder/w2_sub/sgns.words
python hyperwords/text2numpy.py $out_folder/w2_sub/sgns.contexts
rm $out_folder/w2_sub/sgns.contexts

# Create embeddings with SGNS (B). Commands 2-5 are necessary for loading the vectors with embeddings.py
word2vecf/word2vecf -train $out_folder/w5_dyn_sub_del/pairs -pow 0.75 -cvocab $out_folder/w5_dyn_sub_del/counts.contexts.vocab -wvocab $out_folder/w5_dyn_sub_del/counts.words.vocab -dumpcv $out_folder/w5_dyn_sub_del/sgns.contexts -output $out_folder/w5_dyn_sub_del/sgns.words -threads 4 -negative 15 -size 500;
python hyperwords/text2numpy.py $out_folder/w5_dyn_sub_del/sgns.words
rm $out_folder/w5_dyn_sub_del/sgns.words
python hyperwords/text2numpy.py $out_folder/w5_dyn_sub_del/sgns.contexts
rm $out_folder/w5_dyn_sub_del/sgns.contexts

fi

# Evaluate on Word Similarity
echo
echo "WS353 Results"
echo "-------------"

export PYTHONWARNINGS="ignore"
set +x

python hyperwords/ws_eval.py --neg 5 PPMI $out_folder/w2_sub/pmi testsets/de/ws/schm280.txt
python hyperwords/ws_eval.py --eig 0.5 SVD $out_folder/w2_sub/svd testsets/de/ws/schm280.txt
#python hyperwords/ws_eval.py --w+c SGNS $out_folder/w2_sub/sgns testsets/ws/ws353.txt

python hyperwords/ws_eval.py --neg 5 PPMI $out_folder/w5_dyn_sub_del/pmi testsets/de/ws/schm280.txt
python hyperwords/ws_eval.py --eig 0.5 SVD $out_folder/w5_dyn_sub_del/svd testsets/de/ws/schm280.txt
#python hyperwords/ws_eval.py --w+c SGNS $out_folder/w5_dyn_sub_del/sgns testsets/ws/ws353.txt


# Evaluate on Analogies
echo
echo "Google Analogy Results"
echo "----------------------"

python hyperwords/analogy_eval.py PPMI $out_folder/w2_sub/pmi testsets/de/analogy/google.txt
python hyperwords/analogy_eval.py --eig 0 SVD $out_folder/w2_sub/svd testsets/de/analogy/google.txt
#python hyperwords/analogy_eval.py SGNS $out_folder/w2_sub/sgns testsets/de/analogy/google.txt

python hyperwords/analogy_eval.py PPMI $out_folder/w5_dyn_sub_del/pmi testsets/de/analogy/google.txt
python hyperwords/analogy_eval.py --eig 0 SVD $out_folder/w5_dyn_sub_del/svd testsets/de/analogy/google.txt
#python hyperwords/analogy_eval.py SGNS $out_folder/w5_dyn_sub_del/sgns testsets/de/analogy/google.txt

echo
echo "Parasem"
echo "----------------------"

python hyperwords/analogy_eval.py PPMI $out_folder/w2_sub/pmi testsets/de/analogy/parasem.txt
python hyperwords/analogy_eval.py --eig 0 SVD $out_folder/w2_sub/svd testsets/de/analogy/parasem.txt
#python hyperwords/analogy_eval.py SGNS $out_folder/w2_sub/sgns testsets/de/analogy/google.txt

python hyperwords/analogy_eval.py PPMI $out_folder/w5_dyn_sub_del/pmi testsets/de/analogy/parasem.txt
python hyperwords/analogy_eval.py --eig 0 SVD $out_folder/w5_dyn_sub_del/svd testsets/de/analogy/parasem.txt
#python hyperwords/analogy_eval.py SGNS $out_folder/w5_dyn_sub_del/sgns testsets/de/analogy/google.txt
