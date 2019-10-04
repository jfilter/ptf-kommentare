#!/bin/bash
# set -e
set -x

out_folder=/mnt/data2/ptf/co_exp/lemma_nostop_hyper
# out_folder=/mnt/data2/ptf/co_exp/orig_nostop
# out_folder=/mnt/data2/ptf/co_exp/orig
# out_folder=/mnt/data2/ptf/co_exp/lemma

mkdir -p $out_folder

res_dir=/mnt/data2/ptf/co_exp/lemma_nostop_hyper/res

mkdir -p $res_dir

hyperwords=/home/filter/hyperwords


CORPUS=lemma_nostop.txt
# CORPUS=orig_nostop.txt
# CORPUS=orig.txt
# CORPUS=lemma.txt
cp /mnt/data2/ptf/co_exp/$CORPUS $out_folder
#scripts/clean_corpus.sh $CORPUS > $CORPUS.clean
# mv $CORPUS $out_folder/$CORPUS

for window in {2..10}
do
for sub in {4..6}
do
        mkdir -p $out_folder/tmp
        python $hyperwords/hyperwords/corpus2pairs.py --win $window --dyn --sub 1e-$sub --del $out_folder/${CORPUS} > $out_folder/tmp/pairs
        $hyperwords/scripts/pairs2counts.sh $out_folder/tmp/pairs > $out_folder/tmp/counts
        python $hyperwords/hyperwords/counts2vocab.py $out_folder/tmp/counts

        # Calculate PMI matrices for each collection of pairs
        python $hyperwords/hyperwords/counts2pmi.py --cds 0.75 $out_folder/tmp/counts $out_folder/tmp/pmi

for dim in 300 400 500
do

        # Create embeddings with SVD
        python $hyperwords/hyperwords/pmi2svd.py --dim $dim --neg 5 $out_folder/tmp/pmi $out_folder/tmp/svd
        cp $out_folder/tmp/pmi.words.vocab $out_folder/tmp/svd.words.vocab
        cp $out_folder/tmp/pmi.contexts.vocab $out_folder/tmp/svd.contexts.vocab

        # Evaluate on Word Similarity
        echo
        echo "WS353 Results"
        echo "-------------"

        export PYTHONWARNINGS="ignore"
        set +x
        
        for neg in {1..10}
        do
        python $hyperwords/hyperwords/ws_eval.py --neg $neg PPMI $out_folder/tmp/pmi $hyperwords/testsets/de/ws/schm280.txt > $res_dir/pmi.$window.$sub.$dim.dyn.del.$neg.100.txt
        done
        
        for eig in 0.0 0.25 0.5 0.75 1.0
        do
        python $hyperwords/hyperwords/ws_eval.py --eig $eig SVD $out_folder/tmp/svd $hyperwords/testsets/de/ws/schm280.txt > $res_dir/svd.$window.$sub.$dim.dyn.del.$eig.100.txt
        done
done

rm -rf $out_folder/tmp

done
done
