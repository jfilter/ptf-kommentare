#!/bin/bash
# set -e
set -x

out_folder=/mnt/data2/ptf/co_exp/lemma_nostop_hyper
# out_folder=/mnt/data2/ptf/co_exp/orig_nostop
# out_folder=/mnt/data2/ptf/co_exp/orig
# out_folder=/mnt/data2/ptf/co_exp/lemma

mkdir -p $out_folder

res_dir=/mnt/data2/ptf/co_exp/lemma_nostop_hyper/res2

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
for sub in {4..4}
do
        mkdir -p $out_folder/tmp$window$sub
        python $hyperwords/hyperwords/corpus2pairs.py --win $window --dyn --sub 1e-$sub --del $out_folder/${CORPUS} > $out_folder/tmp$window$sub/pairs
        $hyperwords/scripts/pairs2counts.sh $out_folder/tmp$window$sub/pairs > $out_folder/tmp$window$sub/counts
        python $hyperwords/hyperwords/counts2vocab.py $out_folder/tmp$window$sub/counts

        # Calculate PMI matrices for each collection of pairs
        python $hyperwords/hyperwords/counts2pmi.py --cds 0.75 $out_folder/tmp$window$sub/counts $out_folder/tmp$window$sub/pmi

for dim in 400 500
do
        # Create embeddings with SVD
        python $hyperwords/hyperwords/pmi2svd.py --dim $dim --neg 5 $out_folder/tmp$window$sub/pmi $out_folder/tmp$window$sub/svd$dim
        cp $out_folder/tmp$window$sub/pmi.words.vocab $out_folder/tmp$window$sub/svd$dim.words.vocab

        # Evaluate on Word Similarity
        echo
        echo "WS353 Results"
        echo "-------------"

        export PYTHONWARNINGS="ignore"
        set +x
        
        for testset in schm280.txt gur350.txt gur65.txt simlex999.txt ws353rel.txt ws353sim.txt zg222.txt; do 
        for neg in 1 2 3 5 10 15 20 25
        do
        python $hyperwords/hyperwords/ws_eval.py --neg $neg PPMI $out_folder/tmp$window$sub/pmi $hyperwords/testsets/de/ws/$testset > $res_dir/$testset.pmi.$window.$sub.$dim.dyn.del.$neg.100.txt
        done
        
        for eig in 0.0 0.1 0.2 0.3 0.4 0.5
        do
        python $hyperwords/hyperwords/ws_eval.py --eig $eig SVD $out_folder/tmp$window$sub/svd$dim $hyperwords/testsets/de/ws/$testset > $res_dir/$testset.svd.$window.$sub.$dim.dyn.del.$eig.100.txt
        done
done
done

# rm -rf $out_folder/tmp

done
done
