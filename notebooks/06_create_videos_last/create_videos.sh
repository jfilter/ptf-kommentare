#!/bin/bash
# for file in *.mp4; do
doit() {
	local file=$1 &&	echo $file
	local out=$(ffmpeg -threads 1 -v error -i "$file" -f null - 2>&1);
	if [ "$out" = ""  ]; then
		# echo $out
		# echo $file
		echo "looks good"
	else
		echo $out && echo $file >> problem.txt
		# echo $out && echo $file && rm $file
	fi
}
# done
export -f doit
parallel doit ::: final_videos/*.mp4
