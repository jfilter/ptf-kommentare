#!/bin/bash
for file in videos/*.mp4; do
	out=$(mediainfo --Inform="Video;%Duration%" $file) &&
	if [ $out -lt 18000 ]; then
		echo $out && rm $file
	fi
done
