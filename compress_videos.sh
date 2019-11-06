for f in videos/*.MOV; 
    do ffmpeg -i "$f" -vcodec libx265 -crf 20 "${f%.*}_reduced.mp4"
done