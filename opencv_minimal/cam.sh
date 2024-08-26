ffmpeg -list_devices true -f dshow -i dummy

ffplay -f dshow -i video="USB2.0 HD UVC WebCam"
ffplay -f vfwcap -i 0

ffmpeg -f dshow -i video="USB2.0 HD UVC WebCam" -vcodec libx264 -preset:v ultrafast -tune:v zerolatency -rtsp_transport tcp -f rtsp rtsp://127.0.0.1/test

ffmpeg.exe  -copyinkf -f dshow -i video="USB2.0 UVC VGA WebCam":audio="麦克风 (Realtek High Definition Au" -q 4 -s 640*480 -aspect 4:3 -r 10 -vcodec flv   -ar 22050 -ab 64k -ac 1 -acodec libmp3lame -threads 4 -f flv rtmp://127.0.0.1/RTMP/RtmpVideo
ffmpeg.exe -f dshow -i video="USB2.0 HD UVC WebCam":audio="麦克风 (Realtek(R) Audio)" -q 4 -s 640*480 -aspect 4:3 -r 10 -vcodec flv   -ar 22050 -ab 64k -ac 1 -acodec libmp3lame -threads 4 -rtbufsize 100M -f flv rtmp://127.0.0.1/RTMP/RtmpVideo

