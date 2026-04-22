
import time
import cv2


def create_gst_pipeline(rtsp_src):
    return f"""
            rtspsrc location={rtsp_src} latency=0 drop-on-latency=true !
            rtph265depay ! h265parse ! mppvideodec fast-mode=true !
            queue max-size-buffers=1 leaky=downstream !
            rgaconvert ! video/x-raw, format=BGR, width=4000, height=3000 ! appsink drop=1 max-buffers=1 sync=0
        """

def camera(rtsp_address):
    gst_pipeline = create_gst_pipeline(rtsp_src=rtsp_address)
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    out = rec_vid()
    
    try:
        while True:
            ts = time.monotonic()
            ret, frame = cap.read()
            # print(ret)
            # print(frame.shape, frame.dtype)
            # print(elps-ts)
            # cv2.rectangle(frame, (100,100), (400,400), (0,255,0), 3)
            frame = cv2.resize(frame, (1600,1200))
            out.write(frame)
            # cv2.imshow("online", frame)
            elps = time.monotonic()
            print(elps-ts)
            # if cv2.waitKey(1) & 0xFF == 27:
            #     break

    except:
        pass
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def create_gst_write_pipeline(fname):
    return f"""
            appsrc ! video/x-raw, format=BGR, width=1600, height=1200 !
            rgaconvert ! video/x-raw, format=NV12, width=1600, height=1200 !
            mpph265enc rc-mode=vbr bps=4000000 gop=60 !
            h265parse ! mp4mux ! filesink location={fname} sync=false
        """

def rec_vid():
    out = cv2.VideoWriter(create_gst_write_pipeline("file.mp4"), cv2.CAP_GSTREAMER, 0, 25, (1600,1200), True)
    return out


# camera("rtsp://admin:123456@192.168.168.52:554/stream1")
camera("rtsp://admin:admin@192.168.1.238:554/stream1")
