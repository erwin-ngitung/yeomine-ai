import logging
import queue
import os
from twilio.rest import Client
from typing import List, NamedTuple
import av
import cv2
import numpy as np

# Package for Machine Learning
import torch
from ultralytics import YOLO

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

logger = logging.getLogger(__name__)


@st.cache_data
def get_ice_servers():
    """Use Twilio's TURN server because Streamlit Community Cloud has changed
    its infrastructure and WebRTC connection cannot be established without TURN server now.  # noqa: E501
    We considered Open Relay Project (https://www.metered.ca/tools/openrelay/) too,
    but it is not stable and hardly works as some people reported like https://github.com/aiortc/aiortc/issues/832#issuecomment-1482420656  # noqa: E501
    See https://github.com/whitphx/streamlit-webrtc/issues/1213
    """

    # Ref: https://www.twilio.com/docs/stun-turn/api
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Fallback to a free STUN server from Google."  # noqa: E501
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)
    token = client.tokens.create()

    return token.ice_servers


@st.cache_resource  # type: ignore
def generate_label_colors(name):
    return np.random.uniform(0, 255, size=(len(name), 3))


class Detection(NamedTuple):
    class_id: int
    label: str
    score: float


# Session-specific caching
cache_key = "object_detection"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = "front_coal"
    st.session_state[cache_key] = net

score_threshold = st.slider("Score threshold", 0.0, 1.0, 0.5, 0.05)

# NOTE: The callback will be called in another thread,
#       so use a queue here for thread-safety to pass the data
#       from inside to outside the callback.
# TODO: A general-purpose shared state object may be more useful.
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    model = YOLO("weights/general-detect/yolov8n.pt")

    # Run inference
    img = cv2.resize(img, (650, 650), interpolation=cv2.INTER_AREA)

    results = model.predict(img, conf=0.2, iou=0.7)
    names = model.names

    # Convert the output array into a structured form.
    detections = []
    colors = generate_label_colors(model.names)

    for i, confid in enumerate(results[0].boxes.conf.tolist()):
        if confid >= score_threshold:
            data = results[0].boxes.xyxy[i].tolist()
            label = names[int(results[0].boxes.cls[i])]
            color = colors[int(results[0].boxes.cls[i])]
            x1, y1, x2, y2 = int(data[0]), int(data[1]), int(data[2]), int(data[3])

            detections.append(
                Detection(
                    class_id=int(results[0].boxes.cls[i]),
                    label=label,
                    score=float(confid))
            )

            img = cv2.rectangle(img,
                                (x1, y1),
                                (x2, y2),
                                color, 2)
            img = cv2.putText(img,
                              label,
                              (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                              (36, 255, 12), 2)

    result_queue.put(detections)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False}
)

if st.checkbox("Show the detected labels", value=True):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        while True:
            result = result_queue.get()
            labels_placeholder.table(result)
