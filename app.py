import os
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from werkzeug.utils import secure_filename
from inference import stream_inference

app = Flask(__name__)

UPLOAD_FOLDER = "static/videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CCTV_VIDEOS = [f"sample{i}.mp4" for i in range(1, 7)]

@app.route("/")
def index():
    return render_template("index.html", cctv_videos=CCTV_VIDEOS)

# WebM chunk streaming
@app.route("/stream/<video_name>")
def stream_video(video_name):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_name)
    return Response(stream_inference(video_path),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# Upload video and return streaming URL
@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    return jsonify({"stream_url": url_for("stream_video", video_name=filename)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
