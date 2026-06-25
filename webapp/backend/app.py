from flask import Flask, render_template, request, Response, jsonify, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import cv2
import base64
import numpy as np
import threading
import time
import random
import string
import re
from datetime import datetime
from functools import wraps
from model_manager import ModelManager
from detector import detect_image

app = Flask(__name__, template_folder="../frontend/templates", static_folder="../frontend/static")
app.secret_key = 'fire_system_2026_secret'

basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'fire_system.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


class FireLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    event_time = db.Column(db.DateTime, default=datetime.now)
    confidence = db.Column(db.Float)
    source = db.Column(db.String(50))


class InvitationCode(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(20), unique=True, nullable=False)
    is_used = db.Column(db.Boolean, default=False)


is_detecting = True
is_paused = False
current_status_text = "🟢 监控中"
current_alert_level = "safe"

stop_video_flags = {}

model_manager = ModelManager()
model_manager.load_model("best-11m-144.pt", "image")
model_manager.load_model("best-RT.pt",  "video")
model_manager.load_model("best-v8n.pt",  "camera")

UPLOAD_DIR = os.path.join(app.static_folder, "uploads")
UPLOAD_VIDEO_DIR = os.path.join(app.static_folder, "uploads", "videos")
RESULT_DIR = os.path.join(app.static_folder, "results")
RESULT_VIDEO_DIR = os.path.join(app.static_folder, "results", "videos")
for d in [UPLOAD_DIR, UPLOAD_VIDEO_DIR, RESULT_DIR, RESULT_VIDEO_DIR]:
    os.makedirs(d, exist_ok=True)


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session: return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


@app.route('/start_background_video', methods=['POST'])
def start_bg_video():
    data = request.json
    name = data.get("filename")
    current_time_sec = data.get("currentTime", 0.0)
    if name:
        video_path = os.path.join(UPLOAD_VIDEO_DIR, name)
        stop_video_flags[name] = False
        t = threading.Thread(target=process_video_background, args=(video_path, name, current_time_sec), daemon=True)
        t.start()
    return jsonify({"status": "started"})


@app.route('/stop_background_video', methods=['POST'])
def stop_bg_video():
    data = request.json
    name = data.get("filename")
    if name: stop_video_flags[name] = True
    return jsonify({"status": "stopping"})


@app.route('/video_save_status', methods=['GET'])
@login_required
def video_save_status():
    name = request.args.get("filename")
    still_running = name in stop_video_flags
    return jsonify({"done": not still_running})


def add_fire_record(conf, source_name):
    try:
        with app.app_context():
            new_log = FireLog(confidence=round(float(conf), 2), source=source_name, event_time=datetime.now())
            db.session.add(new_log)
            db.session.commit()
    except Exception as e:
        pass


@app.route("/")
@login_required
def index():
    return render_template("index.html", user=session.get('username'))


@app.route("/detect/image", methods=["POST"])
@login_required
def analyze_image():
    files = request.files.getlist("images")
    results_list = []
    if files:
        model = model_manager.get_model("image")
        for file in files:
            if file.filename == '': continue
            path = os.path.join(UPLOAD_DIR, file.filename)
            file.save(path)
            try:
                res_name = detect_image(model, path, RESULT_DIR)
            except:
                res_name = file.filename
            try:
                results = model.predict(path, conf=0.25, verbose=False)
                has_fire = len(results[0].boxes) > 0
            except:
                has_fire = False
            results_list.append({"result_img": res_name, "has_fire": has_fire})
            if has_fire: add_fire_record(results[0].boxes.conf[0].item(), f"图片: {file.filename}")
        return render_template("index.html", results_list=results_list, show="image", user=session.get('username'))
    return redirect(url_for('index'))


def process_video_background(video_path, filename, start_time_sec=0.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return

    cap.set(cv2.CAP_PROP_POS_MSEC, int(start_time_sec * 1000))

    model = model_manager.get_model("video")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 120: fps = 25.0

    base = os.path.splitext(filename)[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(RESULT_VIDEO_DIR, f"{base}_{timestamp}_result.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    try:
        while cap.isOpened():
            if stop_video_flags.get(filename, False): break

            success, frame = cap.read()
            if not success: break

            results = model.predict(frame, conf=0.45, imgsz=256, verbose=False)

            if results and len(results[0].boxes) > 0:
                annotated = results[0].plot()
            else:
                annotated = frame

            writer.write(annotated)
    finally:
        writer.release()
        cap.release()
        stop_video_flags.pop(filename, None)


@app.route("/analyze/video", methods=["POST"])
@login_required
def analyze_video():
    file = request.files.get("video")
    if file:
        video_path = os.path.join(UPLOAD_VIDEO_DIR, file.filename)
        file.save(video_path)
        return render_template("index.html", video_to_detect=file.filename, show="video", user=session.get('username'))
    return redirect(url_for('index'))


@app.route("/video_feed")
@login_required
def video_stream():
    name = request.args.get('filename')
    return Response(generate_frames(os.path.join(UPLOAD_VIDEO_DIR, name), name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/toggle_video_pause', methods=['POST'])
def toggle_video_pause():
    global is_paused
    is_paused = not is_paused
    return jsonify({"paused": is_paused})


def generate_frames(source, status_key):
    global is_paused, current_status_text, current_alert_level

    if source == 0:
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(source)

    model = model_manager.get_model("camera")
    fire_counter = 0
    last_db_time = 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0 or height == 0: width, height = 640, 480

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = "camera" if source == 0 else "stream"
    save_path = os.path.join(RESULT_VIDEO_DIR, f"{base_name}_{timestamp}_result.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    target_fps = 15.0
    frame_duration = 1.0 / target_fps
    writer = cv2.VideoWriter(save_path, fourcc, target_fps, (width, height))

    next_write_time = time.time() + frame_duration

    try:
        while cap.isOpened():
            if is_paused:
                time.sleep(0.1)
                next_write_time = time.time() + frame_duration
                continue

            if source == 0: cap.grab()
            success, frame = cap.retrieve() if source == 0 else cap.read()
            if not success: break

            annotated = frame.copy()
            last_result = model.predict(frame, conf=0.45, imgsz=256, verbose=False)

            if last_result and len(last_result[0].boxes) > 0:
                fire_counter += 1
                if fire_counter >= 3:
                    current_status_text = "🔴 发现火情！"
                    current_alert_level = "alert"
                    if time.time() - last_db_time > 5:
                        box = last_result[0].boxes[0]
                        with app.app_context():
                            db.session.add(FireLog(confidence=float(box.conf[0]), source=f"{status_key}"))
                            db.session.commit()
                        last_db_time = time.time()

                if fire_counter >= 5:
                    current_alert_level = "critical"
            else:
                fire_counter = 0
                current_status_text = "🟢 监控中"
                current_alert_level = "safe"

            if fire_counter > 0 and last_result and len(last_result[0].boxes) > 0:
                for box in last_result[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0]) if hasattr(box, 'cls') else 0
                    cls_name = model.names.get(cls_id, 'fire')
                    color = (0, 255, 255) if cls_name == 'fire' else (255, 0, 0)

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls_name} {conf:.2f}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (x1, y1 - 20), (x1 + w, y1), color, -1)
                    cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            now = time.time()
            while next_write_time <= now:
                writer.write(annotated)
                next_write_time += frame_duration

            ret, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ret:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    except GeneratorExit:
        pass
    finally:
        writer.release()
        cap.release()


@app.route("/camera_feed")
@login_required
def camera_feed():
    return Response(generate_frames(0, "live_camera"), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/check_status/<path:name>')
def check_status(name):
    return jsonify({"status": current_alert_level})


@app.route('/analyze_frame', methods=['POST'])
@login_required
def analyze_frame():
    try:
        data = request.json.get('image')
        if not data: return jsonify({"boxes": [], "has_fire": False})

        img_data = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        model = model_manager.get_model("video")
        results = model.predict(frame, conf=0.3, verbose=False)
        boxes = []
        has_fire = False
        max_conf = 0

        for box in results[0].boxes:
            has_fire = True
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, 'fire')
            boxes.append({"bbox": [float(x) for x in coords], "conf": conf, "cls": cls_name})
            if conf > max_conf: max_conf = conf

        if has_fire:
            last_log = FireLog.query.order_by(FireLog.id.desc()).first()
            now = datetime.now()
            if not last_log or (now - last_log.event_time).total_seconds() > 3:
                new_log = FireLog(event_time=now, confidence=max_conf,
                                  source=session.get('current_video', 'Live_Video'))
                db.session.add(new_log)
                db.session.commit()

        return jsonify({"boxes": boxes, "has_fire": has_fire})
    except Exception as e:
        return jsonify({"boxes": [], "has_fire": False, "error": str(e)})


@app.route('/models', methods=['GET'])
@login_required
def list_models():
    mode = request.args.get("mode", "image")
    models = model_manager.list_models()
    return jsonify({"models": models, "current": model_manager.current_name(mode), "mode": mode})


@app.route('/switch_model', methods=['POST'])
@login_required
def switch_model():
    data = request.json
    model_name = data.get("model_name", "").strip()
    mode = data.get("mode", "image")
    if not model_name:
        return jsonify({"success": False, "error": "未指定模型名称"})
    try:
        model_manager.switch_model(model_name, mode)
        return jsonify({"success": True, "current": model_manager.current_name(mode), "mode": mode})
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"success": False, "error": str(e)})
    except Exception as e:
        return jsonify({"success": False, "error": f"切换失败: {str(e)}"})


@app.route('/history')
@login_required
def history():
    logs = FireLog.query.order_by(FireLog.event_time.desc()).all()
    return render_template("history.html", logs=logs, user=session.get('username'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_code = request.form.get('invitation_code')
        username = request.form.get('username')
        password = request.form.get('password')
        valid_code = InvitationCode.query.filter_by(code=user_code, is_used=False).first()
        if not valid_code:
            flash("邀请码无效或已被使用")
            return render_template("register.html")
        if not username or len(username) < 2:
            flash("用户名太短")
            return render_template("register.html")
        if not re.match(r'^(?=.*[0-9])(?=.*[a-z])(?=.*[A-Z]).{8,}$', password):
            flash("密码必须包含大小写字母和数字，且至少8位")
            return render_template("register.html")

        db.session.add(User(username=username, password=generate_password_hash(password)))
        valid_code.is_used = True
        db.session.commit()
        flash("注册成功！")
        return redirect(url_for('login'))
    return render_template("register.html")


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            session['user_id'], session['username'] = user.id, user.username
            return redirect(url_for('index'))
        flash("用户名或密码错误")
    return render_template("login.html")


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        new_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        db.session.add(InvitationCode(code=new_code))
        db.session.commit()
        print(f"\n🚀 系统已就绪！新增邀请码: {new_code}\n")
    app.run(host='0.0.0.0', port=5000, debug=True)