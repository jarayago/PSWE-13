from flask import Flask, render_template, Response, request, redirect, session, abort, g
import cv2
from flask_misaka import Misaka
from src.supabase import (
    supabase,
    user_context_processor,
    get_profile_by_user
)
from src.decorators import login_required, password_update_required, profile_required
from src.auth import auth


app = Flask(__name__, template_folder="./templates", static_folder="./static")
Misaka(app)

# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = b"c8af64a6a0672678800db3c5a3a8d179f386e083f559518f2528202a4b7de8f8"
app.context_processor(user_context_processor)
app.register_blueprint(auth)

def generate_frames():
    if not camera.isOpened():
        raise RuntimeError("Error al abrir la cámara.")

    while True:
        success, frame = camera.read()
        if not success:
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            raise RuntimeError("Error al codificar el frame.")

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()



@app.teardown_appcontext
def close_supabase(e=None):
    g.pop("supabase", None)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/stream-video')
def stream():
    session = supabase.auth.refresh_session
    print(session)
    return render_template('streaming.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        users_email = request.form['email']
        users_password  = request.form['password']
        # Crear un nuevo usuario en Supabase
        user = supabase.auth.sign_up({'email':users_email, 'password':users_password })
        if user['status_code'] == 200:
            # Usuario creado con éxito, redirigir a la página de inicio
            return redirect('/')
        else:
            # Error al crear el usuario, mostrar mensaje de error
            return render_template('register.html', error=user['error']['message'])
    else:
        return render_template('register.html')

@app.route("/dashboard")
@login_required
@password_update_required
@profile_required
def dashboard():
    profile = get_profile_by_user()
    return render_template("dashboard.html", profile=profile)




if __name__ == "__main__":
    camera = cv2.VideoCapture(0)
    app.run(host='0.0.0.0', port=5000, debug=True)
