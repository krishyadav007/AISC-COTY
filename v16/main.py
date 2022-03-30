import os
import time
from app import app
import infer
import mailer
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/ui')
def upload_ui():
	return render_template('ui.html')

@app.route('/index')
def show_index():
	return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	else:
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		print('upload_video filename: ' + filename)
		flash('Video successfully uploaded')
		flash('Started proccessing')
		return render_template('ui.html', filename=filename)

@app.route('/display/<filename>')
def display_video(filename):
	#print('display_video filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/mail/<lp_no>')
def mail_api(lp_no):
    mailer.send_mail(lp_no)
    return "All is well"

@app.route('/infer/<file_name>')
def infer_api(file_name):
    time_id = str(int(time.time()))
    infer.proccess(file_name, time_id=time_id) #Want to make this async
    return time_id

@app.route('/results/<time_id>')
def result_api(time_id):
    return infer.results(time_id)

if __name__ == "__main__":
    app.run()