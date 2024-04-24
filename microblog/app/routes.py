from flask import render_template, flash, redirect, url_for, request
import os
import pandas as pd
from app import app
from app.forms import LoginForm
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='Home')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        flash('Login requested for user {}, remember_me={}'.format(form.username.data, form.remember_me.data))
        return redirect(url_for('index'))
    return render_template('login.html', title='Sign In', form=form)
@app.route('/parse', methods=['POST'])
def submit_form():
    dsc = request.form.get('description')
    gender = request.form.get('gender')
    dsc = dsc.upper()
    # Process the form data
    return render_template("result.html",
                               dsc=dsc,
                               gender=gender)
@app.route('/upload', methods=['POST'])

def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:        
        
        file.save(file.filename)
        # Read the Excel file into a DataFrame
        df = pd.read_excel(file.filename)
        
        # Now you can use the DataFrame 'df' to work with the uploaded Excel data
        # For example, print the first few rows of the DataFrame
        print(df.head())
        return render_template('result.html', filename=file.filename, tables=[df.to_html(classes='data')], titles=df.columns.values)